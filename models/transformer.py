"""
VisTR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
Modified from DETR (https://github.com/facebookresearch/detr)
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.utils import checkpoint
from .deformable_transformer import DeformableTransformerEncoderLayer, DeformableTransformerEncoder
from models.ops.modules import MSDeformAttn
from transformers import RobertaModel, RobertaTokenizerFast
from .position_encoding import PositionEmbeddingSine2D

class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, freeze_text_encoder = True, ref_exp=False):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = False
        self.two_stage_num_proposals = 300
        self.ref_exp = ref_exp
        dec_n_points=8
        enc_n_points=6
        dropout = 0.1
        num_feature_levels = 36
        self.no_of_exps, self.id_dim  = 8, 20

        num_feature_levels = num_feature_levels + 1
        

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        # encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
        #                                         dropout, activation, normalize_before)
        # encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        # self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)
        self.instance_encoders = nn.Embedding(self.id_dim*self.no_of_exps, d_model)
        self.txt_pos = PositionEmbeddingSine2D(d_model//2, normalize=True)

        self.no_of_exps
        self._reset_parameters()

        if ref_exp :
            self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
            self.text_encoder = RobertaModel.from_pretrained('roberta-base')
            

            if freeze_text_encoder:
                for p in self.text_encoder.parameters():
                    p.requires_grad_(False)

            self.expander_dropout = 0.1
            txt_input_hidden_size = 768
            self.resizer = FeatureResizer(
                input_feat_size=txt_input_hidden_size,
                output_feat_size=d_model,
                dropout=self.expander_dropout,
            )

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

    def get_valid_ratio(self, mask):
        #print(f'mask {mask.shape}')
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, src, mask, query_embed, pos_embed, mask_, exps_input_ids=None, exps_attn_masks=None):
        # flatten NxCxHxW to HWxNxC
        _, s_h, s_w = mask_.shape
        #print(f'mask_ {mask_.shape}')
        spatial_shapes = []
        masks_ = []
        for idd in range(36) :
            spatial_shapes.append((s_h,s_w))
            #print(f'mask_[idd] {mask_[idd].unsqueeze(0).shape}') #mask_[idd] torch.Size([1, 10, 13])
            masks_.append(mask_[idd].unsqueeze(0))

        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1) #query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        device = src.device
        if self.ref_exp :
            
            no_of_exps = exps_input_ids.shape[0]
            #print(f'exps_input_ids {exps_input_ids.shape} exps_attn_masks {exps_attn_masks.shape}') #exps_input_ids torch.Size([1, 20]) exps_attn_masks torch.Size([1, 20]) # multi - exps_input_ids torch.Size([4, 20]) exps_attn_masks torch.Size([4, 20])
            tokenized = {"input_ids":(exps_input_ids.to(device)), "attention_mask":exps_attn_masks.to(device)}
            #print(f'tokenized {tokenized}')
            encoded_text = self.text_encoder(**tokenized)
            #print(f'encoded_text {encoded_text.last_hidden_state.shape}') encoded_text torch.Size([1, 20, 768])
            text_memory = encoded_text.last_hidden_state.transpose(0, 1)
            #print(f'text_memory {text_memory.shape}') #text_memory torch.Size([20, 1, 768]) # multi - torch.Size([20, 4, 768])
            text_attention_mask = tokenized['attention_mask'].ne(1).bool()
            #print(f'text_attention_mask {text_attention_mask.shape}') # text_attention_mask torch.Size([1, 20]) # multi - text_attention_mask torch.Size([4, 20])
            text_memory_resized = self.resizer(text_memory) #text_memory_resized torch.Size([20, 1, 384])
            #print(f' Refsrc {src.shape} mask {mask.shape} pos_embed {pos_embed.shape}') #src torch.Size([4680, 1, 384]) mask torch.Size([1, 4680]) pos_embed torch.Size([4680, 1, 384])
        instance_attn_mask = torch.zeros(self.no_of_exps, self.id_dim).to(device).bool()
        src = torch.cat([src, self.instance_encoders.weight.unsqueeze(1)], dim=0)
        mask = torch.cat([mask, instance_attn_mask.view(1, -1)], dim=1)
        txt_pos_encoding = self.txt_pos(self.instance_encoders.weight, instance_attn_mask.unsqueeze(0))
        #print(f'txt_pos {txt_pos_encoding.shape}') # txt_pos torch.Size([1, 384, 8, 20])
        pos_embed = torch.cat([pos_embed, txt_pos_encoding.permute(0,2,3,1).view(-1,1,384) ], dim=0)
        #print(f'src {src.shape} mask {mask.shape} pos_embed {pos_embed.shape}')

        #for idd in range(exps_input_ids.shape[0]) :
        spatial_shapes.append((self.no_of_exps,self.id_dim))
        masks_.append(instance_attn_mask.unsqueeze(0))

        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks_], 1)

        tgt = torch.zeros_like(query_embed)
        memory = checkpoint.checkpoint(self.encoder,src.permute(1,0,2), spatial_shapes, level_start_index, valid_ratios, pos_embed.permute(1,0,2), mask)
        #self.encoder(src.permute(1,0,2), spatial_shapes, level_start_index, valid_ratios, pos_embed.permute(1,0,2), mask) #self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        memory = memory.permute(1,0,2)
        # hs_exps = []
        # print(f'text_memory_resized {text_memory_resized.shape}')
        # word_length = text_memory_resized.shape[0] #20
        # for each_exp in range(no_of_exps) :
        #     new_memory = torch.cat([memory[:-word_length*no_of_exps,:,:], memory[-word_length*(no_of_exps-each_exp):-word_length*(no_of_exps-each_exp-1),:,:]], dim=0)
        #     new_mask =  torch.cat([mask[:,:-word_length*no_of_exps], mask[:, -word_length*(no_of_exps-each_exp):-word_length*(no_of_exps-each_exp-1)]], dim=1)
        #     new_pos_embed = torch.cat([pos_embed[:-word_length*no_of_exps,:,:], pos_embed[-word_length*(no_of_exps-each_exp):-word_length*(no_of_exps-each_exp-1),:,:]], dim=0)
        #     hs = self.decoder(tgt, new_memory, memory_key_padding_mask=new_mask,
        #                     pos=new_pos_embed, query_pos=query_embed)
        #     hs_exps.append(hs)
        # hs = torch.cat(hs_exps, dim=1)
        #print(f'no_of_exps {no_of_exps}')
        no_of_exps_tensor = torch.tensor(self.no_of_exps).to(device)
        #print(f'tgt {tgt.shape} query_pos {query_embed.shape}') tgt torch.Size([360, 1, 384]) query_pos torch.Size([360, 1, 384])
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                             pos=pos_embed, query_pos=query_embed, no_of_exps=no_of_exps_tensor)
        hs = hs.view(6, -1, 1, 384)
        #print(f'memory {memory.shape}') memory torch.Size([4680, 1, 384])
        #if self.ref_exp :
        memory = memory[:-self.no_of_exps*self.id_dim,:,:]
        #print(f'memory {memory.shape}') #memory torch.Size([4680, 1, 384])
        #print(f'hs {hs.shape} memory {memory.shape}') #hs torch.Size([6, 360, 1, 384]) memory torch.Size([4680, 1, 384])
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            '''output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)'''
            output = checkpoint.checkpoint(layer, output, mask, src_key_padding_mask, pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None, no_of_exps: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            '''output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)'''
            output = checkpoint.checkpoint(layer, output, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask,pos, query_pos, no_of_exps )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None, no_of_exps: Optional[Tensor] = None):
        #tgt torch.Size([360, 1, 384]) memory torch.Size([4760, 1, 384])  memory_key_padding_mask torch.Size([1, 4760]) pose torch.Size([4760, 1, 384]) query_pos torch.Size([36print(f' tgt {tgt.shape} memory {memory.shape}  memory_key_padding_mask {memory_key_padding_mask.shape} pose {pos.shape} query_pos {query_pos.shape} ')
        if no_of_exps.item() != tgt.shape[1] :
            tgt = tgt.repeat(1, no_of_exps, 1)
        query_pos = query_pos.repeat(1, no_of_exps, 1)
        word_length = 20
        
        tgt_all, query_pos_all, tgt_mask_all, tgt_key_padding_mask_all = tgt, query_pos, tgt_mask, tgt_key_padding_mask
        final_target_all = []
        sub_memory = memory[:-word_length*no_of_exps,:,:]
        for each_exp in range(no_of_exps) :

            tgt, query_pos, tgt_mask, tgt_key_padding_mask = tgt_all[:, each_exp:each_exp+1, :], query_pos_all[:, each_exp:each_exp+1, :], tgt_mask_all, tgt_key_padding_mask_all
            #print(f' tgt {tgt.shape} memory {memory.shape}  memory_key_padding_mask {memory_key_padding_mask.shape} pose {pos.shape} query_pos {query_pos.shape} ')
            q = k = self.with_pos_embed(tgt, query_pos)
            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                                    key_padding_mask=tgt_key_padding_mask)[0]
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)
            #for _ in range(3) :
            
                #print(f'range {-word_length*(no_of_exps-each_exp)} {-word_length*(no_of_exps-each_exp-1)} {memory[:-word_length*no_of_exps,:,:].shape} {memory[sub_memory.shape[0] + word_length*(each_exp): sub_memory.shape[0] +word_length*(each_exp+1), :, :].shape}')
            new_memory = torch.cat([sub_memory, memory[sub_memory.shape[0] + word_length*(each_exp):sub_memory.shape[0] +word_length*(each_exp+1),:,:]], dim=0)
            new_memory_key_padding_mask =  torch.cat([memory_key_padding_mask[:,:-word_length*no_of_exps], memory_key_padding_mask[:, sub_memory.shape[0] + word_length*(each_exp): sub_memory.shape[0] +word_length*(each_exp+1)]], dim=1)
            new_pos = torch.cat([pos[:-word_length*no_of_exps,:,:], pos[sub_memory.shape[0] + word_length*(each_exp): sub_memory.shape[0] +word_length*(each_exp+1),:,:]], dim=0)
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                    key=self.with_pos_embed(new_memory, new_pos),
                                    value=new_memory, attn_mask=memory_mask,
                                    key_padding_mask=new_memory_key_padding_mask)[0]
            
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
            tgt = tgt + self.dropout3(tgt2)
            tgt = self.norm3(tgt)
            final_target_all.append(tgt)

        return torch.cat(final_target_all, dim=1)

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None, no_of_exps: Optional[Tensor] = None):
        #tgt torch.Size([360, 1, 384]) memory torch.Size([4760, 1, 384])  memory_key_padding_mask torch.Size([1, 4760]) pose torch.Size([4760, 1, 384]) query_pos torch.Size([36print(f' tgt {tgt.shape} memory {memory.shape}  memory_key_padding_mask {memory_key_padding_mask.shape} pose {pos.shape} query_pos {query_pos.shape} ')
        if (no_of_exps.item()) != tgt.shape[1] :
            tgt = tgt.repeat(1, no_of_exps, 1)
        query_pos = query_pos.repeat(1, no_of_exps, 1)
        word_length = 20

        tgt_all, query_pos_all, tgt_mask_all, tgt_key_padding_mask_all = tgt, query_pos, tgt_mask, tgt_key_padding_mask
        final_target_all = []
        sub_memory = memory[:-word_length*no_of_exps,:,:]
        for each_exp in range(no_of_exps) :
            tgt, query_pos, tgt_mask, tgt_key_padding_mask = tgt_all[:, each_exp:each_exp+1, :], query_pos_all[:, each_exp:each_exp+1, :], tgt_mask_all, tgt_key_padding_mask_all
            
            tgt2 = self.norm1(tgt)
            q = k = self.with_pos_embed(tgt2, query_pos)
            tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)[0]
            tgt = tgt + self.dropout1(tgt2)
            tgt2 = self.norm2(tgt)

            new_memory = torch.cat([sub_memory, memory[sub_memory.shape[0] + word_length*(each_exp):sub_memory.shape[0] +word_length*(each_exp+1),:,:]], dim=0)
            new_memory_key_padding_mask =  torch.cat([memory_key_padding_mask[:,:-word_length*no_of_exps], memory_key_padding_mask[:, sub_memory.shape[0] + word_length*(each_exp): sub_memory.shape[0] +word_length*(each_exp+1)]], dim=1)
            new_pos = torch.cat([pos[:-word_length*no_of_exps,:,:], pos[sub_memory.shape[0] + word_length*(each_exp): sub_memory.shape[0] +word_length*(each_exp+1),:,:]], dim=0)
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                    key=self.with_pos_embed(new_memory, new_pos),
                                    value=new_memory, attn_mask=memory_mask,
                                    key_padding_mask=new_memory_key_padding_mask)[0]
        #for _ in range(3) :
        
            tgt = tgt + self.dropout2(tgt2)
            tgt2 = self.norm3(tgt)
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
            tgt = tgt + self.dropout3(tgt2)
            final_target_all.append(tgt)

        return torch.cat(final_target_all, dim=1)


    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None, no_of_exps: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, no_of_exps)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, no_of_exps)

class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
