'''
Inference code for VisTR
Modified from DETR (https://github.com/facebookresearch/detr)
'''
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
from PIL import Image
import math
import torch.nn.functional as F
import json
from scipy.optimize import linear_sum_assignment
import pycocotools.mask as mask_util
from sentence_transformers import SentenceTransformer
import cv2
import copy, transformers
from transformers import RobertaTokenizerFast


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--lr_drop', default=100, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--sgd', action='store_true')
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')
    # Model parameters
    parser.add_argument('--model_path', type=str, default=None,
                        help="Path to the model weights.")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_frames', default=36, type=int,
                        help="Number of frames")
    parser.add_argument('--num_ins', default=12, type=int,
                        help="Number of instances")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--img_path', default='/mnt/data/valid_rvos_final/JPEGImages/')
    parser.add_argument('--ann_path', default='/mnt/data/ytvis/annotations/instances_val_sub.json')
    parser.add_argument('--save_path', default='results.json')
    parser.add_argument('--dataset_file', default='ytvos')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='output_ytvos',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    #parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval', action='store_false')
    parser.add_argument('--num_workers', default=0, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    #parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser

CLASSES=['person','giant_panda','lizard','parrot','skateboard','sedan','ape',
         'dog','snake','monkey','hand','rabbit','duck','cat','cow','fish',
         'train','horse','turtle','bear','motorbike','giraffe','leopard',
         'fox','deer','owl','surfboard','airplane','truck','zebra','tiger',
         'elephant','snowboard','boat','shark','mouse','frog','eagle','earless_seal',
         'tennis_racket']
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933],
          [0.494, 0.000, 0.556], [0.494, 0.000, 0.000], [0.000, 0.745, 0.000],
          [0.700, 0.300, 0.600]]
transform = T.Compose([
    T.Resize(300),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b.cpu() * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def exp_for_video(exps_objects, tokenizer) :
    expressions = []
    exps_input_ids = []
    exps_attn_masks = []
    max_tokens = 20
    for each_item in exps_objects.keys() :
        #print(f'{exps_objects[each_item]["expressions"][0]} ')
        exp = exps_objects[each_item]["exp"]
        expressions.append(exp)
        attention_mask = [0] * max_tokens
        padded_input_ids = [0] * max_tokens
        tokenized = tokenizer(exp)
        token_len = min(len(tokenized["input_ids"]), max_tokens-1)
        padded_input_ids[:token_len] = tokenized["input_ids"][:token_len]
        attention_mask[:token_len] = tokenized["attention_mask"][:token_len]
        #print(f'padded_input_ids {len(tokenized["input_ids"])} {len(padded_input_ids)}')
        exps_input_ids.append(torch.tensor(padded_input_ids).unsqueeze(0))
        #print(f'torch.tensor(padded_input_ids).unsqueeze(0) {torch.tensor(padded_input_ids).unsqueeze(0).shape}')
        exps_attn_masks.append(torch.tensor(attention_mask).unsqueeze(0))

    exp_ct = len(expressions)
    if exp_ct > 12 :
        print(expressions)
    while(1) :
        if exp_ct >= 12 :
            break
        exp = ""
        expressions.append(exp)
        attention_mask = [0] * max_tokens
        padded_input_ids = [0] * max_tokens
        tokenized = tokenizer(exp)
        token_len = min(len(tokenized["input_ids"]), max_tokens)
        padded_input_ids[:token_len] = tokenized["input_ids"][:token_len]
        attention_mask[:token_len] = tokenized["attention_mask"][:token_len]
        exps_input_ids.append(torch.tensor(padded_input_ids).unsqueeze(0))
        exps_attn_masks.append(torch.tensor(attention_mask).unsqueeze(0))
        #print(f'torch.tensor(padded_input_ids).unsqueeze(0) {torch.tensor(padded_input_ids).unsqueeze(0).shape}')
        exp_ct = exp_ct + 1
    # for frame in range(36) :
    #     ct = 0
    #     for exp in expressions :
    #         new_expressions.append(exp)
    #         ct = ct + 1
    #     while ct < 12 :
    #         new_expressions.append("")
    #         ct = ct + 1
    
    return torch.cat(exps_input_ids, dim=0), torch.cat(exps_attn_masks, dim=0), expressions

def get_rand_color():
    c = ((np.random.random((3)) * 0.6 + 0.2) * 255).astype(np.int32).tolist()
    return c

def main(args):

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed) 
    num_frames = args.num_frames
    num_ins = args.num_ins
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')#transformers.BertTokenizer.from_pretrained('bert-base-uncased')
    #bert_model = transformers.BertModel.from_pretrained('/mnt/data/exps/bert_pretrained_refcoco')
    saves = 0
    with torch.no_grad():
        model, criterion, postprocessors = build_model(args)
        model.to(device)
        state_dict = torch.load(args.model_path)['model']
        model.load_state_dict(state_dict)
        folder = args.img_path
        videos = json.load(open(args.ann_path,'rb'))['videos']
        vis_num = len(videos.keys())
        print(vis_num)
        result = [] 
        ct = 0
        total = 0
        for id_ in videos.keys():
            print("Process video: ",ct)
            ct = ct + 1
            #if ct <= 86 : continue
            length = len(videos[id_]['frames'])
            file_names = videos[id_]['frames']
            expressions_ = videos[id_]['expressions']

            # if total != saves :
            #     print(f'total != saves total {total} save {saves}')
            #     exit()
            total += length
            #print(f'file_names {file_names} expressions {expressions}')
            clip_num = math.ceil(length/num_frames)

            words_exps, words_mask, expressions = exp_for_video(expressions_, tokenizer)
            print(F'words_exps {words_exps.shape} {words_mask.shape}')
            instance_ct = len(expressions_.keys())
            #print(f'encoded_exps {encoded_exps.shape}')
            #exit()
            
            img_set=[]
            if length<num_frames:
                clip_names = file_names*(math.ceil(num_frames/length))
                clip_names = clip_names[:num_frames]
            else:
                clip_names = file_names[:num_frames]
            if len(clip_names)==0:
                continue
            if len(clip_names)<num_frames:
                clip_names.extend(file_names[:num_frames-len(clip_names)])
            names = []
            for k in range(num_frames):
                #im = cv2.imread(os.path.join(folder, id_, f'{clip_names[k]}.jpg'))
                im = Image.open(os.path.join(folder, id_, f'{clip_names[k]}.jpg'))
                print(f"{folder, id_, f'{clip_names[k]}.jpg'}")
                names.append(clip_names[k])
                img_set.append(transform(im).unsqueeze(0).cuda())
            img=torch.cat(img_set,0)
            print(f'img.shape {img.shape} words_exps {words_exps.shape} words_mask {words_mask.shape}')
            # inference time is calculated for this operation
            outputs = model(img, words_exps.unsqueeze(0).repeat(36,1,1).cuda(), words_mask.unsqueeze(0).repeat(36,1,1).cuda())
            # end of model inference
            logits, boxes, = outputs['pred_logits'].softmax(-1), outputs['pred_boxes']#, outputs['pred_masks'][0]
            
            #pred_masks =F.interpolate(masks.reshape(num_frames,num_ins,masks.shape[-2],masks.shape[-1]),(im.size[1],im.size[0]),mode="bilinear").sigmoid().cpu().detach().numpy()>0.5
            #pred_logits = logits.reshape(num_frames,num_ins,logits.shape[-1]).cpu().detach().numpy()
            #pred_masks = pred_masks[:length] 
            #pred_logits = pred_logits[:length]
            #pred_scores = np.max(pred_logits,axis=-1)
            #pred_logits = np.argmax(pred_logits,axis=-1)

            print(f'boxes {boxes.shape}')
            colors = []
            for i in range(300):
                colors.append(get_rand_color())
            for n in range(length):
                print(f'img {img.shape}')
                im = cv2.imread(os.path.join(folder, id_, f'{clip_names[n]}.jpg'))
                #im = cv2.UMat(np.array(img.permute(0,2,3,1)[n,:,:,:].cpu().detach().numpy(), dtype=np.uint8))
                print(f'im {im.shape} type im {type(im)}')
                h, w, _ = im.shape
                folder_ = id_
                #im = cv2.imread(os.path.join(folder, id_, f'{clip_names[n]}.jpg'))
                for m_ in range(instance_ct):
                    print(f'logits {logits.shape} {logits[n,m_*25:(m_+1)*25,:].shape}')
                    res, _ = logits[n,m_*25:(m_+1)*25,:].max(dim=1)
                    print(f'res {res.shape}')
                    _, indices = torch.max(res,dim=0)
                    print(indices)
                    m = int(indices.detach().cpu().numpy()) + m_*25
                    #if logits[n][m][0] <= 0.5 :
                    #    continue
                    #if pred_masks[:,m].max()==0 and m != 0:
                    #    continue
                    #score = pred_scores[:,m].mean()
                    #category_id = pred_logits[:,m][pred_scores[:,m].argmax()]
                    #category_id = np.argmax(np.bincount(pred_logits[:,m]))
                    obj_id = m
                    instance = {'video_id':id_,  }
                    segmentation = []
                    
                        # if pred_scores[n,m]<0.001:
                        #     segmentation.append(None)
                        # else:
                    box = (boxes[n][m]).tolist()
                    #mask = (pred_masks[n,m]*255).astype(np.uint8) 
                    #im[mask] = im[mask] * 0.2 + np.array(colors[m]) * 0.8
                    #rle = mask_util.encode(np.array(mask[:,:,np.newaxis], order='F'))[0]
                    #rle["counts"] = rle["counts"].decode("utf-8")
                    #segmentation.append(rle)
                    #mask = (pred_masks[n,m]*255).astype(np.uint8)
                    name_ = names[n]#clip_names[n]
                    folder_ = id_
                    print(f'clip_names[n] {name_}')
                    print(f"/mnt/data/Visualize/Visualize_VOS_word/{folder_}/{m}")
                    left, top = int(box[0]*w - box[2]*0.5*w), int(box[1]*h - box[3]*0.5*h)
                    right, bottom = int(box[0]*w + box[2]*0.5*w), int(box[1]*h + box[3]*0.5*h)
                    cv2.rectangle(im, (left, top), (right, bottom), (colors[m][0],colors[m][1],colors[m][2]), 4)
                    x, y =  left, top
                    cv2.rectangle(im, (x-10, y-20), (x+150, y+6), (255, 255, 255), -1)
                    #print(f'expressions_ {expressions}')
                    cv2.putText(im, expressions[m_], (x, y + random.randint(-10,10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[m], thickness=2, lineType=cv2.LINE_AA)
                    #img = Image.fromarray(mask)
                    
                    #img.save(f'/home/ubuntu/codes/Visualize_VOS/{folder_}_{m}_{name_}.png')
                    if m == 0 :
                        saves += 1
                    #instance['segmentations'] = segmentation
                    #result.append(instance)
                #os.makedirs(f"/home/ubuntu/codes/Visualize_VOS/{folder_}/", exist_ok=True)
                cv2.imwrite(f'/mnt/data/Visualize/Visualize_VOS_word/{folder_}_{n}.png', im)
        print(f'total {total}')
    #with open(args.save_path, 'w', encoding='utf-8') as f:
    #    json.dump(result,f)
                    
        



if __name__ == '__main__':
    parser = argparse.ArgumentParser('VisTR inference script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
