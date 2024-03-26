"""
Instance Sequence Matching
Modified from DETR (https://github.com/facebookresearch/detr)
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, multi_iou
INF = 100000000

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, num_frames : int = 36, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.num_frames = num_frames
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the sequence level matching
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        indices = []

        #bs = len(targets[0]["labels"])//self.num_frames
        #print(f'outputs["pred_logits"] {outputs["pred_logits"].shape} {outputs["pred_boxes"].shape}') #outputs["pred_logits"] torch.Size([1, 360, 42]) torch.Size([1, 360, 4])
        num_queries = 36
        outputs_pred_logits = outputs["pred_logits"].view(-1,num_queries,42)
        outputs_pred_boxes = outputs["pred_boxes"].view(-1,num_queries,4)
        bs = outputs_pred_logits.shape[0]

        new_targets = []
        #print(f'targets[0]["labels"] {len(targets[0]["labels"])} {len(targets[0]["labels"])}')
        for gts_ct in range(bs) :
            new_targets.append({"labels" : targets[0]["labels"], "boxes" :targets[0]["boxes"], "valid" : targets[0]["valid"]})
            #print(new_targets[gts_ct])
        
        #targets = new_targets 
        #new_targets.append(targets[0])
        #new_targets.append(targets[0])
        #print(f'len(targets[0]["labels"]) {len(targets[0]["labels"])} targets[i]["labels"]  {len(new_targets)} bs {bs}')
        index_i,index_j = [],[]
        new_cost = []
        for i in range(bs):
            #print(f'i --> {i}')
            out_prob = outputs_pred_logits[i].softmax(-1)
            out_bbox = outputs_pred_boxes[i]
            tgt_ids = new_targets[i]["labels"]
            tgt_bbox = new_targets[i]["boxes"]
            tgt_valid = new_targets[i]["valid"]
            #print(f'tgt_ids {len(tgt_ids)}')
            num_out = 1
            num_tgt = len(tgt_ids)//self.num_frames
            out_prob_split = out_prob.reshape(self.num_frames,num_out,out_prob.shape[-1]).permute(1,0,2)
            out_bbox_split = out_bbox.reshape(self.num_frames,num_out,out_bbox.shape[-1]).permute(1,0,2).unsqueeze(1)
            tgt_bbox_split = tgt_bbox.reshape(num_tgt,self.num_frames,4).unsqueeze(0)
            tgt_valid_split = tgt_valid.reshape(num_tgt,self.num_frames)
            frame_index = torch.arange(start=0,end=self.num_frames).repeat(num_tgt).long()
            class_cost = -1 * out_prob_split[:,frame_index,tgt_ids].view(num_out,num_tgt,self.num_frames).mean(dim=-1)
            bbox_cost = (out_bbox_split-tgt_bbox_split).abs().mean((-1,-2))
            iou_cost = -1 * multi_iou(box_cxcywh_to_xyxy(out_bbox_split),box_cxcywh_to_xyxy(tgt_bbox_split)).mean(-1)
            #TODO: only deal with box and mask with empty target
            cost = self.cost_class*class_cost + self.cost_bbox*bbox_cost + self.cost_giou*iou_cost
            new_cost.append(cost.unsqueeze(0))
        cost = torch.cat(new_cost,dim=0)
        #print(f'cost {cost.shape}')
        cost_values, cost_indices = torch.max(cost, dim=1)
        #print(f'cost_values {cost_values.shape} cost_indices {cost_indices}')
        
        out_i, tgt_i = linear_sum_assignment(cost_values.cpu())
        #print(f'idx  len(out_i) {len(out_i)}')
        for j in range(len(out_i)):
            tgt_valid_ind_j = tgt_valid_split[tgt_i[j]].nonzero().flatten()
            index_i.append(tgt_valid_ind_j*num_out +  int(cost_indices[0][j].item()) +out_i[j]*num_queries)
            index_j.append(tgt_valid_ind_j + tgt_i[j]* self.num_frames )
        if index_i==[] or index_j==[]:
            indices.append((torch.tensor([]).long().to(out_prob.device),torch.tensor([]).long().to(out_prob.device)))
        else:
            index_i = torch.cat(index_i).long()
            index_j = torch.cat(index_j).long()
            indices.append((index_i,index_j))
        #print(indices)
        #exit()
        return indices

def build_matcher(args):
    return HungarianMatcher(num_frames = args.num_frames, cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)
