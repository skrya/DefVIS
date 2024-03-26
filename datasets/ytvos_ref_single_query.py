"""
YoutubeVIS data loader
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools.ytvos import YTVOS
from pycocotools.ytvoseval import YTVOSeval
import datasets.transforms as T
from pycocotools import mask as coco_mask
import os
from PIL import Image
from random import randint
import cv2
import random
from transformers import RobertaTokenizerFast
import json

class YTVOSDataset:
    def __init__(self, img_folder, ann_file, ann_to_exp_path, transforms, return_masks, tokenizer, num_frames):
        self.img_folder = img_folder
        self.ann_file = ann_file
        self._transforms = transforms
        self.return_masks = return_masks
        self.num_frames = num_frames
        self.prepare = ConvertCocoPolysToMask(return_masks, tokenizer=tokenizer)
        self.ytvos = YTVOS(ann_file)
        self.cat_ids = self.ytvos.getCatIds()
        self.vid_ids = self.ytvos.getVidIds()
        self.vid_infos = []
        for i in self.vid_ids:
            info = self.ytvos.loadVids([i])[0]
            info['filenames'] = info['file_names']
            self.vid_infos.append(info)
        self.img_ids = []
        for idx, vid_info in enumerate(self.vid_infos):
            frame_id = random.randint(0,len(vid_info['filenames'])-1)
            self.img_ids.append((idx, frame_id))
        if ann_to_exp_path is not None :
            #print(f'ann_to_exp_path {ann_to_exp_path}')
            with open(ann_to_exp_path) as f:
                self.ann_to_exp = json.load(f)
    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        vid,  frame_id = self.img_ids[idx]
        vid_id = self.vid_infos[vid]['id']
        img = []
        vid_len = len(self.vid_infos[vid]['file_names'])
        inds = list(range(self.num_frames))
        inds = [i%vid_len for i in inds][::-1]
        # if random 
        # random.shuffle(inds)
        for j in range(self.num_frames):
            img_path = os.path.join(str(self.img_folder), self.vid_infos[vid]['file_names'][frame_id-inds[j]])
            img.append(Image.open(img_path).convert('RGB'))
        ann_ids = self.ytvos.getAnnIds(vidIds=[vid_id])
        target = self.ytvos.loadAnns(ann_ids)
        target = {'image_id': idx, 'video_id': vid, 'frame_id': frame_id, 'annotations': target}
        target = self.prepare(img[0], target, inds, self.num_frames, self.ann_to_exp)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        
        return torch.cat(img,dim=0), target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        if not polygons:
            mask = torch.zeros((height,width), dtype=torch.uint8)
        else:
            rles = coco_mask.frPyObjects(polygons, height, width)
            mask = coco_mask.decode(rles)
            if len(mask.shape) < 3:
                mask = mask[..., None]
            mask = torch.as_tensor(mask, dtype=torch.uint8)
            mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False, tokenizer=None):
        self.return_masks = return_masks
        self.tokenizer = tokenizer
        self.max_tokens = 20

    def __call__(self, image, target, inds, num_frames, ann_to_exp):
        w, h = image.size
        image_id = target["image_id"]
        frame_id = target['frame_id']
        image_id = torch.tensor([image_id])

        anno = target["annotations"]
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]
        boxes = []
        classes = []
        segmentations = []
        area = []
        iscrowd = []
        valid = []
        expressions = []
        exps_input_ids = []
        exps_attn_masks = []
        # add valid flag for bboxes
        #for i, ann in enumerate(anno):
        ann = anno[random.randint(0,len(anno)-1)]
        exp = ann_to_exp[str(ann['id'])][0]
        expressions.append(exp)
        attention_mask = [0] * self.max_tokens
        padded_input_ids = [0] * self.max_tokens
        tokenized = self.tokenizer(exp)
        token_len = min(len(tokenized["input_ids"]), self.max_tokens-1)
        padded_input_ids[:token_len] = tokenized["input_ids"][:token_len]
        attention_mask[:token_len] = tokenized["attention_mask"][:token_len]
        exps_input_ids.append(torch.tensor(padded_input_ids).unsqueeze(0))
        exps_attn_masks.append(torch.tensor(attention_mask).unsqueeze(0))
        for j in range(num_frames):
            bbox = ann['bboxes'][frame_id-inds[j]]
            areas = ann['areas'][frame_id-inds[j]]
            segm = ann['segmentations'][frame_id-inds[j]]
            clas = ann["category_id"]
            # for empty boxes
            if bbox is None:
                bbox = [0,0,0,0]
                areas = 0
                valid.append(0)
                clas = 0
            else:
                valid.append(1)
            crowd = ann["iscrowd"] if "iscrowd" in ann else 0
            boxes.append(bbox)
            area.append(areas)
            segmentations.append(segm)
            classes.append(clas)
            iscrowd.append(crowd)
            

        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        classes = torch.tensor(classes, dtype=torch.int64)
        if self.return_masks:
            masks = convert_coco_poly_to_mask(segmentations, h, w)
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id

        # for conversion to coco api
        area = torch.tensor(area) 
        iscrowd = torch.tensor(iscrowd)
        target["valid"] = torch.tensor(valid)
        
        target["exps_input_ids"] = torch.cat(exps_input_ids, dim=0)
        target["exps_attn_masks"] = torch.cat(exps_attn_masks, dim=0)
        #print(f'exps_input_ids {target["exps_input_ids"].shape} exps_attn_masks {target["exps_attn_masks"].shape}')
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        return  target


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResize(scales, max_size=800),
            T.PhotometricDistort(),
            T.Compose([
                     T.RandomResize([400, 500, 600]),
                     T.RandomSizeCrop(384, 600),
                     # To suit the GPU memory the scale might be different
                     T.RandomResize([300], max_size=540),#for r50
                     #T.RandomResize([280], max_size=504),#for r101
            ]),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([360], max_size=640),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.ytvos_path)
    assert root.exists(), f'provided YTVOS path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train/JPEGImages", root / "annotations" / f'{mode}_train_sub.json', root / "ann_id_to_exp.json"),
        "val": (root / "valid/JPEGImages", root / "annotations" / f'{mode}_val_sub.json', None),
    }
    img_folder, ann_file, actual_ann_file = PATHS[image_set]
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    dataset = YTVOSDataset(img_folder, ann_file, actual_ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks, tokenizer=tokenizer, num_frames = args.num_frames)
    return dataset
