import json
from pycocotools import mask as mask_utils
from collections import defaultdict
import cv2
import os
import numpy as np

res_path = '/mnt/data/exps/r50_def_enc_Ref_multiquery/results_54.json'
annotation_path = '/mnt/data/meta_expressions_rvos/valid/meta_expressions.json' #'/mnt/data/ytvis/annotations/instances_train_sub.json'#
image_path = '/mnt/data/valid_VOS_pukka/JPEGImages/'
out_path = '/mnt/data/Inference/Annotation'
vis_thresh = 0.001

annotation = json.load(open(annotation_path))
res = json.load(open(res_path))
results = defaultdict(list)

CLASSES=['person','giant_panda','lizard','parrot','skateboard','sedan','ape',
         'dog','snake','monkey','hand','rabbit','duck','cat','cow','fish',
         'train','horse','turtle','bear','motorbike','giraffe','leopard',
         'fox','deer','owl','surfboard','airplane','truck','zebra','tiger',
         'elephant','snowboard','boat','shark','mouse','frog','eagle','earless_seal',
         'tennis_racket']

def get_rand_color():
    c = ((np.random.random((3)) * 0.6 + 0.2) * 255).astype(np.int32).tolist()
    return c

print('{} tracklets'.format(len(res)))
for r in res:
    results[str(r['video_id'])].append(r)
videos = annotation['videos']
ct = 0
for vid in videos.keys():
    #vid = video['id']
    ct = ct + 1
    print('processing video {} ...'.format(ct))
    tracks = results[str(vid)]
    colors = []
    for i in range(len(tracks)):
        colors.append(get_rand_color())
    expressions_ = videos[vid]['expressions']
    for im_id, image in enumerate(videos[vid]['frames']):
        im = cv2.imread(os.path.join(image_path, vid, f'{image}.jpg'))
        h, w, _ = im.shape
        #print(f'im_id {im_id}')
        for track_id, track in enumerate(tracks):
            exp = expressions_[str(track_id)]["exp"]
            if track['score'] < vis_thresh:
                continue
            cls = CLASSES[track['category_id'] - 1]
            score = track['score']
            #try:
            mask_enc = track['segmentations'][im_id]
            #bbox_enc = track['boxes'][im_id]
            #except:
            #    break
            if mask_enc is not None:
                #rint(f'mask not nnone')
                mask_dec = mask_utils.decode(mask_enc)
                mask_dec = mask_dec > 0.5
                '''im[mask_dec] = im[mask_dec] * 0.4 + np.array(colors[track_id]) * 0.6
                valid = np.where(mask_dec == 1)
                mid = len(valid[0]) // 2
                if len(valid[0]) > 0:
                    y, x = valid[0][mid] + track_id*30, valid[1][mid]
                #print(f'bbox_enc {bbox_enc}')
                #left, top = int(bbox_enc[0]*w - bbox_enc[2]*0.5*w), int(bbox_enc[1]*h - bbox_enc[3]*0.5*h)
                #right, bottom = int(bbox_enc[0]*w + bbox_enc[2]*0.5*w), int(bbox_enc[1]*h + bbox_enc[3]*0.5*h)
                #cv2.rectangle(im, (left, top), (right, bottom), (colors[track_id][0],colors[track_id][1],colors[track_id][2]), 4)
                #x, y =  left, top
                cv2.rectangle(im, (x-10, y-20), (x+150, y+6), (255, 255, 255), -1)
                cv2.putText(im, str(cls)+'_'+str(round(score, 2))+'_'+str(exp), (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[track_id], thickness=2, lineType=cv2.LINE_AA)'''
            os.makedirs(f"{out_path}/{vid}/{track_id}", exist_ok=True)      
            #im_name = vid + '_' + image + '.jpg'
            cv2.imwrite(os.path.join(out_path, vid, str(track_id), f'{image}.png'), mask_dec*255)