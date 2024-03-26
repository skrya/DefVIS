## Deformable vistr: Spatio temporal deformable attention for video instance segmentation

This is the official implementation of the [DefVisTR paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9746665):

<p align="center">
<img src="https://user-images.githubusercontent.com/16319629/110786946-b99aa080-82a7-11eb-98e4-85478ca4eeac.png" width="600">
</p>


### Installation
We provide instructions how to install dependencies via conda.
First, clone the repository locally:
```
git clone https://github.com/skrya/DefVIS.git
```
Then, install PyTorch 1.6 and torchvision 0.7:
```
conda install pytorch==1.6.0 torchvision==0.7.0
```
Install pycocotools
```
conda install cython scipy
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install git+https://github.com/youtubevos/cocoapi.git#"egg=pycocotools&subdirectory=PythonAPI"
```
Compile DCN module(requires GCC>=5.3, cuda>=10.0)
```
cd models/dcn
python setup.py build_ext --inplace
```

### Compiling CUDA operators
```bash
cd ./models/ops
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```

### Preparation

Download and extract 2019 version of YoutubeVIS  train and val images with annotations from
[CodeLab](https://competitions.codalab.org/competitions/20128#participate-get_data) or [YoutubeVIS](https://youtube-vos.org/dataset/vis/).
We expect the directory structure to be the following:
```
VisTR
├── data
│   ├── train
│   ├── val
│   ├── annotations
│   │   ├── instances_train_sub.json
│   │   ├── instances_val_sub.json
├── models
...
```

Download the pretrained deformable DETR models [Defomable DeTR Repository (44.5 AP)](https://github.com/fundamentalvision/Deformable-DETR/tree/main) on COCO and save it to the pretrained path.

### Compile CUDA operators
```
cd ./models/ops
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```
### Training

Training of the model requires 4 GPU cards with each 15GB.

To train baseline VisTR on a single node with 4 gpus for 60 epochs (Bsz 1), trains in 1 day 10:28 hrs:

```
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --backbone resnet50 --ytvos_path /mnt/data/ytvis/ --masks --pretrained_weights ../VisTR/<deformable_detr_coco_r50>.pth
```
### Inference

```
python inference.py --masks --model_path /mnt/data/exps/r50_def_enc_VisTR/checkpoint0059.pth --save_path /mnt/data/exps/results.json --img_path /mnt/data/ytvis/valid/JPEGImages/ --ann_path /mnt/data/ytvis/valid_vis_codelab.json  --backbone resnet50
```


### License

DefVIS is released under the Apache 2.0 license. Please see the [LICENSE](LICENSE) file for more information.

### Acknowledgement
We would like to thank the [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR.git) open-source project for its awesome work, part of the code are modified from its project.

### Citation

Please consider citing our paper in your publications if the project helps your research. BibTeX reference is as follow.

```
@inproceedings{yarram2022deformable,
  title={Deformable vistr: Spatio temporal deformable attention for video instance segmentation},
  author={Yarram, Sudhir and Wu, Jialian and Ji, Pan and Xu, Yi and Yuan, Junsong},
  booktitle={ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={3303--3307},
  year={2022},
  organization={IEEE}
}
```

