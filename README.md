# Crowd Counting

Computer Vision Final Project, Crowd counting + density map

# Introduction

in this project we used P2PNet model [[Paper](https://arxiv.org/abs/2107.12746)][[Github repo](https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet)] to estimate number of people in crowd scences.

# Setup

1. clone this repository: `git clone https://github.com/amindehnavi/Crowd-Counting-P2PNet`  
2. change the current directory into `CrowdCounting-P2PNet` folder: `cd CrowdCounting-P2PNet`  
3. install requirements libraries using `pip install -r requirements.txt` command.  
4. download the [vgg16-bn](https://download.pytorch.org/models/vgg16_bn-6c64b313.pth)/[vgg16](https://download.pytorch.org/models/vgg16-397923af.pth) pretrained weight on ImageNet and put it in `checkpoints` folder

# Demo

to run the model on a some images set video, run the following command  
`CUDA_VISIBLE_DEVICES=0 python run_test.py --weight_path ./weights/SHTechA.pth --output_dir ./output/ --device cuda --shape 640 480 --threshold 0.75 --images --images_dir ./Dataset --density_map`

to test on a video  
`CUDA_VISIBLE_DEVICES=0 python run_test.py --weight_path ./weights/SHTechA.pth --output_dir ./output/ --device cuda --shape 640 480 --threshold 0.75 --video --video_path /path/to/video`

Note 1: to get the density map, set the `--density_map` flag in command line.  
Note 2: to add the density map on original image, set the `--add_density_to_image` flag in command line.
