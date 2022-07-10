# Crowd-Counting
Computer Vision Final Project, Crowd counting + density map

# Introduction

in this project we used P2PNet model [[Paper](https://arxiv.org/abs/2107.12746)][Github repo](https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet)] to estimate number of people in crowd scences.  
A brief introduction of P2PNet can be found at [机器之心 (almosthuman)](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650827826&idx=3&sn=edd3d66444130fb34a59d08fab618a9e&chksm=84e5a84cb392215a005a3b3424f20a9d24dc525dcd933960035bf4b6aa740191b5ecb2b7b161&mpshare=1&scene=1&srcid=1004YEOC7HC9daYRYeUio7Xn&sharer_sharetime=1633675738338&sharer_shareid=7d375dccd3b2f9eec5f8b27ee7c04883&version=3.1.16.5505&platform=win#rd).

# Setup
1. clone this repository: `git clone https://github.com/amindehnavi/Crowd-Counting-P2PNet`
2. change the current directory into `CrowdCounting-P2PNet` folder: `cd CrowdCounting-P2PNet`
3. install requirements libraries using `pip install -r requirements.txt` command.

# Demo

to run the model on a some images set video, run the following command  
`CUDA_VISIBLE_DEVICES=0 python run_test.py --weight_path ./weights/SHTechA.pth --output_dir ./output/ --device cuda --shape 640 480 --threshold 0.75 --images --images_dir ./Dataset --density_map`

to test on a video  
`CUDA_VISIBLE_DEVICES=0 python run_test.py --weight_path ./weights/SHTechA.pth --output_dir ./output/ --device cuda --shape 640 480 --threshold 0.75 --video --video_path /path/to/video`

Note 1: to get the density map, set the `--density_map` flag in command line.  
Note 2: to add the density map on original image, set the `--add_density_to_image` flag in command line.
