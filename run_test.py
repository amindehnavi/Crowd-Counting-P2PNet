import argparse
import datetime
import random
from timeit import default_timer
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as standard_transforms
import numpy as np
from scipy.ndimage import filters
from scipy.stats import gaussian_kde
import scipy
from PIL import Image
import cv2
from crowd_datasets import build_dataset
from engine import *
from models import build_model
import os
import warnings
from timeit import default_timer
warnings.filterwarnings('ignore')


# function to make video from images in a directory
def make_video(args):
    images_list = os.listdir(args.output_dir)
    images_list.sort()
    images = [cv2.imread(os.path.join(args.output_dir, image)) for image in images_list]
    size = (tuple(args.shape))
    video = cv2.VideoWriter(os.path.join(args.output_dir,'output.avi'), 0, 1, size)
    for image in images:
        video.write(image)
    cv2.destroyAllWindows()
    video.release()
    print('Video saved!')

# function to support both video and images mode and return frame/image for each iteration
def get_image(args):
    if args.video:
        # read video file
        print('Reading video file...')
        cap = cv2.VideoCapture(args.video_path)
        ret, frame = cap.read()
        if not ret:
            print('Cannot open video file')
            exit(1)
        while ret:
            start_time = default_timer()
            frame = cv2.resize(frame, tuple(args.shape))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield frame, start_time
            ret, frame = cap.read()
    if args.images:
        # read images from directory
        print('Reading images from directory...')
        images_list = os.listdir(args.images_dir)
        images_list.sort()
        for img_p in images_list:
            start_time = default_timer()
            if img_p.split('.')[1] != 'jpg':
              continue
            print(f'Image {img_p} is processing...')
            img_path = os.path.join(args.images_dir, img_p)
            # img_num = img_p.split('.')[0]
            # read the given image
            img_raw = Image.open(img_path).convert('RGB')
            # get size of the image
            orig_shape = img_raw.size
            print(f'Image width:{orig_shape[0]}')
            print(f'Image Height:{orig_shape[1]}')

            # resize the image to the same size of the model input
            img_raw = img_raw.resize(
                (args.shape[0], args.shape[1]), Image.ANTIALIAS)
            yield img_raw, start_time


# function to create density maps for images
def gaussian_filter_density(gt):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 100
    # build kdtree to speed up nearest neighbor search
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)

    print('Generate density map...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        if gt_count > 3:
            # query kdtree to find 4 nearest neighbors
            distances, locations = tree.query(pts, k=4)
            # calculate the sigma of the gaussian kernel
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        elif gt_count == 3:
            # query kdtree to find 3 nearest neighbors
            distances, locations = tree.query(pts, k=3)
            # calculate the sigma of the gaussian kernel
            sigma = (distances[i][1]+distances[i][2])*0.1
        elif gt_count == 2:
            # query kdtree to find 2 nearest neighbors
            distances, locations = tree.query(pts, k=2)
            # calculate the sigma of the gaussian kernel
            sigma = (distances[i][1])*0.1
        else:
            # in case of only one point, set the sigma to be the average of the image size
            sigma = np.average(np.array(gt.shape))/2./2.
        # apply gaussian filter
        density += filters.gaussian_filter(pt2d, sigma, mode='constant')
    return density


def get_args_parser():
    parser = argparse.ArgumentParser(
        'Set parameters for P2PNet evaluation', add_help=False)

    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="name of the convolutional backbone to use. vgg16_bn, vgg16")
    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")
    parser.add_argument('--output_dir', default='',
                        help='path where to save')
    parser.add_argument('--weight_path', default='',
                        help='path where the trained weights saved')
    parser.add_argument('--gpu_id', default=0, type=int,
                        help='the gpu used for evaluation')
    parser.add_argument('--density_path', default='./density',
                        help='the density map directory')
    parser.add_argument('--density_map', action='store_true',
                        help='whether to generate density map')
    parser.add_argument('--device', default='cuda', type=str,
                        help='the device used for evaluation. cuda or cpu')
    parser.add_argument('--threshold', default=0.5, type=float,
                        help='the classification threshold')
    parser.add_argument('--shape', default=(640, 480), nargs='+',
                        type=int, help='the shape of the input image')
    parser.add_argument('--add_density_to_img', action='store_true',
                        help='whether to add density map to image')
    parser.add_argument('--video', action='store_true',
                        help='whether to evaluate on video')
    parser.add_argument('--images', action='store_true',
                        help='whether to evaluate on images')
    parser.add_argument('--images_dir', default='./Dataset',
                        type=str, help='path to images directory')
    parser.add_argument('--video_path', default='', help='path to video file')

    return parser


def main(args):
    print(args)
    # make the directories if not exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.density_path):
        os.makedirs(args.density_path)

    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)
    # set the device
    device = torch.device(args.device)
    # build the P2PNet model
    model = build_model(args)
    # move the model to GPU if available otherwise CPU
    model.to(device)
    # load pretrained weights
    if args.weight_path is not None:
        checkpoint = torch.load(args.weight_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        print('Model loaded successfully!')

    # convert model to eval mode
    model.eval()
    # create the pre-processing transform
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # # read the images from given directory
    # print('Reading images from {}'.format(args.images_dir))
    # images_list = os.listdir(args.images_dir)
    # # sort the images based on their names
    # images_list.sort()

    # counter for the number of images
    count = 0
    # Error list to store the error of each image
    ERROR = []
    # Ground truth list to store the number of ground truths of each image
    GTs = []
    # Predicted list to store the number of predicted objects of each image
    Preds = []
    # Frame per second list to store the time taken to process each image
    FPSs = []

    # read the label file
    with open('/content/CrowdCounting-P2PNet/Dataset/Labels.txt', 'r') as f:
        labels = f.readlines()

    print('Start Inference...')
    print('Using {} as input image shape'.format(tuple(args.shape)))
    print('---------------------------------------------------------------------------------------------------------------------')

    for img_raw,start_time in get_image(args):
        # pre-proccessing
        img = transform(img_raw)

        # create a tensor to store the image
        samples = torch.Tensor(img).unsqueeze(0)
        # move the image to GPU if available otherwise CPU
        samples = samples.to(device)
        # run inference
        outputs = model(samples)
        # convert the output logits to probability
        outputs_scores = torch.nn.functional.softmax(
            outputs['pred_logits'], -1)[:, :, 1][0]
        # get the predicted points
        outputs_points = outputs['pred_points'][0]

        threshold = args.threshold
        # filter the predictions which their scores are higher than threshold
        points = outputs_points[outputs_scores >
                                threshold].detach().cpu().numpy().tolist()
        # get the number of final predicted points
        predict_cnt = int((outputs_scores > threshold).sum())
        # get the corresponding scores
        outputs_scores = torch.nn.functional.softmax(
            outputs['pred_logits'], -1)[:, :, 1][0]

        if args.images:
          # append the number of ground truths to the GTs list
          GTs.append(int(labels[count][:-1]))
          # append the number of predicted objects to the Preds list
          Preds.append(predict_cnt)

          # Compute the ERROR for the given image
          ERROR.append(abs(int(labels[count][:-1]) - predict_cnt))
          print(f'GTs: {int(labels[count][:-1])}')
          print(f'Preds: {predict_cnt}')
          print(f'ERROR of this Image: {ERROR[-1]}')

        # draw the predictions
        # convert RGB to BGR for drawing using cv2
        img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
        for p in points:
            # put a filled circle at the predicted points
            img_to_draw = cv2.circle(
                img_to_draw, (int(p[0]), int(p[1])), 2, (0, 0, 255), -1)

        if args.images:
            # put the number of predicted objects and GTs on the image for images mode
            img_to_draw = cv2.putText(
                img_to_draw, f'Pred:{predict_cnt}, GT:{labels[count][:-1]}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            # put the number of predicted objects for video mode
            img_to_draw = cv2.putText(
                img_to_draw, f'Pred:{predict_cnt}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if args.density_map:
            # create the density map for images
            density_map = np.zeros((args.shape[1], args.shape[0]))
            for p in points:
                x = int(p[1])
                y = int(p[0])
                if x < density_map.shape[0] and y < density_map.shape[1]:
                    density_map[x, y] = 1
            # get the density map
            density_map = gaussian_filter_density(density_map)
            density_map = density_map / density_map.max()
            density_map = plt.cm.jet(density_map)
            density_map *= 255
            density_map_img = Image.fromarray(np.uint8(density_map))

            density_map_img.save(os.path.join(args.density_path,
                                              f'{count+1}.png'))
            print(f'Density map of image {count+1} is saved!')

        if args.add_density_to_img:
            dens_img = np.zeros(
                (density_map.shape[0], density_map.shape[1], 3))
            dens_img[..., 0] = density_map[..., 2]
            dens_img[..., 1] = density_map[..., 1]
            dens_img[..., 2] = density_map[..., 0]
            img_to_draw = 0.5*img_to_draw + 0.5*dens_img

        taken_time = default_timer() - start_time
        FPSs.append(1/taken_time)

        # save the visualized image
        cv2.imwrite(os.path.join(args.output_dir,
                    f'pred_{count+1}.jpg'), img_to_draw)
        print(f'Prediction of image {count+1} is saved!')
        print(
            f'Processing of Image {count+1} is finished after {taken_time}s!')
        count += 1
        print('--------------------------------------------\n')

    if args.video:
      make_video(args)
      print('Mean FPS:', np.mean(FPSs))
      
    if args.images:
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print(f'Inference Finished!')

        print('Metrics:')

        print('Mean FPS:', np.mean(FPSs))

        print(f'MAE: {np.mean(ERROR)}')
        print(f'SAE: {np.sum(ERROR)}')
        print(f'GTs: {np.sum(GTs)}')
        print(f'Preds: {np.sum(Preds)} \n')

        print(
            f'Max difference is {np.max(ERROR)} for Image[{np.argmax(ERROR)+1}]')
        print(
            f'Max Pred is {np.max(Preds)} and GT is {GTs[np.argmax(Preds)]} for Image[{np.argmax(Preds)+1}]')
        print(
            f'Min Pred is {np.min(Preds)} and GT is {GTs[np.argmin(Preds)]} for Image[{np.argmin(Preds)+1}]')
        print(
            f'Max GT is {np.max(GTs)} and Pred is {Preds[np.argmax(GTs)]} for Image[{np.argmax(GTs)}]')
        print(
            f'Min GT is {np.min(GTs)} and Pred is {Preds[np.argmin(GTs)]} for Image[{np.argmin(GTs)}]')

        # save the report to a text file
        with open(f'./output/{args.shape[0]}_{args.shape[1]}.txt', 'w') as f:
            f.write(f'Mean FPS: {np.mean(FPSs)}\n')
            f.write(f'MAE: {np.mean(ERROR)}\n')
            f.write(f'SAE: {np.sum(ERROR)}\n')
            f.write(f'GTs: {np.sum(GTs)}\n')
            f.write(f'Preds: {np.sum(Preds)}\n')
            f.write(
                f'Max difference is {np.max(ERROR)} for Image[{np.argmax(ERROR)+1}]\n')
            f.write(
                f'Max Pred is {np.max(Preds)} and GT is {GTs[np.argmax(Preds)]} for Image[{np.argmax(Preds)+1}]\n')
            f.write(
                f'Min Pred is {np.min(Preds)} and GT is {GTs[np.argmin(Preds)]} for Image[{np.argmin(Preds)+1}]\n')
            f.write(
                f'Max GT is {np.max(GTs)} and Pred is {Preds[np.argmax(GTs)]} for Image[{np.argmax(GTs)+1}]\n')
            f.write(
                f'Min GT is {np.min(GTs)} and Pred is {Preds[np.argmin(GTs)]} for Image[{np.argmin(GTs)+1}]\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'P2PNet evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
