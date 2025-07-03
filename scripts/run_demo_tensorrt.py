import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore')  # TODO

import argparse
import cv2
import glob
import imageio
import logging
import os
import numpy as np
from typing import List
import copy

import omegaconf
import onnxruntime as ort
import open3d as o3d
import torch
import yaml
import time
from onnx_tensorrt import tensorrt_engine
import tensorrt as trt

import sys
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')

import Utils as utils


def preprocess(image_path, args):
    input_image = imageio.imread(image_path)
    if args.height and args.width:
        input_image = cv2.resize(input_image, (args.width, args.height))
    else:
        input_image = cv2.resize(input_image, fx=1, fy=1, dsize=None)
    resized_image = torch.as_tensor(input_image.copy()).float()[None].permute(0,3,1,2).contiguous()
    return resized_image, input_image


def get_pytorch_model(args):
    with open(f'{os.path.dirname(args.pretrained)}/cfg.yaml', 'r') as file:
        cfg = yaml.safe_load(file)
    cfg = omegaconf.OmegaConf.create(cfg)
    model = StereoAnything(cfg)

    checkpoint = torch.load(args.pretrained, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    model.cuda()
    model.eval()
    return model


def get_onnx_model(args):
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    model = ort.InferenceSession(
        args.pretrained, sess_options=session_options, providers=['CUDAExecutionProvider']) # 
    return model


def get_engine_model(args):
    with open(args.pretrained, 'rb') as file:
        engine_data = file.read()
    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(engine_data)
    engine = tensorrt_engine.Engine(engine)
    return engine


def inference_pytorch(args, model, left_img, right_img):
    with torch.amp.autocast('cuda', enabled=True):
        if not args.hiera:
            padder = InputPadder(left_img.shape, divis_by=32, force_square=False)
            left_img, right_img = padder.pad(left_img.cuda(), right_img.cuda())
            left_disp = model(left_img, right_img, iters=args.valid_iters, test_mode=True)
            left_disp = padder.unpad(left_disp)
        else:
            left_disp = model.run_hierachical(
                left_img, right_img, iters=args.valid_iters, test_mode=True, small_ratio=0.5)
    return left_disp.float().cpu().numpy().squeeze()


def inference(image_list: List[str], model, args: argparse.Namespace):

    for image in image_list:
        left_img, input_left = preprocess(image, args)
        right_img, _ = preprocess(image.replace('/im0', '/im1'), args)

        start_time = time.time()
        if args.pretrained.endswith('.pth'):
            left_disp = inference_pytorch(args, model, left_img, right_img)
        elif args.pretrained.endswith('.onnx'):  
            left_disp = model.run(None, {
                'left': left_img.numpy(),
                'right': right_img.numpy()})[0]
        else:
            left_disp = model.run([left_img.numpy(), right_img.numpy()])[0]
        end_time = time.time()
        logging.info(f'Inference time: {end_time - start_time:.3f} seconds')

        left_disp = left_disp.squeeze()  # HxW

        vis = utils.vis_disparity(left_disp)
        vis = np.concatenate([input_left, vis], axis=1)
        imageio.imwrite(os.path.join(args.save_path, 'visual', image.split('/')[-1]), vis)

        if args.pc:
            save_path = image.split('/')[-1].split('.')[0] + '.ply'

            baseline = 193.001/1e3
            doffs = 65.555
            K = np.array([1998.842, 0, 588.364,
                        0, 1998.842, 505.864,
                        0,0,1]).reshape(3,3)
            depth = K[0,0]*baseline/(left_disp + doffs)
            xyz_map = utils.depth2xyzmap(depth, K)
            pcd = utils.toOpen3dCloud(xyz_map.reshape(-1,3), input_left.reshape(-1,3))
            keep_mask = (np.asarray(pcd.points)[:,2]>0) & (np.asarray(pcd.points)[:,2]<=args.z_far)
            keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
            pcd = pcd.select_by_index(keep_ids)
            o3d.io.write_point_cloud(os.path.join(args.save_path, 'cloud', save_path), pcd)

            ######## (Optional) remove cloud noise
            logging.info("denoise point cloud...")
            _, ind = pcd.remove_radius_outlier(nb_points=100, radius=0.1)
            inlier_cloud = pcd.select_by_index(ind)
            o3d.io.write_point_cloud(
                os.path.join(args.save_path, 'denoised_cloud', save_path), inlier_cloud)


def parse_args() -> omegaconf.OmegaConf:
    parser = argparse.ArgumentParser(description='Stereo 2025')

    # File options
    parser.add_argument('--left_img', '-l', required=True, help='Path to left image.')
    parser.add_argument('--save_path', '-s', default='tmp', help='Path to save results.')
    parser.add_argument('--pretrained', default='2024-12-13-23-51-11/model_best_bp2.pth',
                        help='Path to pretrained model')

    # Inference options
    parser.add_argument('--hiera', action='store_true', help='Hierarchical inference')
    parser.add_argument('--valid_iters', type=int, default=32,
                        help='number of flow-field updates during forward pass')
    parser.add_argument('--height', type=int, default=288, help='Image height')
    parser.add_argument('--width', type=int, default=480, help='Image width')
    parser.add_argument('--pc', action='store_true', help='Save point cloud')
    parser.add_argument('--z_far', default=10, type=float, help='max depth to clip in point cloud')
    parser.add_argument('--num_runs', type=int, default=5, help='Number of inference runs')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    image_list = [args.left_img]
    os.makedirs(args.save_path, exist_ok=True)
    paths = ['continuous/disparity', 'visual', 'denoised_cloud', 'cloud']
    for p in paths:
        os.makedirs(os.path.join(args.save_path, p), exist_ok=True)

    assert os.path.isfile(args.pretrained), f'Pretrained model {args.pretrained} not found'
    logging.info('Pretrained model loaded from %s', args.pretrained)
    utils.set_seed(0)
    if args.pretrained.endswith('.pth'):
        torch.autograd.set_grad_enabled(False)
        model = get_pytorch_model(args)
    elif args.pretrained.endswith('.onnx'):
        model = get_onnx_model(args)
    elif args.pretrained.endswith('.engine') or args.pretrained.endswith('.plan'):
        model = get_engine_model(args)
    else:
        assert False, f'Unknown model format {args.pretrained}.'

    num_runs = 5
    for i in range(num_runs):
        inference(image_list, model, args)

if __name__ == '__main__':
    main()