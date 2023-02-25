import cv2
import math
import numpy as np
import os
import os.path as osp
import random
import time
import torch
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from torch.utils import data as data
import yaml
from basicsr.utils.options import ordered_yaml
from basicsr.data import build_dataloader, build_dataset
from basicsr.utils.img_process_util import filter2D
import torch.nn.functional as F
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.utils import DiffJPEG
from thop import profile,clever_format
import gdssr.archs
import gdssr.data
import gdssr.models
import cv2
from basicsr.utils.img_util import tensor2img
from basicsr.archs.rrdbnet_arch import RRDBNet
from gdssr.archs.gdssr_arch import GDSSR_RRDBNet_test
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
import argparse

def tile_process(model, img, scale, tile, tile_pad=10):
    """It will first crop input images to tiles, and then process each tile.
    Finally, all the processed tiles are merged into one images.
    Modified from: https://github.com/ata4/esrgan-launcher
    """
    if tile==0:
        with torch.no_grad():
            result,deg = model(img)
        return result
    with torch.no_grad():
            degrees, feat = model.de_net(im)

    batch, channel, height, width = img.shape
    output_height = height * scale
    output_width = width * scale
    output_shape = (batch, channel, output_height, output_width)

    # start with black image
    output = img.new_zeros(output_shape)
    tiles_x = math.ceil(width / tile)
    tiles_y = math.ceil(height / tile)

    # loop over all tiles
    for y in range(tiles_y):
        for x in range(tiles_x):
            # extract tile from input image
            ofs_x = x * tile
            ofs_y = y * tile
            # input tile area on total image
            input_start_x = ofs_x
            input_end_x = min(ofs_x + tile, width)
            input_start_y = ofs_y
            input_end_y = min(ofs_y + tile, height)

            # input tile area on total image with padding
            input_start_x_pad = max(input_start_x - tile_pad, 0)
            input_end_x_pad = min(input_end_x + tile_pad, width)
            input_start_y_pad = max(input_start_y - tile_pad, 0)
            input_end_y_pad = min(input_end_y + tile_pad, height)

            # input tile dimensions
            input_tile_width = input_end_x - input_start_x
            input_tile_height = input_end_y - input_start_y
            tile_idx = y * tiles_x + x + 1
            input_tile = img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

            # upscale tile
            try:
                with torch.no_grad():
                    output_tile, d = model(input_tile, deg_feat = feat)
            except RuntimeError as error:
                print('Error', error)
            #print(f'\tTile {tile_idx}/{tiles_x * tiles_y}')

            # output tile area on total image
            output_start_x = input_start_x * scale
            output_end_x = input_end_x * scale
            output_start_y = input_start_y * scale
            output_end_y = input_end_y * scale

            # output tile area without padding
            output_start_x_tile = (input_start_x - input_start_x_pad) * scale
            output_end_x_tile = output_start_x_tile + input_tile_width * scale
            output_start_y_tile = (input_start_y - input_start_y_pad) * scale
            output_end_y_tile = output_start_y_tile + input_tile_height * scale

            # put tile into output image
            output[:, :, output_start_y:output_end_y,
                        output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                                                    output_start_x_tile:output_end_x_tile]
    return output
# opt_path = 'options/val.yml'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default='options/GDSSR_GAN.yml')
    parser.add_argument('--model_path', type=str, default='checkpoints/GDSSR_GAN.pth')
    parser.add_argument('--im_path', type=str, default='./test')
    parser.add_argument('--res_path', type=str, default='./result')
    parser.add_argument('--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    args = parser.parse_args()

    with open(args.opt, mode='r') as f:
        opt = yaml.load(f, Loader=ordered_yaml()[0])
    try:
        os.makedirs(args.res_path)
    except:
        pass
    with open(args.opt, mode='r') as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)
    model = GDSSR_RRDBNet_test(**opt['network_g'])
    loadnet = torch.load(args.model_path, map_location=torch.device('cpu'))
    model.load_state_dict(loadnet['params'], strict=True)
    model.to('cuda:0')
    model.eval()
    input = torch.randn(1, 3, 224, 224).to('cuda:0')
    flops, params = profile(model, inputs=(input, ))
    flops, params = clever_format([flops, params], "%.3f")
    print("flops:",flops,"params:",params)

    im_list = os.listdir(args.im_path)
    im_list.sort()
    im_list = [name for name in im_list if name.endswith('.png') or name.endswith('.jpg')]
    from tqdm import tqdm
    with torch.no_grad():
        for name in tqdm(im_list,total=len(im_list)):
            path = os.path.join(args.im_path, name)
            im = cv2.imread(path)
            im = img2tensor(im)
            im = im.unsqueeze(0).cuda(0)/255.
            sr = tile_process(model, im, opt['scale'], args.tile)
            im_sr = tensor2img(sr, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1))
            save_path = os.path.join(args.res_path, name+'_out.png')
            cv2.imwrite(save_path, im_sr)