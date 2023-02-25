import cv2
import glob
import numpy as np
import os.path as osp

import argparse
import torch
from torch.autograd import Variable
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr





def main():
    # Configurations
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_gt', type=str, default='/group/30042/chongmou/ft_local/Real-ESRGAN-master/testdata_real/AIM19/valid-gt-clean')
    parser.add_argument('--folder_restored', type=str, default='results/aim19')
    args = parser.parse_args()

    img_list = sorted(glob.glob(osp.join(args.folder_gt, '*.png')))
    lr_list = sorted(glob.glob(osp.join(args.folder_restored, '*.png')))

    psnr_all = []
    ssim_all = []
    for i, (img_path, lr_path) in enumerate(zip(img_list,lr_list)):
        basename, ext = osp.splitext(osp.basename(img_path))
        img_gt = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img_restored = cv2.imread(osp.join(lr_path), cv2.IMREAD_UNCHANGED)
        img_gty = cv2.cvtColor(img_gt, cv2.COLOR_BGR2YCR_CB)[:,:,0]
        img_resy = cv2.cvtColor(img_restored, cv2.COLOR_BGR2YCR_CB)[:,:,0]
        # calculate lpips
        psnr_val = compare_psnr(img_gty, img_resy)
        ssim_val = compare_ssim(img_gt, img_restored,multichannel=True)
        #print(lpips_val)
        psnr_all.append(psnr_val)
        ssim_all.append(ssim_val)

    print(f'Average: PSNR: {sum(psnr_all) / len(psnr_all):.6f}')
    print(f'Average: SSIM: {sum(ssim_all) / len(ssim_all):.6f}')


if __name__ == '__main__':
    main()
