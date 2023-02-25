import cv2
import glob
import numpy as np
import os.path as osp
from torchvision.transforms.functional import normalize
from basicsr.utils import img2tensor
import lpips
import argparse
from tqdm import tqdm
import torch

result = 0
count = 0

def split_test(Model,res,gt):
    global result,count
    b,c,h,w = res.shape
    if(h*w<2000*2000):
        with torch.no_grad():
            result += Model(res,gt)
            count +=1 
    else:
        split_test(Model,res[:,:,:h//2,:w//2],gt[:,:,:h//2,:w//2])
        split_test(Model,res[:,:,:h//2,w//2:],gt[:,:,:h//2,:w//2])
        split_test(Model,res[:,:,h//2:,:w//2],gt[:,:,h//2:,:w//2])
        split_test(Model,res[:,:,h//2:,w//2:],gt[:,:,h//2:,w//2:])

def main():
    global result,count
    # Configurations
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_gt', type=str, default='/group/30042/chongmou/ft_local/Real-ESRGAN-master/testdata_real/AIM19/valid-gt-clean')
    parser.add_argument('--folder_restored', type=str, default='results/aim19')
    parser.add_argument("--split", action="store_true")
    args = parser.parse_args()
    loss_fn_vgg = lpips.LPIPS(net='vgg').cuda(0)
    lpips_all = []
    img_list = sorted(glob.glob(osp.join(args.folder_gt, '*.png')))
    lr_list = sorted(glob.glob(osp.join(args.folder_restored, '*.png')))
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    for i, (img_path, lr_path) in enumerate(zip(img_list,lr_list)):
        basename, ext = osp.splitext(osp.basename(img_path))
        img_gt = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        img_restored = cv2.imread(osp.join(lr_path), cv2.IMREAD_UNCHANGED).astype(
            np.float32) / 255.

        img_gt, img_restored = img2tensor([img_gt, img_restored], bgr2rgb=True, float32=True)
        # norm to [-1, 1]
        normalize(img_gt, mean, std, inplace=True)
        normalize(img_restored, mean, std, inplace=True)

        # calculate lpips
        result = 0
        count = 0
        if args.split:
            split_test(loss_fn_vgg,img_restored.unsqueeze(0).cuda(0), img_gt.unsqueeze(0).cuda(0))
            lpips_val = (result/count).cpu().data.numpy()[0,0,0,0]
        else:
            lpips_val = loss_fn_vgg(img_restored.unsqueeze(0).cuda(0), img_gt.unsqueeze(0).cuda(0)).cpu().data.numpy()[0,0,0,0]
        #print(result,count,lpips_val)
        lpips_all.append(lpips_val)

    print(f'Average: LPIPS: {sum(lpips_all) / len(lpips_all):.6f}')


if __name__ == '__main__':
    main()
