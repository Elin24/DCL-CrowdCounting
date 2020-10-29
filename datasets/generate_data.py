#-*-encoding: utf-8 -*-

import glob
import os
import os.path as path
from PIL import Image
import scipy.io as scio
import numpy as np
import scipy.ndimage
import pickle
from tqdm import tqdm

# gauss kernel
def gen_gauss_kernels(kernel_size=15, sigma=4):
    kernel_shape  = (kernel_size, kernel_size)
    kernel_center = (kernel_size // 2, kernel_size // 2)

    arr = np.zeros(kernel_shape).astype(float)
    arr[kernel_center] = 1

    arr = scipy.ndimage.filters.gaussian_filter(arr, sigma, mode='constant') 
    kernel = arr / arr.sum()
    return kernel

def gaussian_filter_density(non_zero_points, map_h, map_w):
    """
    Fast gaussian filter implementation : using precomputed distances and kernels
    """
    gt_count = non_zero_points.shape[0]
    density_map = np.zeros((map_h, map_w), dtype=np.float32)

    for i in range(gt_count):
        point_y, point_x = non_zero_points[i]
        #print(point_x, point_y)
        kernel_size = 15 // 2
        kernel = gen_gauss_kernels(kernel_size * 2 + 1, 4)
        min_img_x = int(max(0, point_x-kernel_size))
        min_img_y = int(max(0, point_y-kernel_size))
        max_img_x = int(min(point_x+kernel_size+1, map_h - 1))
        max_img_y = int(min(point_y+kernel_size+1, map_w - 1))
        #print(min_img_x, min_img_y, max_img_x, max_img_y)
        kernel_x_min = int(kernel_size - point_x if point_x <= kernel_size else 0)
        kernel_y_min = int(kernel_size - point_y if point_y <= kernel_size else 0)
        kernel_x_max = int(kernel_x_min + max_img_x - min_img_x)
        kernel_y_max = int(kernel_y_min + max_img_y - min_img_y)
        #print(kernel_x_max, kernel_x_min, kernel_y_max, kernel_y_min)

        density_map[min_img_x:max_img_x, min_img_y:max_img_y] += kernel[kernel_x_min:kernel_x_max, kernel_y_min:kernel_y_max]
    return density_map

mod = 16
subx = 'dot'
dataset = ['SHHA', 'SHHB', 'UCF-QNRF', 'UCF-CC-50', 'GCC'][0]
if dataset == 'SHHA':
    # ShanghaiTech_A
    root, nroot = path.join('ShanghaiTech_Crowd_Detecting', 'partA'), f'SHHA{mod}'
elif dataset == 'SHHB':
    # ShanghaiTech_B
    root, nroot = path.join('ShanghaiTech_Crowd_Detecting', 'partB'), f'SHHB{mod}'
elif dataset == 'UCF-QNRF':
    # UCF-QNRF
    root, nroot = 'UCF-QNRF_ECCV18', 'UCF-QNRF_{mod}'
elif dataset == 'UCF-CC-50':
    # UCF-CC-50
    root, nroot = 'UCF-CC-50', f'UCF-CC-50_{mod}'
elif dataset == 'GCC':
    root, nroot = path.join('GCC', 'GCC-scene'), path.join(f'GCC-{mod}')

if 'SHH' in dataset:
    # ShanghiTech A and B
    imgps = glob.glob(path.join(root, '*', 'img', '*.jpg'))
elif 'UCF' in dataset:
    #UCF-QNRF and UCF-CC-50
    imgps = glob.glob(path.join(root, '*', '*.jpg'))
elif 'GCC' in dataset:
    imgps = glob.glob(path.join(root, 'scene_*', 'pngs', '*.png'))

a = 3614#3614
for i, imgp in enumerate(imgps[a:]):
    print(f'[{i+a}]: {imgp}.')
    img = Image.open(imgp)
    w, h = img.size

    if 'SHH' in dataset:
        # ShanghiTech
        mat_path = imgp.replace('.jpg', '.mat').replace('img', 'ground_truth').replace('IMG_', 'GT_IMG_')
        imgNo = path.basename(imgp).replace('IMG_', '').replace('.jpg', '')
        nimgfold = path.join(nroot, 'train' if 'train' in imgp else 'test', 'img')
        gt = scio.loadmat(mat_path)["image_info"][0,0][0,0][0].astype(np.float32) - 1.
    elif 'UCF' in dataset:
        # UCF
        mat_path = imgp.replace('.jpg', '_ann.mat')
        imgNo = path.basename(imgp).replace('img_', '').replace('.jpg', '')
        if 'QNRF' in dataset:
            nimgfold = path.join(nroot, 'train' if 'Train' in imgp else 'test', 'img')
        else:
            nimgfold = path.join(nroot, 'all', 'img')
        gt = scio.loadmat(mat_path)['annPoints'].astype(np.float32) - 1.
    elif 'GCC' in dataset:
        mat_path = imgp.replace('.png', '.mat').replace('pngs', 'mats')
        imgNo = path.basename(imgp).replace('.png', '')
        gt = scio.loadmat(mat_path)["image_info"][0,0][0].astype(np.float32)
        gt = gt[:, ::-1]
        nimgfold = path.join(nroot, 'img')

    if max(w, h) > 1024:
        if w == max(w, h):
            nw, nh = 1024, round(h * 1024 / w / mod) * mod
        else:
            nh, nw = 1024, round(w * 1024 / h / mod) * mod
    else:
        nw, nh = round((w / mod)) * mod, round((h / mod)) * mod
        

    # new resized image save
    if not path.exists(nimgfold):
        os.makedirs(nimgfold)
    img.resize((nw, nh), Image.BILINEAR).save(path.join(nimgfold, imgNo + ('.jpg' if 'GCC' != dataset else '.png')))
    if len(gt) > 0:
        gt[:, 0] = gt[:, 0].clip(0, w - 1)
        gt[:, 1] = gt[:, 1].clip(0, h - 1)
        gt[:, 0] = (gt[:, 0] / w * nw).round().astype(int)
        gt[:, 1] = (gt[:, 1] / h * nh).round().astype(int)

    # new gt maps save
    # ngtfold = nimgfold.replace('img', 'pkl')
    # if not path.exists(ngtfold):
    #     os.makedirs(ngtfold)
    # with open(path.join(ngtfold, f'{imgNo}.pkl'), 'wb+') as f:
    #     pickle.dump(gt, f)
    
    # new den csv save
    csvfold = nimgfold.replace('img', 'den')
    if not path.exists(csvfold):
        os.makedirs(csvfold)
    den = gaussian_filter_density(gt, nh, nw)
    np.savetxt(path.join(csvfold, f'{imgNo}.csv'), den, delimiter=",")

    print(f'-- OK --')