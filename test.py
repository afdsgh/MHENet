import torch
import torch.nn.functional as F
import sys

sys.path.append('./models')
import numpy as np
import os
import argparse
import cv2
model_name = 'model'
from exp.model import MHENet
from data import test_dataset
import imageio
print(model_name)
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=416, help='testing size')
parser.add_argument('--gpu_id', type=str, default='2', help='select gpu id')
parser.add_argument('--test_path', type=str, default='/home/user1/wyq/Datasets/test', help='test dataset path')
opt = parser.parse_args()

dataset_path = opt.test_path

# Set device for test
if opt.gpu_id == '0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id == '1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')

# Load model architecture
model = MHENet()
model.cuda()
model.eval()

# Define the epochs you want to test
# epochs_to_test = ['best', 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
epochs_to_test = ['best']
fp = 'fuse'
rp = 'rgb'
dp = 'depth'

for epoch in epochs_to_test:
    # Load checkpoint
    ckpt_path = f'./checkpoints/{model_name}/{model_name}_epoch_{epoch}.pth'
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint {ckpt_path} not found. Skipping.")
        continue

    checkpoint = torch.load(ckpt_path)
    checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint, strict=True)

    # Save path for this epoch
    save_path_base = f'./test_maps/{model_name}_epoch_{epoch}'

    # Test datasets
    test_datasets = ['chameleon', 'camo_testing', 'cod10k_testing', 'nc4k_testing']
    # test_datasets = ['data', 'chameleon', 'camo_testing', 'cod10k_testing', 'nc4k_testing']

    for dataset in test_datasets:
        save_path_f = os.path.join(save_path_base, fp, dataset)
        save_path_r = os.path.join(save_path_base, rp, dataset)
        save_path_d = os.path.join(save_path_base, dp, dataset)
        if not os.path.exists(save_path_f):
            os.makedirs(save_path_f)
        if not os.path.exists(save_path_r):
            os.makedirs(save_path_r)
        if not os.path.exists(save_path_d):
            os.makedirs(save_path_d)

        image_root = os.path.join(dataset_path, dataset, 'rgb/')
        gt_root = os.path.join(dataset_path, dataset, 'gt/')
        depth_root = os.path.join(dataset_path, dataset, 'depth/')

        test_loader = test_dataset(image_root, gt_root, depth_root, opt.testsize)

        for i in range(test_loader.size):
            image, gt, depth,name, image_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            depth = depth.cuda()

            with torch.no_grad():
                res, res2, res3 = model(image, depth)
                res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
                res = res.sigmoid().cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)

                res2 = F.interpolate(res2, size=gt.shape, mode='bilinear', align_corners=False)
                res2 = res2.sigmoid().cpu().numpy().squeeze()
                res2 = (res2 - res2.min()) / (res2.max() - res2.min() + 1e-8)

                res3 = F.interpolate(res3, size=gt.shape, mode='bilinear', align_corners=False)
                res3 = res3.sigmoid().cpu().numpy().squeeze()
                res3 = (res3 - res3.min()) / (res3.max() - res3.min() + 1e-8)


            # Save result
            # print(f'Saving: {os.path.join(save_path, name, fuse)}')
            imageio.imwrite(os.path.join(save_path_f, name), (res * 255).astype(np.uint8))
            # imageio.imwrite(os.path.join(save_path_r, name), (res2 * 255).astype(np.uint8))
            # imageio.imwrite(os.path.join(save_path_d, name), (res3 * 255).astype(np.uint8))

        print(f'Test Done for {dataset} at epoch {epoch}')