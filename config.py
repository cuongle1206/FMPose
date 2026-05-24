import os.path as osp
import torch
import argparse

parser          = argparse.ArgumentParser()
parser.add_argument('--dir', default='./project')
parser.add_argument('--exp', default='h36m') # select the database (h36m, 3dhp, 3dpw)
parser.add_argument('--seed', type=int, default=42) # 40-44
parser.add_argument('--act', type=str, default='swish') # (relu, lrelu, sigmoid, tanh, swish)
parser.add_argument('--topk', type=int, default=48)
parser.add_argument('--num_gcn', type=int, default=1)
parser.add_argument('--sampling', action="store_true")
parser.add_argument('--solver', type=str, default='midpoint')
parser.add_argument('--n_steps', type=int, default=25)
parser.add_argument('--wandb', action="store_true")
args            = parser.parse_args()
device          = 'cuda' if torch.cuda.is_available() else 'cpu'
if args.exp == '3dpw':
    base_dir      = osp.join('models', 'trained_models', '3dpw')
else: # h36m + 3dhp + ablations
    base_dir      = osp.join('models', 'trained_models', 'h36m')
model_best = osp.join(base_dir, f'model_best_{args.topk}_{args.seed}.pt')
model_last = osp.join(base_dir, f'model_last_{args.topk}_{args.seed}.pt')

if args.exp == '3dpw':
    n_epochs        = 100
    lr_step         = 80
    batch_size      = 128
    lr              = 1e-4
else: #h36m and ablations
    n_epochs        = 100
    lr_step         = 90
    batch_size      = 64
    lr              = 1e-4

# width and height of heatmaps and images
hm_h = hm_w     = 64 # heatmap size
img_h = img_w   = 256 # img size
N_3DJOINTS      = 17 # Number of 3D joints - h36m
N_2DJOINTS      = 16 # Number of 2D joints - mpii

# Multi-hypothesis setting
n_hypo, std_dev = 200, 1.0

# ODE setting
n_steps         = args.n_steps
step_size       = 1/n_steps

# Hard pose detection
gt_sigma        = 2.0  # sigma in px of the ground truth heatmaps for training the 2d detector
p3d_std         = 0.010  # stddev in m corresponding to gt_sigma px stddev
# conversion factor to relate between covariance matrices from 3d pose hypotheses and from heatmaps:
hm_px_to_mm     = (p3d_std / gt_sigma)**2
