# Cuong Le - CVL, Linköping University
import sys, os, shutil
parent_dir = os.path.dirname("./")
sys.path.append(parent_dir)
import pickle, math
import numpy as np
import torch
from torchvision import transforms
from time import time

# database
import data.data_h36m as dataset
import config as c
from utils.data import *
from utils.metrics import *
from models.model import FlowNet, WrappedModel

# flow_matching
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import ODESolver
from functools import partial

# visualization
from PIL import Image
from yacs.config import CfgNode as CN
import matplotlib.pyplot as plt
import data.hrnet.model as hrnet

def get_hrnet_config_file(config_file):
    _C = CN()

    _C.OUTPUT_DIR = ''
    _C.LOG_DIR = ''
    _C.DATA_DIR = ''
    _C.GPUS = (0,)
    _C.WORKERS = 4
    _C.PRINT_FREQ = 20
    _C.AUTO_RESUME = False
    _C.PIN_MEMORY = True
    _C.RANK = 0

    # Cudnn related params
    _C.CUDNN = CN()
    _C.CUDNN.BENCHMARK = True
    _C.CUDNN.DETERMINISTIC = False
    _C.CUDNN.ENABLED = True

    # common params for NETWORK
    _C.MODEL = CN()
    _C.MODEL.NAME = 'pose_hrnet'
    _C.MODEL.INIT_WEIGHTS = True
    _C.MODEL.PRETRAINED = ''
    _C.MODEL.NUM_JOINTS = 17
    _C.MODEL.TAG_PER_JOINT = True
    _C.MODEL.TARGET_TYPE = 'gaussian'
    _C.MODEL.IMAGE_SIZE = [256, 256]  # width * height, ex: 192 * 256
    _C.MODEL.HEATMAP_SIZE = [64, 64]  # width * height, ex: 24 * 32
    _C.MODEL.SIGMA = 2
    _C.MODEL.EXTRA = CN(new_allowed=True)

    _C.LOSS = CN()
    _C.LOSS.USE_OHKM = False
    _C.LOSS.TOPK = 8
    _C.LOSS.USE_TARGET_WEIGHT = True
    _C.LOSS.USE_DIFFERENT_JOINTS_WEIGHT = False

    # DATASET related params
    _C.DATASET = CN()
    _C.DATASET.ROOT = ''
    _C.DATASET.DATASET = 'mpii'
    _C.DATASET.TRAIN_SET = 'train'
    _C.DATASET.TEST_SET = 'valid'
    _C.DATASET.DATA_FORMAT = 'jpg'
    _C.DATASET.HYBRID_JOINTS_TYPE = ''
    _C.DATASET.SELECT_DATA = False

    # training data augmentation
    _C.DATASET.FLIP = True
    _C.DATASET.SCALE_FACTOR = 0.25
    _C.DATASET.ROT_FACTOR = 30
    _C.DATASET.PROB_HALF_BODY = 0.0
    _C.DATASET.NUM_JOINTS_HALF_BODY = 8
    _C.DATASET.COLOR_RGB = False

    # train
    _C.TRAIN = CN()

    _C.TRAIN.LR_FACTOR = 0.1
    _C.TRAIN.LR_STEP = [90, 110]
    _C.TRAIN.LR = 0.001

    _C.TRAIN.OPTIMIZER = 'adam'
    _C.TRAIN.MOMENTUM = 0.9
    _C.TRAIN.WD = 0.0001
    _C.TRAIN.NESTEROV = False
    _C.TRAIN.GAMMA1 = 0.99
    _C.TRAIN.GAMMA2 = 0.0

    _C.TRAIN.BEGIN_EPOCH = 0
    _C.TRAIN.END_EPOCH = 140

    _C.TRAIN.RESUME = False
    _C.TRAIN.CHECKPOINT = ''

    _C.TRAIN.BATCH_SIZE_PER_GPU = 32
    _C.TRAIN.SHUFFLE = True

    # testing
    _C.TEST = CN()

    # size of images for each device
    _C.TEST.BATCH_SIZE_PER_GPU = 32
    # Test Model Epoch
    _C.TEST.FLIP_TEST = False
    _C.TEST.POST_PROCESS = False
    _C.TEST.SHIFT_HEATMAP = False

    _C.TEST.USE_GT_BBOX = False

    # nms
    _C.TEST.IMAGE_THRE = 0.1
    _C.TEST.NMS_THRE = 0.6
    _C.TEST.SOFT_NMS = False
    _C.TEST.OKS_THRE = 0.5
    _C.TEST.IN_VIS_THRE = 0.0
    _C.TEST.COCO_BBOX_FILE = ''
    _C.TEST.BBOX_THRE = 1.0
    _C.TEST.MODEL_FILE = ''

    # debug
    _C.DEBUG = CN()
    _C.DEBUG.DEBUG = False
    _C.DEBUG.SAVE_BATCH_IMAGES_GT = False
    _C.DEBUG.SAVE_BATCH_IMAGES_PRED = False
    _C.DEBUG.SAVE_HEATMAPS_GT = False
    _C.DEBUG.SAVE_HEATMAPS_PRED = False

    _C.defrost()
    _C.merge_from_file(config_file)
    _C.freeze()

    return _C

image_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
        transforms.Resize([255, 255])
    ])

def get_pretrained_model(
        config_file = './data/hrnet/mpii_hrnet_w32_255x255.yaml',
        model_weights = './data/hrnet/pretrained/pose_hrnet_w32_256x256.pth'
):
    cfg = get_hrnet_config_file(config_file)
    m = hrnet.get_pose_net(
        cfg, False
    )

    model_weights = torch.load(model_weights)

    m.load_state_dict(model_weights)
    m.eval()

    return m

def normalize_poses(poses, data_format='2d', k=32):
    # mean center 2d/3d poses and additionally divide 2d poses by std
    if data_format == '2d':
        poses_2d    = poses # N, 2, 16 or N, 2, 16, k
        poses_2d_x  = poses_2d[:, 0] # N, 16
        poses_2d_x  = (poses_2d_x - poses_2d_x.mean(axis=1, keepdims=True))
        poses_2d_y  = poses_2d[:, 1] # N, 16
        poses_2d_y  = (poses_2d_y - poses_2d_y.mean(axis=1, keepdims=True))
        poses       = np.concatenate((poses_2d_x, poses_2d_y), axis=1)
        poses       = poses / poses.std(axis=1, keepdims=True)
        if len(poses.shape) == 2: poses = poses.reshape(-1,2,16)
        else: poses = poses.reshape(-1,2,16,k)
    elif data_format == '3d':
        poses_3d    = poses * 1e-3 # N, 3, 17 [m -> mm]
        poses_3d_x  = poses_3d[:, 0]
        poses_3d_x  = poses_3d_x - poses_3d_x.mean(axis=1, keepdims=True)
        poses_3d_y  = poses_3d[:, 1]
        poses_3d_y  = poses_3d_y - poses_3d_y.mean(axis=1, keepdims=True)
        poses_3d_z  = poses_3d[:, 2]
        poses_3d_z  = poses_3d_z - poses_3d_z.mean(axis=1, keepdims=True)
        poses       = np.concatenate((poses_3d_x, -poses_3d_y, -poses_3d_z), axis=1)
        poses       = poses.reshape(-1,3,17)
    return poses


def test_sample(idx, img):
    with torch.inference_mode():
        heatmap     = hrnet(image_transforms(img).unsqueeze(0))[0].numpy().astype(np.float16)
    J, H, W     = heatmap.shape
    k           = 96
    topk_idx_unsorted = np.argpartition(-heatmap.reshape(J,-1), k, axis=1)[:, :k]
    topk_values = np.take_along_axis(heatmap.reshape(J,-1), topk_idx_unsorted, axis=1)
    topk_sorted_order = np.argsort(-topk_values, axis=1)
    sorted_ind  = np.take_along_axis(topk_idx_unsorted, topk_sorted_order, axis=1)
    inds_y, inds_x = (sorted_ind//H)/H, (sorted_ind%W)/W
    topk        = np.stack((inds_y,inds_x), axis=0) # 2, 16, k
    topks       = normalize_poses(topk.reshape(1,2,16,k),'2d',k)
    p2d_k       = torch.from_numpy(topks).transpose(1,2).float().to(c.device)

    T               = torch.linspace(0,1,c.n_steps).to(c.device)  # sample times
    x0_all          = 1.0 * torch.randn(c.n_hypo,17,3)
    p2d_hypos       = torch.stack([p2d_k[...,torch.randperm(c.args.topk)] for _ in range(c.n_hypo)])
    odefunc_ags     = partial(odefunc, p2d=p2d_hypos)
    solver          = ODESolver(velocity_model=WrappedModel(odefunc_ags))  # create an ODESolver class
    x1_all          = solver.sample(time_grid=T, x_init=x0_all.to(c.device), method=c.args.solver, step_size=c.step_size, return_intermediates=False)
    x1_all          = x1_all.detach().cpu()
    x1_all_ra       = x1_all - x1_all[:,0:1,:]
    best_hypo       = x1_all_ra.mean(0) # best pose is selected to be the mean pose
    
    # Visualization
    fig             = plt.figure(figsize=(10,10))
    mid, left, right = 'tab:green', 'tab:orange', 'tab:blue'
    B_mpii          = [[0,1], [1,2], [2,6], [5,4], [4,3], [3,6], [6,7], [7,8], [8,9], [8,12], [12,11], [11,10], [8,13], [13,14], [14,15]]
    C_mpii          = [right, right, right, left, left, left, mid, mid, mid, right, right, right, left, left, left]
    B_h36m          = [[0,1], [1,2], [2,3], [0,4], [4,5], [5,6], [0,7], [7,8], [8,9], [9,10], [8,11], [11,12], [12,13], [8,14], [14,15], [15,16]]
    C_h36m          = [right, right, right, left, left, left, mid, mid, mid, mid, left, left, left, right, right, right]
    
    ax              = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=10., azim=-30)
    ax.set_xlim([-1.2,1.2])
    ax.set_ylim([-1.0,1.0])
    ax.set_zlim([-0.0,1.0])
    ax.set_aspect('equal')
    ax.grid('off')
    ax.xaxis.set_pane_color((0.1, 0.1, 0.1, 0.00))
    ax.yaxis.set_pane_color((0.1, 0.1, 0.1, 0.00))
    ax.zaxis.set_pane_color((0.1, 0.1, 0.1, 0.02))
    ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.00))
    ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.00))
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.00))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    roi = np.asarray(img).copy() / 255.
    y = np.linspace(-0.6, 0.6, roi.shape[1])
    z = np.linspace(-0.0, 1.2, roi.shape[0])
    y, z = np.meshgrid(y, z)
    x = np.ones_like(y) * -1
    ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=np.flipud(roi), shade=False, zorder=0)
    
    best_hypo_m    = best_hypo.numpy()
    best_hypo_m[:,-1]  += 1.2
    min_joint       = np.argmin(best_hypo_m[:,-2])
    best_hypo_m[:,-2]  -= best_hypo_m[min_joint,-2]
    ax.scatter(best_hypo_m[:,2], best_hypo_m[:,0], best_hypo_m[:,1], c='k', zorder=5, s=10)
    for i, b_pair in enumerate(B_h36m):
        ax.plot(best_hypo_m[b_pair,2], best_hypo_m[b_pair,0], best_hypo_m[b_pair,1], lw=2, c=C_h36m[i])
    
    n_hypos = 100
    for h in range(n_hypos):
        x1_h_m          = x1_all_ra[h].numpy()
        x1_h_m[:,-1]    += 1.2                      # only for better visual
        min_joint       = np.argmin(x1_h_m[:,-2])   # only for better visual
        x1_h_m[:,-2]    -= x1_h_m[min_joint,-2]     # only for better visual
        for i, b_pair in enumerate(B_h36m):
            ax.plot(x1_h_m[b_pair,2], x1_h_m[b_pair,0], x1_h_m[b_pair,1], lw=1, c=C_h36m[i], alpha=0.05)
    # os.makedirs('./demo/results/LSP', exist_ok=True)
    # plt.savefig('./viz/results/LSP/'+str(idx)+'.png', dpi=200, format='png', bbox_inches='tight')
    os.makedirs('./demo/results/football', exist_ok=True)
    plt.savefig('./demo/results/football/'+str(idx)+'.png', dpi=200, format='png', bbox_inches='tight')
    plt.close()
    
    
if __name__ == "__main__":
    
    hrnet = get_pretrained_model()
    hrnet.eval()
    
    seed_everything(c.args.seed)
    odefunc         = FlowNet(hid_dim=1024, act_fn=c.args.act, k=c.args.topk, n_layers=c.args.num_gcn).to(c.device)
    odefunc.load(c.model_best, c.device)
    odefunc.eval()
    print("FlowNet num. params: ", sum(p.numel() for p in odefunc.parameters()))
    
    # Set your folder path
    # folder_path = './demo/demo_imgs/LSP'
    folder_path = './demo/demo_imgs/Football'

    # Supported image extensions
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')

    # Loop through all files in the folder
    for idx, filename in enumerate(os.listdir(folder_path)):
        if filename.lower().endswith(image_extensions):
            image_path = os.path.join(folder_path, filename)
            try:
                with Image.open(image_path) as img:
                    print(f'Processing: {filename} - Size: {img.size}, Format: {img.format}')
                    # You can perform your image operations here
                    test_sample(idx, img)
            except Exception as e:
                print(f'Error opening {filename}: {e}')
    
    
    print("----- Finito -----")