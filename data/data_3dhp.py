import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import tables as tb
from time import time
import config as c
from alive_progress import alive_bar

def normalize_poses(poses, data_format='2d',k=32):
    # mean center 2d/3d poses and additionally divide 2d poses by std
    if data_format == '2d':
        poses_2d    = poses # N, 2, 16 or N, 2, 16, K
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

def normalize_samples(poses, k=128):
    # mean center 2d/3d poses and additionally divide 2d poses by std
    poses_2d    = poses # N, J, k
    poses_2d_x  = poses_2d[:, 0] # N, J, k
    poses_2d_x  = (poses_2d_x - poses_2d_x.mean(dim=1, keepdim=True))
    poses_2d_y  = poses_2d[:, 1] # N, J, k
    poses_2d_y  = (poses_2d_y - poses_2d_y.mean(dim=1, keepdim=True))
    poses       = torch.cat((poses_2d_x, poses_2d_y), axis=1)
    poses       = poses / poses.std(dim=1, keepdim=True)
    poses       = poses.reshape(-1,2,16,k)
    return poses

class HPDatasetH5(Dataset):
    """
    MPI-INF-3DHP Dataset (https://vcai.mpi-inf.mpg.de/3dhp-dataset/)
    """
    def __init__(self, mode='all', hr_version='org'):
        if hr_version == 'h36m':
            h5_file         = './data/3DHP_dataset_h36m_hrnet.h5'
        elif hr_version == 'org':
            h5_file         = './data/3DHP_dataset_orig_hrnet.h5'
        dataset         = tb.open_file(h5_file, mode="r")
        datasplit       = dataset.root.dataset.testset
        if mode == 'all':       subjects_to_use = ['TS1', 'TS2', 'TS3', 'TS4', 'TS5', 'TS6']
        elif mode == 'GS':      subjects_to_use = ['TS1', 'TS2']
        elif mode == 'noGS':    subjects_to_use = ['TS3', 'TS4']
        elif mode == 'out':     subjects_to_use = ['TS5', 'TS6']
        else: raise ValueError("Invalid mode")
        
        self.frame_info = []  
        poses_3d        = []
        p2d_gt          = []
        p2d_hrnet_unnorm = []
        topks_unnorm    = []
        k = 96

        t_start         = time()
        with alive_bar(refresh_secs=0.2) as bar:
            for i, sample in enumerate(datasplit.iterrows(stop=None, step=1)):
                if sample['subject'].decode('utf-8') not in subjects_to_use:
                    continue

                self.frame_info.append([sample['subject'].decode('utf-8'), sample['frame_orig']])
                poses_3d.append(sample['gt_3d_uni'])
                p2d_gt.append(sample['gt_2d'])
                p2d_hrnet_unnorm.append(sample['argmax_2d'])

                if c.args.sampling:
                    heatmap     = torch.from_numpy(sample['heatmap'])
                    heatmap[heatmap < 0] = 0 # negs -> zeros
                    J, H, W     = heatmap.shape
                    heatmap_flat = heatmap.view(J,-1)
                    samples     = torch.multinomial(heatmap_flat, k, replacement=True)
                    inds_y, inds_x = (samples//H)/H, (samples%W)/W
                    sample_coords = torch.stack([inds_y, inds_x], dim=0) # 2, J, k
                    topks_unnorm.append(sample_coords)
                else:
                    heatmap     = sample['heatmap']
                    J, H, W     = heatmap.shape
                    topk_idx_unsorted = np.argpartition(-heatmap.reshape(J,-1), k, axis=1)[:, :k]
                    topk_values = np.take_along_axis(heatmap.reshape(J,-1), topk_idx_unsorted, axis=1)
                    topk_sorted_order = np.argsort(-topk_values, axis=1)
                    sorted_ind  = np.take_along_axis(topk_idx_unsorted, topk_sorted_order, axis=1)
                    inds_y, inds_x = (sorted_ind//H)/H, (sorted_ind%W)/W
                    topk        = np.stack((inds_y,inds_x), axis=0) # 2, 16, k
                    topks_unnorm.append(topk)
                bar()

        dataset.close()
        print("Took {:.1f} seconds.".format(time()-t_start))
        
        # list to np array
        poses_3d    = np.stack(poses_3d, axis=0)
        self.p2d_gt      = np.stack(p2d_gt, axis=0)
        p2d_hrnet_unnorm = np.stack(p2d_hrnet_unnorm, axis=0)
        # topks_unnorm = np.stack(topks_unnorm, axis=0)

        # preprocess 3d gt poses
        p3d_gt      = poses_3d.copy()
        p3d_gt      -= p3d_gt[:, :, 0, None]            # root center gt poses
        p3d_gt[:, 1:, :] *= -1                          # invert y and z
        self.p3d_gt = torch.from_numpy(p3d_gt).transpose(1,2).float() # origin-aligned 3d pose N,17,3
        
        poses_3d    = normalize_poses(poses_3d,'3d')    # just subtract the mean
        self.poses_3d = torch.from_numpy(poses_3d).transpose(1,2).float() # mean-aligned 3d pose N,17,3
        # -----------------------------------------

        # preprocess 2d pose
        self.p2d_hrnet_unnorm = torch.from_numpy(p2d_hrnet_unnorm).transpose(1,2).float()
        p2d_hrnet   = normalize_poses(p2d_hrnet_unnorm,'2d') # standardized to mean 0 and std 1
        self.p2d_hrnet = torch.from_numpy(p2d_hrnet).transpose(1,2).float()

        if c.args.sampling:
            # preprocess sampled k 2d poses
            hm_samples  = torch.stack(topks_unnorm, dim=0)
            ks          = normalize_samples(hm_samples, k)
            self.topks  = ks.transpose(1,2).float()
        else:
            # preprocess top-k 2d poses
            topks_unnorm = np.stack(topks_unnorm, axis=0)
            topks       = normalize_poses(topks_unnorm,'2d',k)
            self.topks  = torch.from_numpy(topks).transpose(1,2).float()
        # -----------------------------------------

    def __len__(self):
        return self.poses_3d.size(0)

    def __getitem__(self, idx):
        return {
            'p3d_gt': self.p3d_gt[idx],
            'p3d': self.poses_3d[idx],
            'p2d_gt': self.p2d_gt[idx],
            'p2d_unnorm': self.p2d_hrnet_unnorm[idx],
            'p2d': self.p2d_hrnet[idx],
            'p2d_topk': self.topks[idx],
            'frame_info': self.frame_info[idx]
            }
