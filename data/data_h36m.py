import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import tables as tb
from time import time
import config as c
from alive_progress import alive_bar

subject_all     = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
subjects_train  = ['S1', 'S5', 'S6', 'S7', 'S8']
subjects_test   = ['S9', 'S11']
actions_all     = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Photo', 'Posing', 'Purchases',
                   'Sitting', 'SittingDown', 'Smoking', 'Waiting', 'WalkDog', 'WalkTogether', 'Walking']
subactions_all  = ['0', '1']
cameras_all     = ['54138969', '55011271', '58860488', '60457274']

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

class H36MDataset(Dataset):
    def __init__(self, quick_eval=False, quick_eval_stride=16, train_set=True, actions=actions_all, hardsubset=False):
        h5_file         = './data/H36M_dataset_h36m_hrnet.h5'
        dataset         = tb.open_file(h5_file, mode="r")
        if train_set:
            datasplit       = dataset.root.dataset.trainset
            subjects_to_use = subjects_train
        else:
            datasplit       = dataset.root.dataset.testset
            subjects_to_use = subjects_test
        
        self.frame_info  = []  
        poses_3d    = []
        p2d_gt      = []
        p2d_hrnet_unnorm = []
        topks_unnorm = []
        
        k = 96
        if quick_eval: stride = quick_eval_stride
        else: stride = 1
        
        t_start         = time()
        with alive_bar(refresh_secs=0.2) as bar:
            for i, sample in enumerate(datasplit.iterrows(stop=None, step=stride)):
                if sample['action'].decode('utf-8') not in actions:
                    continue
                if sample['subject'].decode('utf-8') not in subjects_to_use:
                    continue
                
                if hardsubset:
                    if 'hardsubset' in datasplit.coldescrs.keys():
                        if not sample['hardsubset']: continue
                    else:
                        print('Hardsubset is not available')
                        return
                
                self.frame_info.append([sample['action'].decode('utf-8'), sample['subject'].decode('utf-8'),
                                        sample['subaction'].decode('utf-8'), sample['cam'].decode('utf-8'),
                                        sample['frame_orig']])
                poses_3d.append(sample['gt_3d'])
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
        p2d_gt      = np.stack(p2d_gt, axis=0)
        p2d_hrnet_unnorm = np.stack(p2d_hrnet_unnorm, axis=0)
        
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
            'p2d_unnorm': self.p2d_hrnet_unnorm[idx],
            'p2d': self.p2d_hrnet[idx],
            'p2d_topk': self.topks[idx],
            'frame_info': self.frame_info[idx]
            }
