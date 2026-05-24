import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import tables as tb
from time import time
import smplx
import config as c
from alive_progress import alive_bar

num_joints_3d = 23
num_joints_2d = 14
num_cond_joints = 16

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
        poses_3d    = poses # N, 24, 3
        poses_3d_x  = poses_3d[:, :, 0]
        poses_3d_x  = poses_3d_x - poses_3d_x.mean(axis=1, keepdims=True)
        poses_3d_y  = poses_3d[:, :, 1]
        poses_3d_y  = poses_3d_y - poses_3d_y.mean(axis=1, keepdims=True)
        poses_3d_z  = poses_3d[:, :, 2]
        poses_3d_z  = poses_3d_z - poses_3d_z.mean(axis=1, keepdims=True)
        poses       = np.stack((poses_3d_x, poses_3d_y, poses_3d_z), axis=-1)
    
    elif data_format == '3d-torch':
        poses_3d    = poses # N, 17, 3
        poses_3d_x  = poses_3d[:, :, 0]
        poses_3d_x  = poses_3d_x - poses_3d_x.mean(axis=1, keepdims=True)
        poses_3d_y  = poses_3d[:, :, 1]
        poses_3d_y  = poses_3d_y - poses_3d_y.mean(axis=1, keepdims=True)
        poses_3d_z  = poses_3d[:, :, 2]
        poses_3d_z  = poses_3d_z - poses_3d_z.mean(axis=1, keepdims=True)
        poses       = torch.stack((poses_3d_x, -poses_3d_y, -poses_3d_z), axis=-1)
    
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

class PW3DDataset(Dataset):
    """
    3DPW dataset
    """
    def __init__(self, device='cuda', mode='train', hr_version='org'):
        if hr_version == 'h36m':
            h5_file = './data/3DPW_dataset_h36m_hrnet.h5'
        elif hr_version == 'org':
            h5_file = './data/3DPW_dataset_orig_hrnet.h5'
        dataset = tb.open_file(h5_file, mode="r")
        self.device = device

        if mode == 'train': datasplit = dataset.root.dataset.trainset
        elif mode == 'valid': datasplit = dataset.root.dataset.validset
        else: datasplit = dataset.root.dataset.testset

        self.frame_info = []  
        poses_3d        = []
        p2d_gt          = []
        p2d_hrnet_unnorm = []
        topks_unnorm    = []
        thetas          = []
        betas           = []
        cams            = []
        k = 96

        t_start         = time()
        with alive_bar(refresh_secs=0.2) as bar:
            for i, sample in enumerate(datasplit.iterrows(stop=None, step=1)):
                self.frame_info.append([sample['subject'].decode('utf-8'), sample['action'].decode('utf-8'),
                                        sample['frame_orig'], sample['cam']])
                
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
                
                thetas.append(sample['poses'])
                betas.append(sample['betas'])
                cams.append(sample['cam'])
                bar()

        dataset.close()
        print("Took {:.1f} seconds.".format(time()-t_start))
        
        # list to np array
        poses_3d        = np.stack(poses_3d, axis=0)
        p2d_gt          = np.stack(p2d_gt, axis=0)
        self.p2d_gt     = p2d_gt
        p2d_hrnet_unnorm = np.stack(p2d_hrnet_unnorm, axis=0)
        # topks_unnorm    = np.stack(topks_unnorm, axis=0)
        
        # H36m skeleton format GT
        thetas          = torch.from_numpy(np.stack(thetas)).to("cuda")
        betas           = torch.from_numpy(np.stack(betas)).to("cuda")
        trans           = torch.from_numpy(poses_3d[:,0,:]).to("cuda")
        cams            = torch.from_numpy(np.stack(cams))
        smpl_model      = smplx.create(model_path = "./data/smpl_models",
                                       model_type = "smpl",
                                       gender = "neutral",
                                       use_face_contour = False,
                                       num_betas = 10,
                                       num_expression_coeffs = 10,
                                       ext = "npz").to("cuda")
        
        seq_pred        = smpl_model(betas = betas.reshape(-1,10),
                                     global_orient = thetas[:,0:1,:],
                                     body_pose = thetas[:,1:,:],
                                     transl = trans,
                                     return_verts = True)
        mesh            = seq_pred.vertices
        J_regressor_h36m = torch.from_numpy(np.load("./data/J_regressor_h36m.npy")).to("cuda").float()
        p3d_17j         = torch.einsum('nvc,jv -> njc', mesh, J_regressor_h36m).detach().cpu()
        p3d_17j[:,[1,2,3,4,5,6]] = p3d_17j[:,[4,5,6,1,2,3]]
        p3d_17j_wc      = torch.cat((p3d_17j, torch.ones((p3d_17j.shape[0], p3d_17j.shape[1], 1))), axis=-1) # N x 17 x 4, wc homog
        p3d_17j_cc      = torch.bmm(p3d_17j_wc, cams.permute(0,2,1))[:,:,:-1] # N x 17 x 3, cc
        p3d_gt_17j      = ((p3d_17j_cc - p3d_17j_cc[:,0:1,:])*1e3)# root-aligned, zero center
        p3d_gt_17j[:, :, 1:] *= -1                          # invert y and z
        self.p3d_gt_17j = (p3d_gt_17j).float()
        self.p3d_17j_cc = normalize_poses(p3d_17j_cc, '3d-torch').float()
        
        # preprocess 2d pose
        self.p2d_hrnet_unnorm = torch.from_numpy(p2d_hrnet_unnorm).transpose(1,2).float()
        p2d_hrnet       = normalize_poses(p2d_hrnet_unnorm,'2d') # standardized to mean 0 and std 1
        self.p2d_hrnet  = torch.from_numpy(p2d_hrnet).transpose(1,2).float()

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
        return self.p3d_17j_cc.size(0)

    def __getitem__(self, idx):
        return {
            'p3d_gt_17j': self.p3d_gt_17j[idx],
            'p3d_17j': self.p3d_17j_cc[idx],
            'p2d_gt': self.p2d_gt[idx],
            'p2d_unnorm': self.p2d_hrnet_unnorm[idx],
            'p2d_topk': self.topks[idx],
            'frame_info': self.frame_info[idx]
            }
