import os
import random
import time
import numpy as np
import torch
import config as c

# 17 Joints H36M skeleton
H36M_NAMES = ['']*17
H36M_NAMES[0]  = 'Hip'
H36M_NAMES[1]  = 'RHip'
H36M_NAMES[2]  = 'RKnee'
H36M_NAMES[3]  = 'RFoot'
H36M_NAMES[4]  = 'LHip'
H36M_NAMES[5]  = 'LKnee'
H36M_NAMES[6]  = 'LFoot'
H36M_NAMES[7] = 'Spine'
H36M_NAMES[8] = 'Thorax'
H36M_NAMES[9] = 'Neck/Nose'
H36M_NAMES[10] = 'Head'
H36M_NAMES[11] = 'LShoulder'
H36M_NAMES[12] = 'LElbow'
H36M_NAMES[13] = 'LWrist'
H36M_NAMES[14] = 'RShoulder'
H36M_NAMES[15] = 'RElbow'
H36M_NAMES[16] = 'RWrist'

# 16 joints MPII skeleton used for the HRNet detections
MPII_NAMES = ['']*16
MPII_NAMES[0]  = 'RFoot'
MPII_NAMES[1]  = 'RKnee'
MPII_NAMES[2]  = 'RHip'
MPII_NAMES[3]  = 'LHip'
MPII_NAMES[4]  = 'LKnee'
MPII_NAMES[5]  = 'LFoot'
MPII_NAMES[6]  = 'Hip'
MPII_NAMES[7]  = 'Spine'
MPII_NAMES[8]  = 'Thorax'
MPII_NAMES[9]  = 'Head'
MPII_NAMES[10] = 'RWrist'
MPII_NAMES[11] = 'RElbow'
MPII_NAMES[12] = 'RShoulder'
MPII_NAMES[13] = 'LShoulder'
MPII_NAMES[14] = 'LElbow'
MPII_NAMES[15] = 'LWrist'

bones = {'h36m': np.array([[0, 1], [1, 2], [3, 4], [4, 5], [6, 7], [7, 8], [8, 9], [7, 10], [10, 11], [11, 12], [7, 13], [13, 14], [14, 15]])}

# select and permute h36m joints to be consistent with MPII skeleton
H36M17j_TO_MPII = [3, 2, 1, 4, 5, 6, 0, 7, 8, 10, 16, 15, 14, 11, 12, 13, 9]

ticks = ['RFoot', 'RKnee', 'RHip', 'LHip', 'LKnee', 'LFoot', 'Hip', 'Spine', 'Thorax', 'Head', 'RWrist', 'RElbow', 'RShoul', 'LShoul', 'LElbow', 'LWrist']

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def print_log(str, print_time=True):
    if print_time:
        localtime = time.asctime(time.localtime(time.time()))
        str = "[ " + localtime + ' ] ' + str
    print(str)

def create_loaders_h36m(dataset):
    print("Loading training dataset. Current seed: " + str(c.args.seed) + "\n")
    t_start         = time.time()
    os.makedirs('./data/loader_h36m/', exist_ok=True)
    if c.args.sampling: loader_path = "./data/loader_h36m/seed_" + str(c.args.seed) + "/" 
    else: loader_path = "./data/loader_h36m/"
    trainset_path   = loader_path + 'trainset.pt'
    
    if (os.path.exists(trainset_path)):
        print("Trainset already exists, loading")
        with open(trainset_path, "rb") as f: train_dataset = torch.load(f)
    else:
        print("Trainset do not exist, creating")
        train_dataset   = dataset.H36MDataset(train_set=True)
        with open(trainset_path, "wb") as f: torch.save(train_dataset, f)
    
    valset_path     = loader_path + 'valset.pt'
    if (os.path.exists(valset_path)):
        print("Valset (hard) already exists, loading")
        with open(valset_path, "rb") as f: val_dataset = torch.load(f)
    else:
        print("Valset (hard) set do not exist, creating")
        val_dataset = dataset.H36MDataset(train_set=False, quick_eval=True, hardsubset=True)
        with open(valset_path, "wb") as f: torch.save(val_dataset, f)
    
    easyset_path    = loader_path + 'easyset.pt'
    if (os.path.exists(easyset_path)):
        print("Easyset already exists, loading")
        with open(easyset_path, "rb") as f: easy_dataset = torch.load(f)
    else:
        print("Easyset set do not exist, creating")
        easy_dataset = dataset.H36MDataset(train_set=False, quick_eval=True)
        with open(easyset_path, "wb") as f: torch.save(easy_dataset, f)
    
    hardset_path    = loader_path + 'hardset.pt'
    if (os.path.exists(hardset_path)):
        print("Hardset already exists, loading")
        with open(hardset_path, "rb") as f: hard_dataset = torch.load(f)
    else:
        print("Hardset set do not exist, creating")
        hard_dataset    = dataset.H36MDataset(train_set=False, hardsubset=True)
        with open(hardset_path, "wb") as f: torch.save(hard_dataset, f)
    
    print("Took {:.2f} seconds.\n".format(time.time()-t_start))
    print("Number of training samples: ", len(train_dataset))
    print("Number of validate samples: ", len(val_dataset))
    print("Number of easytest samples: ", len(easy_dataset))
    print("Number of hardtest samples: ", len(hard_dataset))
    print()

    return train_dataset, val_dataset, easy_dataset, hard_dataset

def create_loaders_3dhp(dataset, hr_version='org'):
    print("Loading training dataset. Current seed: " + str(c.args.seed) + "\n")
    t_start         = time.time()
    os.makedirs('./data/loader_3dhp/', exist_ok=True)
    if c.args.sampling: loader_path = "./data/loader_3dhp/seed_" + str(c.args.seed) + "/" 
    else: loader_path = "./data/loader_3dhp/"
    testset_path    = loader_path + 'testset_'+hr_version+'.pt'
    GSset_path      = loader_path + 'GSset_'+hr_version+'.pt'
    noGSset_path    = loader_path + 'noGSset_'+hr_version+'.pt'
    outdoorset_path = loader_path + 'outdoorset_'+hr_version+'.pt'
    
    if (os.path.exists(testset_path)):
        print("Testset already exists, loading")
        with open(testset_path, "rb") as f: testset = torch.load(f)
    else:
        print("Testset set do not exist, creating")
        testset     = dataset.HPDatasetH5(mode='all', hr_version=hr_version)
        with open(testset_path, "wb") as f: torch.save(testset, f)
            
    if (os.path.exists(GSset_path)):
        print("GSset already exists, loading")
        with open(GSset_path, "rb") as f: GSset = torch.load(f)
    else:
        print("GSset set do not exist, creating")
        GSset       = dataset.HPDatasetH5(mode='GS', hr_version=hr_version)
        with open(GSset_path, "wb") as f: torch.save(GSset, f)
            
    if (os.path.exists(noGSset_path)):
        print("noGSset already exists, loading")
        with open(noGSset_path, "rb") as f: noGSset = torch.load(f)
    else:
        print("noGSset set do not exist, creating")
        noGSset     = dataset.HPDatasetH5(mode='noGS', hr_version=hr_version)
        with open(noGSset_path, "wb") as f: torch.save(noGSset, f)
            
    if (os.path.exists(outdoorset_path)):
        print("Outdoor set already exists, loading")
        with open(outdoorset_path, "rb") as f: outdoorset = torch.load(f)
    else:
        print("Outdoor set set do not exist, creating")
        outdoorset = dataset.HPDatasetH5(mode='out', hr_version=hr_version)
        with open(outdoorset_path, "wb") as f: torch.save(outdoorset, f)
        
    print("Took {:.2f} seconds.\n".format(time.time()-t_start))
    print("Number of testing samples: ", len(testset))
    print("Number of GS samples: ", len(GSset))
    print("Number of no GS samples: ", len(noGSset))
    print("Number of outdoor samples: ", len(outdoorset))
    print()

    return testset, GSset, noGSset, outdoorset

def create_loaders_3dpw(dataset):
    print("Loading training dataset. Current seed: " + str(c.args.seed) + "\n")
    t_start         = time.time()
    os.makedirs('./data/loader_3dpw/', exist_ok=True)
    loader_path     = "./data/loader_3dpw/"
    
    trainset_path   = loader_path + 'trainset.pt'
    if (os.path.exists(trainset_path)):
        print("Trainset already exists, loading")
        with open(trainset_path, "rb") as f: train_dataset = torch.load(f)
    else:
        print("Trainset do not exist, creating")
        train_dataset   = dataset.PW3DDataset(c.device, mode='train')
        with open(trainset_path, "wb") as f: torch.save(train_dataset, f)
    
    testset_path    = loader_path + 'testset.pt'
    if (os.path.exists(testset_path)):
        print("Testset already exists, loading")
        with open(testset_path, "rb") as f: test_dataset = torch.load(f)
    else:
        print("Testset set do not exist, creating")
        test_dataset = dataset.PW3DDataset(c.device, mode='test')
        with open(testset_path, "wb") as f: torch.save(test_dataset, f)
    
    print("Took {:.2f} seconds.\n".format(time.time()-t_start))
    print("Number of training samples: ", len(train_dataset))
    print("Number of testing samples:  ", len(test_dataset))
    print()

    return train_dataset, test_dataset

def get_crop_bb(joints, img_shape):
    """Estimate the crop based bounding box by finding the max/min joint coords and add a margin"""
    bb_coords = np.concatenate([joints.max(axis=0), joints.min(axis=0)])
    bb_center = np.round(0.5 * (bb_coords[0:2] + bb_coords[2:]))
    bb_width  = np.abs(bb_coords[0] - bb_coords[2])
    bb_height = np.abs(bb_coords[1] - bb_coords[3])
    crop_size = np.round(max(bb_width, bb_height) * 1.2)
    if crop_size < 255: crop_size = 255
    crop_bb = np.round(np.concatenate([bb_center -0.5*crop_size, bb_center +0.5*crop_size])).astype(np.int32)
    return crop_size, crop_bb

def get_crop_bb_3dpw(joints, img_shape):
    """Estimate the crop based bounding box by finding the max/min joint coords and add a margin"""
    bb_coords   = np.concatenate([joints.min(axis=1),joints.max(axis=1)])
    bb_center   = np.round(0.5 * (bb_coords[:2] + bb_coords[2:]))
    bb_width    = np.abs(bb_coords[0] - bb_coords[2])
    bb_height   = np.abs(bb_coords[1] - bb_coords[3])
    bb_radius   = np.asarray([bb_width, bb_height]) * 1.2
    crop_size   = int(np.round(bb_radius.max()))
    crop_bb     = np.round(np.concatenate([bb_center - 0.5*bb_radius, bb_center + 0.5*bb_radius])).astype(np.int32)
    return crop_size, crop_bb

def get_crop_bb_3dpw_vis(joints, img_shape):
    """Estimate the crop based bounding box by finding the max/min joint coords and add a margin"""
    bb_coords = np.concatenate([joints.max(axis=1), joints.min(axis=1)])
    bb_center = np.round(0.5 * (bb_coords[0:2] + bb_coords[2:]))
    bb_width  = np.abs(bb_coords[0] - bb_coords[2])
    bb_height = np.abs(bb_coords[1] - bb_coords[3])
    crop_size = np.round(max(bb_width, bb_height) * 1.2)
    if crop_size < 255: crop_size = 255
    crop_bb = np.round(np.concatenate([bb_center -0.5*crop_size, bb_center +0.5*crop_size])).astype(np.int32)
    return crop_size, crop_bb