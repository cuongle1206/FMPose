import torch
from torchvision import transforms
import tables as tb
import pickle
from PIL import Image
from PIL import ImageOps
from pathlib import Path
import numpy as np
from tqdm import auto
from utils import get_pretrained_model, get_crop_bb_3dpw

class DatasetEntry(tb.IsDescription):
    action      = tb.StringCol(40)
    subject     = tb.StringCol(5)
    subaction   = tb.StringCol(1)
    cam         = tb.Float32Col(shape=(4, 4))
    frame_orig  = tb.IntCol(4)
    hardsubset  = tb.BoolCol(1)
    gt_3d       = tb.Float32Col(shape=(24, 3))
    gt_2d       = tb.Float32Col(shape=(2, 24))
    argmax_2d   = tb.Float32Col(shape=(2, 16))
    heatmap     = tb.Float16Col(shape=(16, 64, 64))
    poses       = tb.Float32Col(shape=(24, 3))
    betas       = tb.Float32Col(shape=(10,)) 

def main(args):
    """Get pretrained 2D detector model"""
    model, cfg = get_pretrained_model(use_h36m_model=not args.use_orig_hrnet)
    model.to('cuda')
    model.eval()

    if args.use_orig_hrnet: hrnet_suffix = 'orig_hrnet'
    else: hrnet_suffix = 'h36m_hrnet'
    datadir     = Path(args.input_dir)
    outputdir   = Path(args.output_dir)

    outputdir.mkdir(exist_ok=True, parents=True)
    h5file      = tb.open_file(outputdir.joinpath("3DPW_dataset_{}.h5".format(hrnet_suffix)), mode="w", title="Dataset 3DPW with heatmaps")

    group       = h5file.create_group('/', 'dataset', 'Dataset')
    table_train = h5file.create_table(group, 'trainset', DatasetEntry, "Training set")
    table_valid = h5file.create_table(group, 'validset', DatasetEntry, "Valid set")
    table_test  = h5file.create_table(group, 'testset', DatasetEntry, "Test set")
    
    entry_train = table_train.row
    generate_heatmaps_and_fill_dataset(entry_train, model, "train", datadir)
    table_train.flush()
    
    # entry_valid = table_valid.row
    # generate_heatmaps_and_fill_dataset(entry_valid, model, "valid", datadir)
    # table_valid.flush()
    
    entry_test  = table_test.row
    generate_heatmaps_and_fill_dataset(entry_test, model, "test", datadir)
    table_test.flush()


def generate_heatmaps_and_fill_dataset(entry, model, split, datadir, sample_interesting_frames=False):
    # Define the image_transforms used in the original code (+resize)
    image_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
        transforms.Resize([255, 255])
    ])

    if split == 'train':    subjects = ['train']
    elif split == 'valid':  subjects = ['validation']
    else:                   subjects = ['test']

    for subject in subjects:
        sequence_dir = datadir.joinpath('sequenceFiles', subject)
        image_dir = datadir.joinpath("imageFiles")

        for sequence in auto.tqdm(sorted(sequence_dir.iterdir()), desc='Sequence'):
            print('Processing seq:{} ({})'.format(sequence, subject))
            u = pickle._Unpickler(open(sequence, 'rb'))
            u.encoding = 'latin1'
            data = u.load()

            seq_name = data['sequence']
            img_frames = data['img_frame_ids']  # Not in use
            
            for person_idx in range(len(data['poses2d'])):
                poses2d = data['poses2d'][person_idx]  # Shape Nx3x18
                poses3d = data['jointPositions'][person_idx].reshape(-1, 24, 3) # in world coordinate
                img_dir = datadir.joinpath("imageFiles", seq_name)
                smpl_poses = data['poses'][person_idx]
                smpl_betas = data['betas'][person_idx]
                
                
                for fnr, (p2d, p3d, thetas) in enumerate(zip(poses2d, poses3d, smpl_poses)):
                    img_fn = img_dir.joinpath('image_{:05d}.jpg'.format(fnr))
                    img = Image.open(img_fn)

                    if np.all(p2d[2] == 0):
                        continue
                    
                    extrinsics  = data['cam_poses'][fnr]
                    intrinsics  = data['cam_intrinsics']
                    p3d_wc      = np.concatenate((p3d, np.ones((24,1))), axis=-1) # 24 x 4, wc homog
                    p3d_cc      = (extrinsics @ p3d_wc.T)[:-1,:] # 3 x 24, cc
                    p3d_ic      = intrinsics @ p3d_cc
                    p2d         = p3d_ic[:2,:] / p3d_ic[-1] # 2 x 24, pixel
                    crop_size, crop_bb = get_crop_bb_3dpw(p2d, img.size[::-1])
                    roi = img.crop(crop_bb)
                    roi_paded = ImageOps.pad(roi, (crop_size, crop_size), color=(0, 0, 0))
                    roi_paded = np.asarray(roi_paded).copy()
                    
                    preprocessed_img = image_transforms(roi_paded)
                    with torch.inference_mode():
                        prediction = model(preprocessed_img.unsqueeze(0).cuda())[0].cpu()

                    # Convert predictions to numpy and change to float16 precision
                    prediction = prediction.numpy().astype(np.float16)
                    
                    # Naively find the argmax of each joint in 64x64 map and normalize to [0, 1] coords
                    peak = np.unravel_index(np.argmax(prediction.reshape(prediction.shape[0], -1), axis=1), prediction.shape[1:])
                    peak_np = np.asarray(peak, dtype=float) / prediction.shape[1]

                    # Fill the data entry with all necessary data
                    entry["heatmap"]    = prediction
                    entry["subject"]    = subject
                    entry["action"]     = seq_name
                    entry["subaction"]  = "none"
                    entry["cam"]        = extrinsics
                    entry["frame_orig"] = fnr

                    entry["gt_3d"]      = p3d # 24 x 3
                    entry["gt_2d"]      = p2d # 2 x 24
                    entry["argmax_2d"]  = peak_np
                    entry["poses"]      = thetas.reshape(24,3)
                    entry["betas"]      = smpl_betas[:10]

                    entry.append()
    return entry


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str,
                        help='Input directory in which the subject data resides',
                        default='../3dpw/')
    parser.add_argument('--output_dir', type=str,
                        help='Output directory to which the dataset will stored',
                        default='./')
    parser.add_argument('--use_orig_hrnet', action='store_true',
                        help='Use original hrnet weights instead of the weights pretrained on Human3.6M')
    _args = parser.parse_args()

    main(_args)