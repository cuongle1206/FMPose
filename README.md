# Official implementation of FMPose

Flow Matching for Probabilistic Monocular 3D Human Pose Estimation

[![Paper](https://img.shields.io/badge/arXiv-2601.16763-red)](https://arxiv.org/abs/2601.16763) [![OpenReview](https://img.shields.io/badge/OpenReview-TMLR-blue?logo=openreview)](https://openreview.net/forum?id=UlpH4XBLR4)

<div align="center">
<img src="figures/pipeline.png" width="1200" alt="logo"/>
</div>


# Environment setup

```bash
git clone https://github.com/cuongle1206/FMPose.git
cd FMPose
conda create -n fmpose python=3.9
conda activate fmpose
pip install -r requirements.txt
```

The experiments in the paper are conducted on an NVIDIA Tesla A40 GPU with 48GB RAM.

# Data prepration

For the experiments on the Human3.6M dataset:

- Download the Human3.6M database from their [website](http://vision.imar.ro/human3.6m/description.php).

- Clone and follow the instructions from [h36m-fetch](https://github.com/anibali/h36m-fetch) to obtain the ground truths in suitable format.
Run ```h36m-fetch/extract_all.py``` to extract all the zip file and the directory looks like this:
```
|-- h36m-fetch
|   |-- extracted
|   |   |-- S1
|   |   |   |-- Poses_D2_Positions
|   |   |   |-- Poses_D3_Positions_mono
|   |   |   |-- Videos
|   |   |   |-- ...
|   |   |-- ...
|   |   |-- S11
```

For the evaluations on the MPI_INF_3DHP dataset:

- Download the MPI_INF_3DHP dataset from their [website](https://vcai.mpi-inf.mpg.de/3dhp-dataset/).

- Run the script ```get_testset.sh``` inside the zip file to obtain the test folder, namely ```mpi_inf_3dhp_test_set```. The directory looks like this:
```
|-- mpi-inf-3dhp
|   |-- mpi_inf_3dhp_test_set
|   |   |-- TS1
|   |   |   |-- imageSequence
|   |   |   |-- annot_data.mat
|   |   |-- ...
|   |   |-- TS6
```

For the experiments on the 3DPW dataset:

- Download the 3DPW dataset from their [website](https://virtualhumans.mpi-inf.mpg.de/3DPW/).

- Unzip and use as it is. You can put those zip files into a folder ```3dpw/```, such as:
```
|-- 3dpw
|   |-- sequenceFiles
|   |   |-- test
|   |   |-- train
|   |   |-- validation
|   |-- imageFiles
|   |-- ...
```

# Generating 2D heatmaps with HRNet

The scripts for generating 2D heatmaps from images are respectively: ```H36M.py```, ```3DHP.py```, and ```3DPW.py``` inside the folder ```data/```.
Use the flag ```--use_orig_hrnet``` to toggle on/off the usage of fine-tuned HRNet from [Wehrbein et al.](https://github.com/twehrbein/Probabilistic-Monocular-3D-Human-Pose-Estimation-with-Normalizing-Flows).

Get the pretrained HRNet from our [Zenodo](https://zenodo.org/records/20367108?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjVlOTU2NTZhLTZiYWUtNGY2Zi05YmY3LTA1NTlmOGI4ZDIwMyIsImRhdGEiOnt9LCJyYW5kb20iOiJlODdjN2U2NTc4OGY2ZGNkYmZmNmY5NzE1NmNkMmVmNyJ9.vq2NNnZRmJdeLeefY4fKGq_3cYzaLQIZgiJ6FT5lfjomxW7Sf4wQh0v2YeC3GZvRjrl7HAxzY1yAq3fnQQ5x1g) project, and put them into the data folder:
```
|-- data
|   |-- pretrained
|   |   |-- *.pth
```

To generate the data used in our main experiments:
```bash
cd data
python H36M.py --input_dir ../h36m-fetch/processed/ --output_dir ./
python 3DHP.py --input_dir ../mpi-inf-3dhp/mpi_inf_3dhp_test_set/ --output_dir ./ --use_orig_hrnet
python 3DPW.py --input_dir ../3dpw/ --output_dir ./ --use_orig_hrnet
cd ..
```

Each script generates an h5 file called ```{dataset_name}_dataset_{orig/h36m}_hrnet.h5``` containing HRNet's predictions and GT 3D keypoints.

# Main experiments


### Flow matching installation

Follow the instuctions from [Lipman et al.](https://github.com/facebookresearch/flow_matching) to install from PyPI:
```bash
# Install flow matching
pip install flow_matching
```
or from source in developer mode:
```bash
# Flow matching installed from source
git clone https://github.com/facebookresearch/flow_matching
cd flow_matching
pre-commit install
pip install -e .
cd ..
```
Both versions are tested successfully.


### Generate GT data for 3DPW

Download the [SMPL](https://smpl.is.tue.mpg.de/) models and put them to ```data/smpl_models/smpl```.
We used the NEUTRAL model to obtain ground-truth 3D poses for 3DPW:

```
|-- data
|   |-- smpl_models
|   |   |-- smpl   
|   |   |   |-- SMPL_NEUTRAL
|   |   |   |-- SMPL_FEMALE
|   |   |   |-- SMPL_MALE
```

### Training and evaluation for FMPose

To run the main experiments of FMPose on the respective datasets:
```bash
# Train, valid and test on Human3.6M
python main_h36m.py --seed 42 --exp h36m --wandb

# Test on MPI-INF-3DHP, no fine-tuning
python eval_3dhp.py --seed 42 --exp 3dhp --wandb

# Train and test on 3DPW
python main_3dpw.py --seed 42 --exp 3dpw --wandb
```
You can change the seed value via ```--seed``` and toggle the usage of W&B with ```--wandb```.
The quantitative results are shown at the end of each script and also logged to W&B.

### Data loaders and pretrained FMPose

The preprocessed data loaders and pretrained FMPose models (seeds 40-44) can also be found in our [Zenodo](https://zenodo.org/records/20367108?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjVlOTU2NTZhLTZiYWUtNGY2Zi05YmY3LTA1NTlmOGI4ZDIwMyIsImRhdGEiOnt9LCJyYW5kb20iOiJlODdjN2U2NTc4OGY2ZGNkYmZmNmY5NzE1NmNkMmVmNyJ9.vq2NNnZRmJdeLeefY4fKGq_3cYzaLQIZgiJ6FT5lfjomxW7Sf4wQh0v2YeC3GZvRjrl7HAxzY1yAq3fnQQ5x1g).
Unzip and put the loader folders to ```data/```, the pretrained HRNet to ```data/hrnet/pretrained```, the models to ```models/trained_models```, and run the main scripts.
```
|-- data
|   |-- loader_h36m
|   |   | -- ...
|   |-- loader_3dhp
|   |   | -- ...
|   |-- loader_3dpw
|   |   | -- ...
|-- models
|   |-- trained_models
|   |   |-- h36m
|   |   |   | -- model_best_{topk}_{seed}.pt
|   |   |   | -- ...
|   |   |-- 3dpw
|   |   |   | -- model_last_{topk}_{seed}.pt
|   |   |   | -- ...
|   |-- model.py
```

# Inferencing and visualization

We additionally provide an inference code from single image inputs in the folder ```demo```.
The script performs the end-to-end prediction from an imput image to 3D human pose hypotheses and highlights the mean pose.
The demo images are collected from the [KTH-Football](https://www.csc.kth.se/cvap/cvg/?page=footballdataset2) and [Leeds Sport Pose](https://plmlab.math.cnrs.fr/chevallier-teaching/datasets/leeds-sport-pose) datasets.
```bash
# The script loops thru all demo images
python demo/inference.py
```

The pretrained HRNet and FMPose could also be found in our [HuggingFace](https://huggingface.co/datasets/makule/FMPose).

# Citation

If you find our project helpful, please cite the paper as

```bibtex
@article{le2026_fmpose,
    author    = {Le, Cuong and Melnyk, Pavlo and Wandt, Bastian and Wadenbäck, Mårten},
    title     = {Flow Matching for Probabilistic Monocular 3D Human Pose Estimation},
    journal   = {Transactions on Machine Learning Research},
    year      = {2026},
}
```

# Acknowledgements

This research is supported by the Wallenberg Artificial Intelligence, Autonomous Systems and Software Program (WASP), funded by Knut and Alice Wallenberg Foundation. The computational resources were provided by the National Academic Infrastructure for Supercomputing in Sweden (NAISS) at C3SE, and by the Berzelius resource, provided by the Knut and Alice Wallenberg Foundation at the National Supercomputer Centre.
