# This is a configuration file
import torch
from torch.nn.functional import threshold


# config for video dataset
start_frames_video = [0, 0, 1150, 210, 750, 0, 289, 110, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
raw_data_path = '/usr/bmicnas01/data-biwi-01/bmicdatasets-originals/Originals/SerraVideos'
frames_path = '/scratch_net/biwidl214/zahnerj/data/'
data_list_path = '/home/zahnerj/MA_endo_julia/preproc/data_lists/vid/'

# paths to csv-files with image paths and label coordinates
base_data_path = '/home/zahnerj/MA_endo_julia/preproc/data_lists/setup/all.csv'
train_data_path = '/home/zahnerj/MA_endo_julia/preproc/data_lists/split/sorted/train.txt'
test_data_path = '/home/zahnerj/MA_endo_julia/preproc/data_lists/split/sorted/test.txt'

# general parameters
num_of_patients = 26

# parameters fro image transformations
image_resized_size = (256, 455)
image_cropped_size = (256, 256)


# network parameters
input_channels = 3
output_channels = 10
features = [8, 16, 32, 64, 64]
dims = [4096, 1024, 128]

# pretrain parameters
lr_encoder_pretrain = 0.01
momentum = 0.9
batch_size = 10
cycle = 0

# learning rate for main training
lr_full = 0.008

# threshold for classification (not needed for BCE)
thres = 0.75

# number of labeled frames per label type (for weighting in weighted regression)
counter = torch.tensor([164, 100, 674, 1130, 93, 118, 1386, 60, 25, 14])

# optimizer
opti_name = 'Adam'
# opti_name = 'SGD' # (tried out not used)

# checkpoint to load for prediction (baseline)
baseline_checkpoint = 'unet_baseline_lr={}_optim={}.pth'.format(
    lr_full, opti_name)

# checkpoint to load for prediction (full)
pred_checkoint = 'Class_simple_Reg_weighted_50.pth'

# folder to save predictions
pred_name = 'Class_simple_Reg_weighted_50/'

# checkpoint to load for training (pretrain)
pretrain_checkpoint_save = 'pretrain_contrastive_lr={}_optim=Adam_cycle={}'.format(
    lr_encoder_pretrain, cycle)
pretrain_checkpoint_load = 'pretrain_contrastive_lr={}_optim=Adam_cycle={}'.format(
    lr_encoder_pretrain, cycle-1)

# general save path for with prediction, visualization, heatmaps, ifs_overlays, checkpoint directories
save_path = '/scratch_net/biwidl214/zahnerj/data/pred_lab/'

# path to save heatmaps
save_path_hmp = '/scratch_net/biwidl214/zahnerj/data/pred_lab/heatmaps/'

# path to save predictions as numpy arrays for:
# main training
save_path_pred = '/scratch_net/biwidl214/zahnerj/data/pred_lab/predictions/'
# baseline training
save_path_pred_baseline = '/scratch_net/biwidl214/zahnerj/data/pred_lab/predictions/baseline_lr={}_optim={}/'.format(
    lr_full, opti_name)


# paths to save checkpoints from training for:
# main training
save_path_ckp = '/scratch_net/biwidl214/zahnerj/data/pred_lab/checkpoints/'
# baseline training
save_path_baseline = '/scratch_net/biwidl214/zahnerj/data/baseline/checkpoints/'
# pretraining
save_path_ckp_pretrain = '/scratch_net/biwidl214/zahnerj/data/pretrain/checkpoints/'

# path to safe generated gifs
gif_path = '/scratch_net/biwidl214/zahnerj/data/pred_lab/gifs_overlays/' + pred_name

# path to save images
save_path_visual = '/scratch_net/biwidl214/zahnerj/data/pred_lab/visualization/'
