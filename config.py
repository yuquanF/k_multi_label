import os
import torch
from neau_hpc_utils.helper import LogTrainValid


# device params
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0")
device_id = [0]

# global params
batch_size = 64
img_size = 224
root = '/stu07/Remote/stu07/Remote/Apple/Kaggle_plant_pathology/'

# model params
model_name = 'resnet50'
num_classes = 6
pretrained = True

# data params
train_annotations_file = os.path.join(root, 'output_s_m', "train_label.csv")
valid_annotations_file = os.path.join(root, 'output_s_m', "valid_label.csv")

# train params
epochs = 30
patience = 7  # early stopping patience; how long to wait after last time validation loss improved.

# log
log = LogTrainValid()
