import torch
import json
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from net.modules import U_net
from preproc.load_data import Dataset_pyt
import numpy as np
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb
import torch.nn as nn
import config as cfg
from preproc.distortion import BloodBlobRandom
from train import EncoderPretrain
from torch._C import device
from train_whole import UnetModule
import os

# function to predict on test set and save predictions/heatmapts as numpy arrays in the prediction folder ------------------------------


def predict():

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(cfg.image_cropped_size),
        BloodBlobRandom(20, 80, 0.01)
    ])

    test_dataset = Dataset_pyt(
        cfg.base_data_path, cfg.test_data_path, transform=data_transform)
    test_data = DataLoader(test_dataset, num_workers=4)

    # load model from last checkpoint of main training
    model = UnetModule.load_from_checkpoint(
        cfg.save_path_ckp + cfg.pred_checkoint)

    trainer = pl.Trainer(gpus=1)

    # predict
    predictions = trainer.predict(model, test_data)

    # make directories if not there yet
    if os.path.isdir(cfg.save_path_pred) == False:
        os.mkdir(cfg.save_path_pred)
    if os.path.isdir(cfg.save_path_hmp) == False:
        os.mkdir(cfg.save_path_hmp)

    # bring to cpu convert to numpy and save
    for batch, pred in enumerate(predictions):
        pred = pred.cpu().numpy()
        np.save(cfg.save_path_pred + 'batch_{}'.format(batch), pred)

    for batch, (x, y, batch_len) in enumerate(test_data):
        y = torch.squeeze(y).numpy()
        np.save(cfg.save_path_hmp + 'batch_{}'.format(batch), y)

    print('end')


if __name__ == "__main__":
    predict()
