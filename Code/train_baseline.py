# Script to train and predict in the baseline training
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import torch
from net.modules import Baseline
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
from net.loss_fn import ClassReg_Sections
import os

# Pytorch lightning Module Class (wrapper for easier training/testing) ----------------------------------------------------------------
# Used for Baseline


class BaselineModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.model = Baseline()  # define model (U-net)
        self.save_hyperparameters(hparams)
        self.params = hparams  # hyperparameters for model

    # Forward  in --> out
    def forward(self, x):
        return self.model(x[0])

    # Step in training/validation with loss calculation and logging, returns loss
    def step(self, batch, tag="train"):
        x, y, bl = batch
        y_hat = self.forward(x)
        batch_len = bl[0]

        loss_fun = torch.nn.MSELoss()
        loss = loss_fun(y_hat, torch.squeeze(y))

        loss_key = '{}_loss'.format(tag)

        self.log(loss_key, loss, on_epoch=True, on_step=True)

        return loss

    # Training step (calls step with tag 'train')

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    # Validation step (calls step with tag 'validation')
    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    # Prediction step (simple forward)
    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        x, y, bl = batch
        return self(x)

    # Optimizer definition
    def configure_optimizers(self):
        if cfg.opti_name == 'Adam':
            return torch.optim.Adam(self.parameters(), lr=self.params['lr'])
        elif cfg.opti_name == 'SGD':
            return torch.optim.SGD(self.parameters(), lr=self.params['lr'], momentum=self.params['momentum'])


def main():
    # Set device to cuda if available
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.device_count())

    # Login to wandb
    wandb.login(key='dd43afd6c6a913b95fb9279398da16945f9e3180')

    # Data Transforms
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(cfg.image_cropped_size),
        BloodBlobRandom(20, 80, 0.01)
    ])

    # Test and Train/Val labeled dataset
    train_dataset = Dataset_pyt(
        cfg.base_data_path, cfg.train_data_path, transform=data_transform)
    test_dataset = Dataset_pyt(
        cfg.base_data_path, cfg.test_data_path, transform=data_transform)

    # Train validation split
    split = int(len(train_dataset)*0.9)
    training_dataset = Subset(train_dataset, range(0, split))
    val_dataset = Subset(train_dataset, range(split, len(train_dataset)))

    # Data Loaders
    train_data = DataLoader(training_dataset, num_workers=4)
    val_data = DataLoader(val_dataset, num_workers=4)
    test_data = DataLoader(test_dataset, num_workers=4)

    # Weights and biases logger
    wandb_logger = WandbLogger(
        name='Baseline-{}-{}-{}'.format(cfg.opti_name, cfg.lr_full, len(train_dataset)))

    # Early stop callback
    early_stop_callback = EarlyStopping(
        monitor='val_loss_epoch', min_delta=0.00, patience=5, verbose=False, mode='min')

    # Hyperparams
    hparams = {'lr': cfg.lr_full, 'momentum': cfg.momentum,
               'batch_size': cfg.batch_size}

    # Training
    model = BaselineModel(hparams)
    trainer = pl.Trainer(
        gpus=1, callbacks=early_stop_callback, logger=wandb_logger)
    trainer.fit(model, train_data, val_data)

    # Save checkpoint at the end of training
    checkpoint_file = cfg.save_path_baseline + cfg.baseline_checkpoint
    print(checkpoint_file)
    trainer.save_checkpoint(checkpoint_file)
    wandb.save(checkpoint_file)

    # Prediction
    model = BaselineModel.load_from_checkpoint(checkpoint_file)
    trainer = pl.Trainer(gpus=1)
    predictions = trainer.predict(model, test_data)

    # Save predictions & heatmaps as numpy arrays
    if os.path.isdir(cfg.save_path_pred_baseline) == False:
        os.mkdir(cfg.save_path_pred_baseline)
    if os.path.isdir(cfg.save_path_hmp) == False:
        os.mkdir(cfg.save_path_hmp)

    for batch, pred in enumerate(predictions):
        pred = pred.cpu().numpy()
        np.save(cfg.save_path_pred_baseline + 'batch_{}'.format(batch), pred)

    for batch, (x, y, batch_len) in enumerate(test_data):
        y = torch.squeeze(y).numpy()
        np.save(cfg.save_path_hmp + 'batch_{}'.format(batch), y)


if __name__ == "__main__":
    main()
