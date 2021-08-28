# Script to train and predict in the pretraing
from pytorch_lightning import callbacks
from pytorch_lightning import loggers
from pytorch_lightning.accelerators import accelerator
from torch._C import device
import preproc
from torch.utils.data import DataLoader, Subset
import torch
from net.modules import PretrainEncoderProjectionHead
from net.loss_fn import ContrastiveLoss
from preproc.load_data import Dataset_vid
import numpy as np
from torchvision import transforms
import pytorch_lightning as pl
from argparse import Namespace
from preproc.distortion import BloodBlobRandom
import config as cfg
from pytorch_lightning.loggers import WandbLogger
import wandb
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# Pytorch lightning Module Class (wrapper for easier training/testing) ----------------------------------------------------------------
# Used for Pretraining


class EncoderPretrain(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        # Define model
        self.model = PretrainEncoderProjectionHead()

        # Define loss function
        self.loss_fun = ContrastiveLoss(temp_fac=0.1)
        self.save_hyperparameters(hparams)

        # Hyperparameters fro training
        self.params = hparams

    # Forward in --> out

    def forward(self, x):
        return self.model(x[0])

    # Step in training/validation with loss calculation and logging, returns loss
    def step(self, batch, tag='train'):
        x, bl = batch

        # Get output (projection)
        embedding, projection, skip = self.forward(x)

        # Get batch length
        batch_len = bl[0]

        # Number of partitions with defined subset size
        partitions = batch_len//self.params['batch_size']

        # Number of positives for loss average
        no_pp = self.params['batch_size']/2*partitions

        # Initialze loss
        loss = 0

        # Get positive pairs and negative rest from each subset and add up losses
        for i in range(0, partitions):
            for j in range(0, self.params['batch_size'], 2):

                # Indices for positive pair
                p1 = i*self.params['batch_size']+j
                p2 = i*self.params['batch_size']+j+1

                # Indices for negative rest
                neg = np.arange(0, batch_len, dtype=np.int64)
                del_idx = np.arange(i*self.params['batch_size'],
                                    (i+1)*self.params['batch_size'])
                neg = np.delete(neg, del_idx)

                # Get the image with defined index
                p1_emb = torch.index_select(
                    projection, 0, torch.tensor(p1, dtype=torch.int64).to(self.device))
                p2_emb = torch.index_select(
                    projection, 0, torch.tensor(p2, dtype=torch.int64).to(self.device))
                neg_emb = torch.index_select(
                    projection, 0, torch.tensor(neg).to(self.device))

                # Calculate loss (recursive to add up)
                loss = self.loss_fun(loss, p1_emb, p2_emb, neg_emb)

        # Divide the loss by the number of positive samples
        loss = loss/no_pp

        # Logging
        loss_key = '{}_loss'.format(tag)
        self.log(loss_key, loss, on_epoch=True, on_step=True)

        return loss

    # Training step (calls step with tag 'train')
    def training_step(self, batch, batch_idx):
        return self.step(batch, 'train')

    # Validation step (calls step with tag 'val')
    def validation_step(self, batch, batch_idx):
        return self.step(batch, 'val')

    # Optimizer definition
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.params['lr'])


def main():
    # Set device to cuda if available
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.device_count())

    # Hyperparameters (learning rate, subset size)
    hparams = {'lr': cfg.lr_encoder_pretrain, 'batch_size': cfg.batch_size}

    # Login to wandb
    wandb.login(key='dd43afd6c6a913b95fb9279398da16945f9e3180')

    # Early stop callback
    early_stop_callback = EarlyStopping(
        monitor='val_loss_epoch', min_delta=0.00, patience=3, verbose=False, mode='min')

    # Weights and Biases Logger
    wandb_logger = WandbLogger(
        name='pretrain-Adam-{}-cycle={}'.format(cfg.lr_encoder_pretrain, cfg.cycle))

    # Data Transforms
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(cfg.image_resized_size),
        transforms.CenterCrop(cfg.image_cropped_size),
        BloodBlobRandom(20, 80, 0.01)
    ])

    # Video Dataset
    vid_dataset = Dataset_vid(cfg.data_list_path, transform=data_transform)

    # Train / Validation Split
    split = int(len(vid_dataset)*0.9)
    training_dataset = Subset(vid_dataset, range(0, split))
    val_dataset = Subset(vid_dataset, range(split, len(vid_dataset)))

    # Data Loaders
    dataloader_train = DataLoader(
        training_dataset, num_workers=16)
    dataloader_val = DataLoader(val_dataset, num_workers=16)

    # Load  model from last checkpoint if not first training cycle
    if cfg.cycle > 0:
        checkpoint_file = cfg.save_path_ckp_pretrain + cfg.pretrain_checkpoint_load
        model = EncoderPretrain.load_from_checkpoint(checkpoint_file)
    else:
        model = EncoderPretrain(hparams)

    # Training (limit to 4 epochs  and train in cycles to make sure training is not abborted by cluster)
    trainer = pl.Trainer(gpus=1, max_epochs=4,
                         callbacks=early_stop_callback, logger=wandb_logger, limit_train_batches=0.25, limit_val_batches=0.5)
    trainer.fit(model, dataloader_train, dataloader_val)

    # Save model/parameters/losses after training has finished
    checkpoint_file = cfg.save_path_ckp_pretrain + cfg.pretrain_checkpoint_save
    trainer.save_checkpoint(checkpoint_file)
    wandb.save(cfg.pretrain_checkpoint_save)


if __name__ == "__main__":
    main()
