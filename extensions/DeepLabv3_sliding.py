# General libraries
import os
import time
import zipfile

# PyTorch Lightning and related
import pytorch_lightning
# Segmentation Models PyTorch
import segmentation_models_pytorch as smp
# Pytorch and related 
import torch
import torch.nn as nn
import torchmetrics
import wandb
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
# Weights and Biases support
from pytorch_lightning.loggers import WandbLogger
from torch.optim import Adam
# Data handling
from training.extensions.sliding_data_module import SARDataModule
from utils import save_prediction_images

# Speed up computation
torch.set_float32_matmul_precision('medium')

# Initialize W&B logger
wandb_logger = WandbLogger(
    project='SAR_ADSP_2425', 
    name='deeplab_Sliding_long'
)

CLASS_WEIGHTS_FOCUSED = [1, 2, 1, 4, 1]

# ============================= TRAINING MODULE =============================

class SARSegmentationModel(LightningModule):
    """
    A PyTorch Lightning module for SAR image semantic segmentation.

    This model uses a DeeplabV3 architecture from the Segmentation Models PyTorch (smp) library,
    with a MobileNet encoder and class weights for handling imbalanced data.
    """

    def __init__(self, learning_rate=5e-5, num_classes=5):
        """
        Initializes the SAR segmentation model.

        Args:
            learning_rate (float): Learning rate for the optimizer.
            num_classes (int): Number of classes for segmentation.
        """
        super().__init__()
        self.save_hyperparameters()
        self.result_counter = 1

        # Define the DeeplabV3+ model with MobilenetV3 as the encoder
        self.model = smp.DeepLabV3Plus(
            encoder_name="timm-mobilenetv3_large_100",       # Encoder type
            encoder_weights=None,          # No pre-trained weights
            in_channels=1,                 # Input channels (1 for SAR images)
            classes=num_classes             # Output classes
        )

        # Define loss function with class weights
        class_weights_tensor = torch.tensor(CLASS_WEIGHTS_FOCUSED, dtype=torch.float32)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

        # Define learning rate
        self.learning_rate = learning_rate

        # Metrics for IoU (Intersection over Union)
        self.train_iou = torchmetrics.JaccardIndex(num_classes=num_classes, task="multiclass", average='none')
        self.val_iou = torchmetrics.JaccardIndex(num_classes=num_classes, task="multiclass", average='none')

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Model output.
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Training step for a single batch.

        Args:
            batch (tuple): A tuple containing images and masks.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Training loss for the batch.
        """
        images, masks = batch
        outputs = self(images)
        loss = self.criterion(outputs, masks.long())
        preds = outputs.argmax(dim=1)
        iou = self.train_iou(preds, masks)

        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        for class_idx, class_iou in enumerate(iou):
            self.log(f"train_iou_class_{class_idx}", class_iou, on_epoch=True, prog_bar=False)
        mean_iou = iou.mean()
        self.log("train_mean_iou", mean_iou, on_epoch=True, prog_bar=True)
            
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for a single batch.

        Args:
            batch (tuple): A tuple containing images and masks.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Validation loss for the batch.
        """
        images, masks = batch
        outputs = self(images)
        loss = self.criterion(outputs, masks.long())
        preds = outputs.argmax(dim=1)

        # Remove padding from predictions and masks
        preds = preds[:, :650, :1250]
        masks = masks[:, :650, :1250]

        # Compute IoU and log metrics
        iou = self.val_iou(preds, masks)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        for class_idx, class_iou in enumerate(iou):
            self.log(f"val_iou_class_{class_idx}", class_iou, on_epoch=True, prog_bar=False)
        mean_iou = iou.mean()
        self.log("val_mean_iou", mean_iou, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        """
        Test step for a single batch.

        Args:
            batch (tuple): A tuple containing images and masks.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Test loss for the batch.
        """
        images, masks = batch
        outputs = self(images)
        loss = self.criterion(outputs, masks.long())
        preds = outputs.argmax(dim=1)
        
        # Remove padding from predictions and masks
        preds = preds[:, :650, :1250]
        masks = masks[:, :650, :1250]
        
        iou = self.val_iou(preds, masks)

        # Log test metrics
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        for class_idx, class_iou in enumerate(iou):
            self.log(f"test_iou_class_{class_idx}", class_iou, on_epoch=True, prog_bar=False)
        mean_iou = iou.mean()
        self.log("test_mean_iou", mean_iou, on_epoch=True, prog_bar=True)
        
        self.result_counter = save_prediction_images(masks, preds, counter=self.result_counter, result_folder='Deeplab_Sliding')

        return loss

    def configure_optimizers(self):
        """
        Configures the optimizer.

        """
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

# ============================= TRAINING LOOP =============================

def main():
    """
    Main training loop for the SAR segmentation model.
    """
    # Define number of training epochs and batch size
    num_epochs = 200
    batch = 36
    
    # Initialize data module and model
    data_module = SARDataModule(data_dir="dataset/", batch_size=batch, val_split=0.15)
    model = SARSegmentationModel(learning_rate=5e-3, num_classes=5)
    
    # Define a callback to save the best checkpoint
    checkpoint_callback = ModelCheckpoint(
        monitor="val_iou_class_1", mode="max", save_top_k=1, filename="best-checkpoint"
    )

    # Trainer configuration
    trainer = Trainer(
        logger=wandb_logger,
        max_epochs=num_epochs,
        devices=1,
        accelerator="gpu",
        callbacks=[checkpoint_callback]
    )
    
    # Train the model
    start_time = time.time()
    trainer.fit(model, datamodule=data_module)
    end_train = time.time()
    print(f'total training time for {num_epochs} epochs, using batch size of {batch}: \n {end_train - start_time} seconds')
    # Save the best model as a compressed zip file
    best_model_path = checkpoint_callback.best_model_path
    if os.path.exists(best_model_path):
        zip_path = best_model_path + ".zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(best_model_path, arcname=os.path.basename(best_model_path))
        os.remove(best_model_path)
        print('Final model saved and compressed!')

    # ============================= TEST =============================

    # Load the best model from the compressed file
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        zipf.extractall(os.path.dirname(zip_path))
        unzipped_path = zip_path.replace('.zip', '')

    best_model = SARSegmentationModel.load_from_checkpoint(unzipped_path)
    
    # Test the model
    trainer.test(best_model, datamodule=data_module)


if __name__ == '__main__':
    main()