# General libraries
import os
# Image processing
from PIL import Image

# PyTorch and Lightning related libraries
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
# Custom transformations
from custom_transforms import (ExtractFirstChannel, JointCompose, MaskToTensor,
                               RandomizedCrop, RandomizedFlip,
                               RandomizedResize)



class SARDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for semantic segmentation of SAR images.

    Handles data loading, splitting into train/validation/test sets, and applying
    the appropriate transformations for each dataset.
    """

    def __init__(self, data_dir: str = "./", batch_size: int = 8, val_split: float = 0.2):
        """
        Initializes the SARDataModule.

        Args:
            data_dir (str): Path to the dataset directory.
            batch_size (int): Number of samples per batch.
            val_split (float): Proportion of data to use for validation.
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_split = val_split

        # Transformations dictionary for train and validation/test sets
        self.transforms_dict = {
            "train": {
                "image": transforms.Compose([
                    transforms.ToTensor(),
                    ExtractFirstChannel()
                ]),
                "mask": transforms.Compose([
                    MaskToTensor(),
                ]),
                "joint": JointCompose([
                    RandomizedFlip()
                    
                ])
            },
            "validation": {
                "image": transforms.Compose([
                    transforms.ToTensor(),
                    ExtractFirstChannel(),  
                ]),
                "mask": transforms.Compose([
                    MaskToTensor(),
                ]),
                "joint": None
            },
            
            "test": {
                "image": transforms.Compose([
                    transforms.ToTensor(),
                    ExtractFirstChannel(),
                    transforms.Pad((0, 0, 30, 22), padding_mode='reflect')
                ]),
                "mask": transforms.Compose([
                    MaskToTensor(),
                    transforms.Pad((0, 0, 30, 22), padding_mode='reflect')
                ]),
                "joint": None
            }
        }

    def prepare_data(self) -> None:
        """
        Prepares the dataset if necessary. Placeholder in this case as no preparation is needed.
        """
        pass

    def setup(self, stage: str = None) -> None:
        """
        Sets up datasets for training, validation, and testing.

        Args:
            stage (str): The stage of training ("fit", "test", or None for both).
        """

        def get_sorted_file_names(folder_path):
            """
            Retrieves a sorted list of file names in a folder.

            Args:
                folder_path (str): Path to the folder.

            Returns:
                list: Sorted list of file names.
            """
            return sorted([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])

        # Paths for train, validation, and test data
        train_images_dir = os.path.join(self.data_dir, 'train_sliding/images_sliding')
        train_masks_dir = os.path.join(self.data_dir, 'train_sliding/labels_1D_sliding')
        test_images_dir = os.path.join(self.data_dir, 'test/images')
        test_masks_dir = os.path.join(self.data_dir, 'test/labels_1D')

        # Load train and validation data
        train_images_filenames = get_sorted_file_names(train_images_dir)
        train_images_paths = [os.path.join(train_images_dir, f) for f in train_images_filenames]
        train_masks_paths = [os.path.join(train_masks_dir, os.path.splitext(f)[0] + '.png') for f in train_images_filenames]

        # Split data into train and validation sets
        total_size = len(train_images_paths)
        val_size = int(total_size * self.val_split)
        train_size = total_size - val_size
        train_indices, val_indices = random_split(range(total_size), [train_size, val_size], generator=torch.Generator().manual_seed(42))

        
        
        # Load test data
        test_images_filenames = get_sorted_file_names(test_images_dir)
        test_images_paths = [os.path.join(test_images_dir, f) for f in test_images_filenames]
        test_masks_paths = [os.path.join(test_masks_dir, os.path.splitext(f)[0] + '.png') for f in test_images_filenames]
        


        # Initialize datasets for train, validation, and test stages
        if stage == "fit" or stage is None:
            self.train_dataset = SARImageDataset(
                [train_images_paths[i] for i in train_indices], [train_masks_paths[i] for i in train_indices],
                image_transform=self.transforms_dict["train"]["image"],
                mask_transform=self.transforms_dict["train"]["mask"],
                joint_transform=self.transforms_dict["train"]["joint"]
            )
            self.val_dataset = SARImageDataset(
                [train_images_paths[i] for i in val_indices], [train_masks_paths[i] for i in val_indices],
                image_transform=self.transforms_dict["validation"]["image"],
                mask_transform=self.transforms_dict["validation"]["mask"],
                joint_transform=self.transforms_dict["validation"]["joint"]
            )
        if stage == "test" or stage is None:
            self.test_dataset = SARImageDataset(
                test_images_paths, test_masks_paths,
                image_transform=self.transforms_dict["test"]["image"],
                mask_transform=self.transforms_dict["test"]["mask"],
                joint_transform=self.transforms_dict["test"]["joint"]
            )

    def train_dataloader(self):
        """
        Returns the DataLoader for the training dataset.

        """
        return DataLoader(self.train_dataset, batch_size=64, num_workers=20, shuffle=True, persistent_workers=True)
    
    def val_dataloader(self):
        """
        Returns the DataLoader for the validation dataset.

        """
        return DataLoader(self.val_dataset, num_workers=20, batch_size=self.batch_size, persistent_workers=True)

    def test_dataloader(self):
        """
        Returns the DataLoader for the test dataset.

        """
        return DataLoader(self.test_dataset, num_workers=20, batch_size=self.batch_size, persistent_workers=True)


class SARImageDataset(Dataset):
    """
    Custom PyTorch Dataset for SAR images and their corresponding masks.

    Applies individual and joint transformations to both the image and its mask.
    """

    def __init__(self, images_paths, masks_paths, image_transform=None, 
                 mask_transform=None, joint_transform=None):
        """
        Initializes the SARImageDataset.

        Args:
            images_paths (list): List of file paths for the images.
            masks_paths (list): List of file paths for the masks.
            image_transform (callable): Transformation applied to images.
            mask_transform (callable): Transformation applied to masks.
            joint_transform (callable): Transformation applied jointly to images and masks.
        """
        self.images_paths = images_paths
        self.masks_paths = masks_paths
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.joint_transform = joint_transform

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.images_paths)

    def __getitem__(self, idx):
        """
        Retrieves the image and mask at the specified index, applying transformations.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: Transformed image and mask as tensors.
        """
        # Load the image and corresponding mask
        image_path = self.images_paths[idx]
        mask_path = self.masks_paths[idx]
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        
        # Open images
        with Image.open(image_path) as img, Image.open(mask_path) as mask:
            # Apply joint transformations
            if self.joint_transform:
                img, mask = self.joint_transform(img, mask)

            # Apply individual transformations
            img = self.image_transform(img) if self.image_transform else transforms.ToTensor()(img)
            mask = self.mask_transform(mask) if self.mask_transform else MaskToTensor(mask)
            mask = mask.squeeze(0).long()

        return img, mask