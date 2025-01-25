import random
import numpy as np
from torchvision.transforms import functional as F
import torch

class ProbabilisticRandomizedFocusedCrop:
    def __init__(self, crop_transform, crop_size=320, probability=0.5):
        """
        Args:
            crop_transform (callable): The focused crop transformation.
            crop_size (int): The size of the random crop.
            probability (float): The probability of applying the focused crop.
        """
        self.crop_transform = crop_transform
        self.crop_size = crop_size
        self.probability = probability

    def __call__(self, image, mask):
        """
        Applies the focused crop transformation with a given probability if an oil spill is present.
        Otherwise, performs a random crop of the specified size.
        """
        # Convert mask to numpy to check for oil spill
        mask_np = np.array(mask)
        oil_spill_present = np.any(mask_np == 1)

        # Apply focused crop with the given probability if oil spill is present
        if oil_spill_present and random.random() < self.probability:
            cropped_image, cropped_mask = self.crop_transform(image, mask)
            return cropped_image, cropped_mask 

        # Otherwise, perform a random crop
        width, height = image.size
        top = random.randint(0, max(0, height - self.crop_size))
        left = random.randint(0, max(0, width - self.crop_size))

        # Perform random crop on both image and mask
        cropped_image = F.crop(image, top, left, self.crop_size, self.crop_size)
        cropped_mask = F.crop(mask, top, left, self.crop_size, self.crop_size)

        return cropped_image, cropped_mask
    
    
    


class RandomizedFocusedCrop:
    def __init__(self, crop_size=320, max_shift=20):
        """
        Args:
            crop_size (int): Size of the crop.
            max_shift (int): Maximum number of pixels to randomly shift the crop center.
        """
        self.crop_size = crop_size
        self.max_shift = max_shift

    def __call__(self, image, mask):
        # Convert mask to numpy
        mask_np = np.array(mask)
        # Focus only on oil spill label (1)
        oil_spill_mask = (mask_np == 1).astype(np.uint8)

        # Find oil spill region in mask
        oil_spill_indices = np.argwhere(oil_spill_mask > 0)

        if len(oil_spill_indices) > 0:
            # Calculate bounding box around oil spill
            y_min, x_min = oil_spill_indices.min(axis=0)
            y_max, x_max = oil_spill_indices.max(axis=0)

            # Calculate initial crop center
            center_y = (y_min + y_max) // 2
            center_x = (x_min + x_max) // 2

            # Apply random shift to the crop center
            center_y += random.randint(-self.max_shift, self.max_shift)
            center_x += random.randint(-self.max_shift, self.max_shift)

            # Determine crop box
            half_crop = self.crop_size // 2
            top = max(center_y - half_crop, 0)
            left = max(center_x - half_crop, 0)
            bottom = top + self.crop_size
            right = left + self.crop_size

            # Adjust if crop exceeds image boundaries
            if bottom > image.height:
                bottom = image.height
                top = bottom - self.crop_size
            if right > image.width:
                right = image.width
                left = right - self.crop_size

            # Crop the image and mask
            image = F.crop(image, top, left, self.crop_size, self.crop_size)
            mask = F.crop(mask, top, left, self.crop_size, self.crop_size)

        return image, mask
    
    
class MaskToTensor:
    def __call__(self, mask):
        # Converts mask to a PyTorch tensor with dtype=torch.int64
        return torch.as_tensor(np.array(mask), dtype=torch.int64)


class RandomizedFlip:
    def __init__(self, flip_probability=0.5):
        """
        Args:
            flip_probability (float): Probability of applying a flip (default: 0.5).
        """
        self.flip_probability = flip_probability
        self.flip_type = "none"  # Initialize flip type

    def __call__(self, image, mask):
        """
        Randomly applies a horizontal flip, vertical flip, or both to the image and mask.

        Args:
            image (PIL Image): The input image.
            mask (PIL Image): The corresponding mask.

        Returns:
            (PIL Image, PIL Image): The flipped image and mask.
        """
        if random.random() < self.flip_probability:
            # Select the flip type and store it
            self.flip_type = random.choices(
                ['horizontal', 'vertical', 'both'], weights=[1/3, 1/3, 1/3], k=1
            )[0]
            
            if self.flip_type == 'horizontal':
                image = F.hflip(image)
                mask = F.hflip(mask)
            elif self.flip_type == 'vertical':
                image = F.vflip(image)
                mask = F.vflip(mask)
            elif self.flip_type == 'both':
                image = F.hflip(F.vflip(image))
                mask = F.hflip(F.vflip(mask))

        else:
            self.flip_type = "none"  # Set flip type to none if no flip is applied
            
        return image, mask


class RandomizedResize:
    def __init__(self, scale_range=(0.5, 1.5), resize_probability=0.5):
        """
        Args:
            scale_range (tuple): A tuple specifying the min and max scale factors (default: (0.5, 1.5)).
        """
        self.scale_range = scale_range
        self.resize_probability = resize_probability

    def __call__(self, image, mask):
        """
        Randomly resizes the image and mask within the specified scale range.

        Args:
            image (PIL Image): The input image.
            mask (PIL Image): The corresponding mask.

        Returns:
            (PIL Image, PIL Image): The resized image and mask.
        """
        if random.random() < self.resize_probability:
            # Generate a random scale factor within the scale range
            scale_factor = random.uniform(*self.scale_range)

            # Compute new dimensions
            new_width = int(image.width * scale_factor)
            new_height = int(image.height * scale_factor)

            # Resize both the image and the mask
            image = F.resize(image, (new_height, new_width), interpolation=F.InterpolationMode.BILINEAR)
            mask = F.resize(mask, (new_height, new_width), interpolation=F.InterpolationMode.NEAREST)
            return image, mask
        
        return image, mask

class RandomizedCrop:
    def __init__(self, crop_size=320):
        """
        Args:
            crop_size (int): The size of the random crop.
        """
        self.crop_size = crop_size

    def __call__(self, image, mask):
        """
        Applies a random crop of the specified size.
        """
        width, height = image.size
        top = random.randint(0, max(0, height - self.crop_size))
        left = random.randint(0, max(0, width - self.crop_size))

        # Perform random crop on both image and mask
        cropped_image = F.crop(image, top, left, self.crop_size, self.crop_size)
        cropped_mask = F.crop(mask, top, left, self.crop_size, self.crop_size)

        return cropped_image, cropped_mask

class ExtractFirstChannel:
    def __call__(self, tensor):
        """
        Extracts the first channel from the input tensor.

        Args:
            tensor (torch.Tensor): The input tensor with shape (C, H, W), where C is the number of channels.

        Returns:
            torch.Tensor: A tensor with only the first channel and shape (1, H, W).
        """
        return tensor[0:1, :, :]


class JointCompose:
    def __init__(self, transforms_list):
        """
        Initializes the JointCompose transformation.

        Args:
            transforms_list (list): A list of transformations to be applied jointly to the image and mask.
        """
        self.transforms = transforms_list

    def __call__(self, image, mask):
        """
        Applies the sequence of transformations to the input image and mask.

        Args:
            image (PIL Image): The input image.
            mask (PIL Image): The corresponding mask.

        Returns:
            (PIL Image, PIL Image): The transformed image and mask.
        """
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask
