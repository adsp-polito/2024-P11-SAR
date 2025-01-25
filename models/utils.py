import os
from PIL import Image, ImageOps
import numpy as np

# Define the RGB mapping for each label
LABEL_TO_RGB = {
    0: (0, 0, 0),          # Black - Sea Surface
    1: (0, 255, 255),      # Cyan  - Oil Spill
    2: (255, 0, 0),        # Red   - Look-alike
    3: (153, 76, 0),       # Brown - Ship
    4: (0, 153, 0)         # Green - Land
}

def label_to_rgb(mask):
    # Create an empty RGB image
    height, width = mask.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Map each label to its corresponding RGB value
    for label, color in LABEL_TO_RGB.items():
        rgb_image[mask == label] = color

    return rgb_image

def save_prediction_images(masks, preds, counter, result_folder="result"):
    """
    Save masks and predictions side-by-side with borders to distinguish.

    Parameters:
        masks (torch.Tensor): Tensor of ground truth masks (batch_size, height, width).
        preds (torch.Tensor): Tensor of predicted masks (batch_size, height, width).
        label_to_rgb (function): Function to convert a single-channel image to RGB.
        result_folder (str): Folder to save the resulting images.
        border_color (tuple): RGB color for the border.
        border_width (int): Width of the border in pixels.
    """
    results_dir = r'training/models/results'
    # Ensure the result folder exists
    os.makedirs(results_dir, exist_ok=True)

    for i in range(masks.size(0)):
        # Extract individual mask and prediction
        mask = masks[i, :, :].cpu().numpy()  # Shape: (height, width)
        pred = preds[i, :, :].cpu().numpy()  # Shape: (height, width)

        # Convert to RGB using the provided function
        pred_rgb = label_to_rgb(pred)  # Result: (H, W, 3)
        mask_rgb = label_to_rgb(mask)  # Result: (H, W, 3)

        # Create PIL images
        mask_pil = Image.fromarray(mask_rgb)
        pred_pil = Image.fromarray(pred_rgb)

        # Add white border to images
        mask_pil = ImageOps.expand(mask_pil, border=5, fill=(255, 255, 255))
        pred_pil = ImageOps.expand(pred_pil, border=5, fill=(255, 255, 255))

        # Concatenate the image and prediction
        combined = Image.new("RGB", (mask_pil.width + pred_pil.width, mask_pil.height))
        combined.paste(mask_pil, (0, 0))
        combined.paste(pred_pil, (mask_pil.width, 0))

        # Save to file
        combined.save(f"{results_dir}/{result_folder}/output_{counter}.png")
        counter += 1
    
    return counter
