import os

from PIL import Image, ImageOps


def reflect_pad_horizontal(img, pad_left, pad_right):
    """
    Reflectively pad the image on the left and right sides.

    Args:
        img (PIL.Image): The image to pad.
        pad_left (int): Number of pixels to reflect on the left side.
        pad_right (int): Number of pixels to reflect on the right side.

    Returns:
        PIL.Image: A new image with horizontal reflection padding.
    """
    w, h = img.size

    # Create a new blank image with extra space for left/right pads
    mode = img.mode
    new_width = w + pad_left + pad_right
    padded_img = Image.new(mode, (new_width, h))

    # 1) Reflect the left side
    left_section = img.crop((0, 0, pad_left, h)) 
    left_section_reflected = ImageOps.mirror(left_section)
    padded_img.paste(left_section_reflected, (0, 0))

    # 2) Paste original image
    padded_img.paste(img, (pad_left, 0))

    # 3) Reflect the right side
    right_section = img.crop((w - pad_right, 0, w, h))
    right_section_reflected = ImageOps.mirror(right_section)
    padded_img.paste(right_section_reflected, (w + pad_left, 0))

    return padded_img


def create_sliding_window_dataset(
    image_dir,
    mask_dir,
    output_image_dir,
    output_mask_dir,
    crop_top_bottom=5,
    pad_left_right=15,
    patch_width=320,
    patch_height=320
):
    """
    Generate 320x320 patches for each image and mask using a sliding-window approach,
    with reflection padding on the left and right sides.

    Args:
        image_dir (str): Path to the folder containing original images.
        mask_dir (str): Path to the folder containing corresponding segmentation masks.
        output_image_dir (str): Where to store the patched images.
        output_mask_dir (str): Where to store the patched masks.
        crop_top_bottom (int): Number of pixels to remove from the top and bottom.
        pad_left_right (int): Number of pixels to reflectively pad on the left and right.
        patch_width (int): Width of each patch to extract.
        patch_height (int): Height of each patch to extract.
    """
    
    # Ensure output directories exist
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)
    
    file_list = sorted(os.listdir(image_dir))
    
    for filename in file_list:
        # Adjust if you use different image extensions
        if not (filename.lower().endswith('.png') or 
                filename.lower().endswith('.jpg') or 
                filename.lower().endswith('.jpeg')):
            continue
        
        img_path = os.path.join(image_dir, filename)

        msk_path = os.path.join(mask_dir, filename).replace('.jpg', '.png')  # same filename in mask_dir, different file extension
        
        # Open image and mask
        with Image.open(img_path) as img, Image.open(msk_path) as msk:
            
            # Crop top/bottom 
            width, height = img.size
            new_height = height - 2 * crop_top_bottom
            cropped_img = img.crop((0, crop_top_bottom, width, height - crop_top_bottom))
            cropped_msk = msk.crop((0, crop_top_bottom, width, height - crop_top_bottom))
            
            # Reflective padding on the left and right
            padded_img = reflect_pad_horizontal(cropped_img, pad_left_right, pad_left_right)
            padded_msk = reflect_pad_horizontal(cropped_msk, pad_left_right, pad_left_right)
            
            padded_width, padded_height = padded_img.size
            
            # Slide through patches
            patch_index = 0
            for top in range(0, padded_height, patch_height):
                if top + patch_height > padded_height:
                    break  # avoid partial patch at the bottom
                
                for left in range(0, padded_width, patch_width):
                    if left + patch_width > padded_width:
                        break  # avoid partial patch on the right
                    
                    # Extract patch
                    box = (left, top, left + patch_width, top + patch_height)
                    patch_img = padded_img.crop(box)
                    patch_msk = padded_msk.crop(box)
                    
                    # Generate output filenames
                    base_name, _ = os.path.splitext(filename)
                    patch_img_name = f"{base_name}_patch{patch_index}{'.jpg'}"
                    patch_msk_name = f"{base_name}_patch{patch_index}{'.png'}"
                    
                    # Save to output folder
                    patch_img.save(os.path.join(output_image_dir, patch_img_name))
                    patch_msk.save(os.path.join(output_mask_dir, patch_msk_name))
                    
                    patch_index += 1


if __name__ == "__main__":
    # Example usage:
    image_dir = r"\dataset\train\images"
    mask_dir = r"\dataset\train\labels_1D"
    output_image_dir = r"\dataset\train_sliding\images_sliding"
    output_mask_dir = r"\dataset\train_sliding\labels_1D_sliding"
    
    create_sliding_window_dataset(
        image_dir,
        mask_dir,
        output_image_dir,
        output_mask_dir,
        crop_top_bottom=5,
        pad_left_right=15,
        patch_width=320,
        patch_height=320
    )
