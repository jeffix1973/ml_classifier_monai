import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import torch
from torchvision.transforms import Compose, Resize
from monai.transforms import (
    ScaleIntensity,
    RandFlip, RandRotate, RandZoom, RandAdjustContrast, RandGaussianNoise, NormalizeIntensity
)

import classifier.utils as utils


def debug_dataset_images(data_root_path, csv_file, resize_shape, num_images):
    df = pd.read_csv(csv_file)
    sample_df = df.sample(n=num_images)
    
    output_folder = 'debug'
    if os.path.exists(output_folder):
        # empty the folder using os
        for file in os.listdir(output_folder):
            os.remove(os.path.join(output_folder, file))
    else:
        utils.create_dir(output_folder)

    for index, row in sample_df.iterrows():
         
        if data_root_path is not None:
            join_path = "/".join(getattr(row, 'OLEA_INSTANCE_PATH').split("/")[-3:])
            image_path = data_root_path + "/" + join_path
            if not image_path.endswith('.dcm'):
                image_path += '.dcm'
        else:
            image_path = row['OLEA_INSTANCE_PATH']
        
        original_img, processed_img = preprocess_pipeline(image_path, resize_shape, 'train')
        
        # Convert to NumPy arrays for visualization
        original_img_np = original_img.numpy() if isinstance(original_img, torch.Tensor) else original_img
        augmented_img_np = processed_img.numpy() if isinstance(processed_img, torch.Tensor) else processed_img
        
        # Extract filename without extension .dcm
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Plot and save images
        plot_and_save_images(original_img_np, augmented_img_np, output_folder, image_name)

    print(f"Processed and saved {num_images} images in '{output_folder}' for debugging.")


def construct_image_path(root_path, relative_path):
    join_path = "/".join(relative_path.split("/")[-3:])
    full_path = os.path.join(root_path, join_path)
    return full_path + '.dcm' if not full_path.endswith('.dcm') else full_path


def preprocess_pipeline(dcm_path, resize_shape, mode):
    
    # Load the image using load_dicom
    img = load_dicom(dcm_path)
    # add channel dimension
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
    # Convert to PyTorch tensor
    img_tensor = torch.from_numpy(img).permute(2, 0, 1)  # Convert to [C, H, W] format

    # Make a copy of the original image tensor for augmentation
    augmented_img = img_tensor.clone()

    # Scale intensity
    scale_intensity = ScaleIntensity()
    img_tensor = scale_intensity(img_tensor)

    # Apply MONAI augmentations only on train images
    if mode == 'train':
        augmented_img = apply_augmentation(augmented_img)

    # Resize images
    resize_transform = Resize(resize_shape, antialias=True)
    img_tensor = resize_transform(img_tensor)
    augmented_img = resize_transform(augmented_img)
    # Normalize intensity
    normalize_intensity = NormalizeIntensity()
    augmented_img = normalize_intensity(augmented_img)
    
    return img_tensor, augmented_img


def load_dicom(dcm_path):
    # Load DICOM image
    dcm_data = pydicom.dcmread(dcm_path)
    img = dcm_data.pixel_array.astype(np.float32)
    return img


def apply_augmentation(image_tensor):

    # Define augmentations
    augmentations = Compose([
        RandFlip(prob=0.3, spatial_axis=0),  # Flip along height axes
        RandFlip(prob=0.3, spatial_axis=1),  # Flip along width axes
        RandRotate(range_x=np.pi/12, prob=0.3),
        RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.3),
        RandAdjustContrast(gamma=(0.9, 1.1), prob=0.3),
        RandGaussianNoise(prob=0.3),
    ])

    augmented_img = augmentations(image_tensor)
    
    return augmented_img


def plot_and_save_images(original, processed, output_folder, image_name):
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    
    # Check and squeeze the channel dimension if necessary
    if original.ndim == 3 and original.shape[0] == 1:
        original_squeezed = np.squeeze(original)
    else:
        original_squeezed = original

    if processed.ndim == 3 and processed.shape[0] == 1:
        processed_squeezed = np.squeeze(processed)
    else:
        processed_squeezed = processed
    
    axes[0].imshow(original_squeezed, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(processed_squeezed, cmap='gray')
    axes[1].set_title('Processed Image')
    axes[1].axis('off')

    output_path = os.path.join(output_folder, f"{image_name}.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


