import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

import classifier.utils as utils
import classifier.preprocessing as preprocessing

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
        
        original_img, processed_img, _ = preprocessing.preprocess_pipeline(image_path, resize_shape, 'train')
        
        # Convert to NumPy arrays for visualization
        original_img_np = original_img.numpy() if isinstance(original_img, torch.Tensor) else original_img
        augmented_img_np = processed_img.numpy() if isinstance(processed_img, torch.Tensor) else processed_img
        
        # Extract filename without extension .dcm
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Plot and save images
        plot_and_save_images(original_img_np, augmented_img_np, output_folder, image_name)

    print(f"Processed and saved {num_images} images in '{output_folder}' for debugging.")

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
