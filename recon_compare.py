import os
import nibabel as nib
import numpy as np
from skimage.metrics import structural_similarity as ssim

'''
Script to compare two reconstructed images using SSIM (Structural Similarity Index Measure).
The goal is to evaluate the reconstruction done using Mark's method against Siemen's reconstruction.
'''

def main(siemens_path, marks_path):
    #load nifti images
    if not os.path.exists(siemens_path):
        raise FileNotFoundError(f"Siemens image not found at {siemens_path}")
    if not os.path.exists(marks_path):
        raise FileNotFoundError(f"Mark's image not found at {marks_path}")  
    
    mark_img = nib.load(marks_path)
    siemens_img = nib.load(siemens_path)    

    mark_data = mark_img.get_fdata()
    siemens_data = siemens_img.get_fdata()

    if np.iscomplexobj(mark_data):
        mark_data = np.abs(mark_data)
    if np.iscomplexobj(siemens_data):
        siemens_data = np.abs(siemens_data)
    
    # Ensure the images have the same shape
    if mark_data.shape != siemens_data.shape:
        raise ValueError(f"Image shapes do not match: Mark's image shape {mark_data.shape}, Siemens image shape {siemens_data.shape}")  
    
    # Normaliz intensity to 0-1
    mark = np.nan_to_num(mark_data)
    siemens = np.nan_to_num(siemens_data)

    mark_min = np.min(mark)
    mark_max = np.max(mark)
    if mark_max - mark_min != 0:
        mark = (mark - mark_min) / (mark_max - mark_min)

    siemens_min = np.min(siemens)
    siemens_max = np.max(siemens)
    if siemens_max - siemens_min != 0:
        siemens = (siemens - siemens_min) / (siemens_max - siemens_min)

    # Compute RNSE 
    rmse = np.sqrt(np.mean((mark - siemens) ** 2))

    # Compute PSNR (max value is 1)
    mse = np.mean((mark - siemens) ** 2)
    if mse == 0:
        psnr = float('inf')  # Infinite PSNR if images are identical
    else:
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    
    # Compute SSIM
    ssim_value = ssim(mark, siemens, data_range=1.0, channel_axis=-1)

    # Print results
    print(f"RMSE: {rmse:.4f}")
    print(f"PSNR: {psnr:.4f} dB")
    print(f"SSIM: {ssim_value:.4f}")

    return rmse, psnr, ssim_value

if __name__ == "__main__":
    siemens_path = "path_to_siemens_image.nii"  # Replace with actual path
    marks_path = "path_to_marks_image.nii"      # Replace with actual path
    main(siemens_path, marks_path)
    
