import os
import subprocess
from generate_sct_masks import generate_sct_masks
import nibabel as nib
import numpy as np
from calculate_snr import main as calculate_snr
from calculate_ghosting import main as calculate_ghosting

'''
The goal is to evaluate the reconstruction done using Mark's method against Siemen's reconstruction.
Script uses the quality metrics used in the paper:
" Optimized navigator-based correction of breathing-induced B0 field fluctuations in multi-echo gradient-echo imaging of the spinal cord
Laura Beghini, Silvan Büeler, Martina D. Liechti, Alexander Jaffray, Gergely David, S. Johanna Vannesjo
medRxiv 2024.12.05.24318389; doi: https://doi.org/10.1101/2024.12.05.24318389"
'''

def main(img1_echo1_path, img1_echo2_path, seg1_path):
    #load nifti images
    if not os.path.exists(img1_echo1_path):
        raise FileNotFoundError(f"Image 1 echo 1 not found at {img1_echo1_path}")
    if not os.path.exists(img1_echo2_path):
        raise FileNotFoundError(f"Image 1 echo 2 not found at {img1_echo2_path}")
    
    img1_ghosting = calculate_ghosting(img1_echo1_path, seg1_path)

    img1_snr = calculate_snr(img1_echo1_path, img1_echo2_path)

    return {
        "img1_ghosting": img1_ghosting,
        "img1_snr": img1_snr,
    }

if __name__ == "__main__":
    img1_echo1_path = '/Users/rhiannonbutler/shimming-toolbox/output_dicom_to_nifti_whole/tmp_dcm2bids/sub-test1/-1906368996_Zurich_Data_gre_spine_000000_e2.nii.gz'
    img1_echo2_path = '/Users/rhiannonbutler/shimming-toolbox/output_dicom_to_nifti_whole/tmp_dcm2bids/sub-test1/-1906368996_Zurich_Data_gre_spine_000000_e3.nii.gz'
    seg1_path = '/Users/rhiannonbutler/spinalcordtoolbox/-1906368996_Zurich_Data_gre_spine_000000_e2_seg.nii.gz'
    results = main(img1_echo1_path, img1_echo2_path, seg1_path)
    print(f"Results: {results}")