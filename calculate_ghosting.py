import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation
import os

def main(img_path, seg_path):
    # need to calculate for mark's as well but will need to run on all 15 slices
    sc_mask_img = nib.load(seg_path)
    sc_mask = sc_mask_img.get_fdata() > 0.5
    img_affine = sc_mask_img.affine

    # dialat the mask
    structure = np.zeros((3, 3, 3))
    structure[1, :, :] = 1

    dilated_sc = binary_dilation(sc_mask, structure=structure, iterations=3)
    csf_ring_mask = dilated_sc.astype(int) - sc_mask.astype(int)
    # Add this temporary line to your script to view your mask
    nib.save(nib.Nifti1Image(csf_ring_mask.astype(np.uint8), img_affine), 'test_csf_ring.nii.gz')  

    # need to not hard code this
    echo_index = 2
    #load nifti images
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found at {img_path}")
    
    # load images and masks
    data = np.nan_to_num(np.abs(nib.load(img_path).get_fdata()))
    mean_ghosting = 0 
    for i in range(data.shape[2]):
        mri_slice_data = data[:, :, i]
        csf_slice_mask = csf_ring_mask[:, :, i]
        s_csf_m = mri_slice_data[csf_slice_mask == 1]
   
        sc_slice_mask = sc_mask[:, :, i]
        s_sc_m = mri_slice_data[sc_slice_mask == 1]

        mean_ghosting = mean_ghosting + (np.mean(s_csf_m) / np.mean(s_sc_m) if len(s_csf_m) > 0 and np.mean(s_sc_m) > 0 else 0)

    mean_ghosting = mean_ghosting / data.shape[2]  
    return mean_ghosting

if __name__ == "__main__":
    siemens_path = '/Users/rhiannonbutler/shimming-toolbox/output_dicom_to_nifti_whole/tmp_dcm2bids/sub-test1/-1906368996_Zurich_Data_gre_spine_000000_e2.nii.gz'
    siemens_seg_path = '/Users/rhiannonbutler/spinalcordtoolbox/-1906368996_Zurich_Data_gre_spine_000000_e2_seg.nii.gz'
    ghosting_metric = main(siemens_path, siemens_seg_path)
    print(f"Ghosting Metric: {ghosting_metric}")
