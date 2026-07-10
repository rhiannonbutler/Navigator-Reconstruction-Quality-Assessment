import numpy as np
import nibabel as nib

'''
Goal is to calculate whole cord SNR. To measure noise, we will look at the difference between the two echoes. 
The difference image will be used to estimate noise, and the mean of the first echo will be used to estimate signal. 
The SNR will be calculated as the mean of the first echo divided by the standard deviation of the difference image.
'''

def main(echo1_path, echo2_path, img_seg_path):
    # Load the two echo images
    echo1_img = nib.load(echo1_path)
    echo2_img = nib.load(echo2_path)
    img_seg = nib.load(img_seg_path)
    
    # Get the data from the images
    echo1_data = np.nan_to_num(np.abs(echo1_img.get_fdata()))
    echo2_data = np.nan_to_num(np.abs(echo2_img.get_fdata()))
    img_seg_data = img_seg.get_fdata()


    nslices = echo1_data.shape[2] 
    slice_wise_std = np.zeros(nslices)
    slice_wise_mean = np.zeros(nslices)
    slice_wise_snr = np.zeros(nslices)

    diff_image = echo1_data - echo2_data
    
    for z in range(nslices):
        sc_slice_mask = img_seg_data[:, :, z] > 0.5
        sc_masked = echo1_data[sc_slice_mask == 1]
        diff_mask_seg = diff_image[sc_slice_mask == 1]
        slice_wise_std[z] = np.ma.std(diff_mask_seg) / np.sqrt(2) 
        slice_wise_mean[z] = np.ma.mean(sc_masked)
        slice_wise_snr[z] = slice_wise_mean[z] / slice_wise_std[z]
    
    # Compute max and mean STDs
    mean_snr = np.nanmean(slice_wise_snr)
    median_snr = np.nanmedian(slice_wise_snr)

    return mean_snr, median_snr

if __name__ == "__main__":
    echo1_path = '/Users/rhiannonbutler/shimming-toolbox/output_dicom_to_nifti_whole/tmp_dcm2bids/sub-test1/-1906368996_Zurich_Data_gre_spine_000000_e2.nii.gz'
    echo2_path = '/Users/rhiannonbutler/shimming-toolbox/output_dicom_to_nifti_whole/tmp_dcm2bids/sub-test1/-1906368996_Zurich_Data_gre_spine_000000_e3.nii.gz'
    img_seg_path = '/Users/rhiannonbutler/spinalcordtoolbox/-1906368996_Zurich_Data_gre_spine_000000_e2_seg.nii.gz'
    mean_snr, median_snr = main(echo1_path, echo2_path, img_seg_path)
    print(f"Mean SNR: {mean_snr}, Median SNR: {median_snr}")