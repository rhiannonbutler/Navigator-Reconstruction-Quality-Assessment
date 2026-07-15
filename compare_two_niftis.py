from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
try:
    import nibabel as nib
except ImportError as e:
    raise ImportError(
        "nibabel is required to run this script. "
        "Install it with: pip install nibabel"
    ) from e


def load_nifti(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    if nib is None:
        raise RuntimeError("nibabel is required to read NIfTI files. Install it with: pip install nibabel")
    img = nib.load(str(path))
    return img.get_fdata(dtype=np.float32), img.affine


def select_slice(data: np.ndarray, slice_index: int | None = None) -> np.ndarray:
    if data.ndim == 2:
        return data
    if data.ndim == 3:
        if slice_index is None:
            slice_index = data.shape[2] // 2
        return data[:, :, slice_index]
    if data.ndim == 4:
        if slice_index is None:
            slice_index = data.shape[2] // 2
        return data[:, :, slice_index, 0]
    raise ValueError(f"Unsupported NIfTI dimensionality: {data.ndim}")


def get_slice_count(data: np.ndarray) -> int:
    if data.ndim == 2:
        return 1
    if data.ndim == 3:
        return data.shape[2]
    if data.ndim == 4:
        return data.shape[2]
    raise ValueError(f"Unsupported NIfTI dimensionality: {data.ndim}")


def select_slice_from_volume(data: np.ndarray, slice_index: int) -> np.ndarray:
    if data.ndim == 2:
        return data
    if data.ndim == 3:
        return data[:, :, slice_index]
    if data.ndim == 4:
        return data[:, :, slice_index, 0]
    raise ValueError(f"Unsupported NIfTI dimensionality: {data.ndim}")


def get_slice_positions(data: np.ndarray, affine: np.ndarray) -> np.ndarray:
    if data.ndim == 2:
        return np.array([0.0], dtype=np.float32)
    n_slices = data.shape[2]
    zs = np.empty(n_slices, dtype=np.float32)
    for k in range(n_slices):
        zs[k] = float((affine @ np.array([0.0, 0.0, float(k), 1.0]))[2])
    return zs


def match_slice_indices(z_a: np.ndarray, z_b: np.ndarray, tol: float = 1e-3) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for ia, za in enumerate(z_a):
        diffs = np.abs(z_b - za)
        ib = int(np.argmin(diffs))
        if diffs[ib] <= tol:
            pairs.append((ia, ib))
    return pairs


def crop_to_common_first_dim(img_a: np.ndarray, img_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Crop both images to the smaller first-dimension extent around the middle."""
    def centered_crop(img: np.ndarray, target_width: int) -> np.ndarray:
        if img.shape[0] <= target_width:
            return img
        start = (img.shape[0] - target_width) // 2
        return img[start:start + target_width, ...]
    width = min(img_a.shape[0], img_b.shape[0])
    return centered_crop(img_a, width), centered_crop(img_b, width)


def adjust_affine_for_crop(affine, x0):
    if x0 is None:
        return affine
    affine = affine.copy()
    offset = affine[:3, :3] @ np.array([x0, 0, 0], dtype=np.float64)
    affine[:3, 3] += offset
    return affine


def otsu_threshold(data: np.ndarray, nbins: int = 256) -> float:
    """
    Find a robust foreground/background split point for a single volume,
    independent of whatever the other volume looks like.

    Using `data > 0.0` as a "tissue mask" is not robust: real scanner data
    almost never contains exact zeros in the background (there's a
    non-zero noise floor), so that mask can end up including nearly the
    whole volume for a "raw" image while correctly excluding background
    for a masked/zero-padded image. That asymmetry is what breaks the
    A/B gain matching. Otsu's method instead looks at each volume's own
    intensity histogram and finds the threshold that best separates the
    background/noise mode from the tissue mode, so it degrades gracefully
    whether or not a volume happens to be zero-padded.
    """
    values = data[np.isfinite(data)]
    values = values[values > 0]
    if values.size == 0:
        return 0.0
    hist, bin_edges = np.histogram(values, bins=nbins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    hist = hist.astype(np.float64)

    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    if weight1[-1] == 0:
        return 0.0

    sum1 = np.cumsum(hist * bin_centers)
    mean1 = sum1 / np.maximum(weight1, 1e-12)

    sum2 = np.cumsum((hist * bin_centers)[::-1])[::-1]
    mean2 = sum2 / np.maximum(weight2, 1e-12)

    # Between-class variance for every possible split; skip the last index
    # since weight2[1:] would be undefined past the histogram edge.
    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    if inter_class_variance.size == 0:
        return 0.0
    idx = int(np.argmax(inter_class_variance))
    return float(bin_centers[idx])


def compare_niftis(path_a: str | Path, path_b: str | Path, output: str | Path | None = None,
                   slice_index: int | None = None, iterate_all_slices: bool = False,
                   slice_tol: float = 1e-3, save_normalized: bool = True,
                   low_pct: float = 2.0, high_pct: float = 98.0) -> list[Path]:
    data_a, affine_a = load_nifti(path_a)
    data_b, affine_b = load_nifti(path_b)
    print("runnng spinalcord_gre-main compare_two_niftis.py")
    if data_a.ndim != data_b.ndim:
        raise ValueError(f"Input dimensionalities differ: {data_a.ndim} vs {data_b.ndim}")
    if data_a.ndim == 2:
        pairs = [(None, None)]
    else:
        z_a = get_slice_positions(data_a, affine_a)
        z_b = get_slice_positions(data_b, affine_b)
        pairs = match_slice_indices(z_a, z_b, tol=slice_tol)
        if len(pairs) == 0:
            raise ValueError("No matching slices found between the two NIfTI volumes")
        if not iterate_all_slices:
            if slice_index is None:
                mid = len(pairs) // 2
                pairs = [pairs[mid]]
            else:
                if slice_index < 0 or slice_index >= len(pairs):
                    raise ValueError(f"Slice index {slice_index} is out of range for {len(pairs)} matched slices")
                pairs = [pairs[slice_index]]
    outputs = []

    # ROBUST NORMALIZATION:
    # 1. Isolate structural anatomy voxels using a per-volume Otsu threshold
    #    (NOT `> 0.0` -- see otsu_threshold() docstring for why that's unsafe).
    threshold_a = otsu_threshold(data_a)
    threshold_b = otsu_threshold(data_b)
    tissue_mask_a = data_a > threshold_a
    tissue_mask_b = data_b > threshold_b

    # 2. Anchor a low and high percentile of each volume's tissue distribution.
    #    A single-point match (e.g. just the median) only corrects *gain* around
    #    that one point -- it leaves any *offset*/bias, or a difference in overall
    #    dynamic range, uncorrected, which is why one image can still look
    #    noticeably brighter or dimmer than the other even after matching.
    #    Anchoring two points (low_pct, high_pct) lets us solve for both a scale
    #    and an offset, i.e. a full affine match: a_matched = a*scale + offset.
    lo_a = np.percentile(data_a[tissue_mask_a], low_pct) if np.any(tissue_mask_a) else 0.0
    hi_a = np.percentile(data_a[tissue_mask_a], high_pct) if np.any(tissue_mask_a) else 1.0
    lo_b = np.percentile(data_b[tissue_mask_b], low_pct) if np.any(tissue_mask_b) else 0.0
    hi_b = np.percentile(data_b[tissue_mask_b], high_pct) if np.any(tissue_mask_b) else 1.0
    if hi_a <= lo_a:
        hi_a = lo_a + 1.0
    if hi_b <= lo_b:
        hi_b = lo_b + 1.0

    # 3. Solve the affine map that sends [lo_a, hi_a] -> [lo_b, hi_b], then apply
    #    it to the whole of A. This matches both brightness level and contrast
    #    range of A to B, not just a single scalar gain.
    scale = (hi_b - lo_b) / (hi_a - lo_a)
    offset = lo_b - lo_a * scale
    data_a_matched = data_a * scale + offset

    # 4. Use B's high-percentile anchor as the shared display peak window cap
    # (98th percentile cuts out artifact noise spikes that blow up 99/100 thresholds).
    # Both A and B are now on the same intensity scale by construction, so a single
    # shared vmax_reference is meaningful for both.
    vmax_reference = hi_b
    if vmax_reference <= 0:
        vmax_reference = 1.0

    # Crop and normalize full volumes if requested
    if data_a.ndim > 2 and save_normalized:
        width = min(data_a.shape[0], data_b.shape[0])

        def centered_crop_volume(img: np.ndarray, target_width: int) -> np.ndarray:
            if img.shape[0] <= target_width:
                return img
            start = (img.shape[0] - target_width) // 2
            return img[start:start + target_width, ...]

        data_a_crop = centered_crop_volume(data_a_matched, width)
        data_b_crop = centered_crop_volume(data_b, width)

        # Scale uniformly into a clean relative 0 to 1 visual range using our robust reference
        data_a_norm = np.clip(data_a_crop / vmax_reference, 0.0, 1.0)
        data_b_norm = np.clip(data_b_crop / vmax_reference, 0.0, 1.0)

        if output is None:
            out_dir = Path(".")
            out_base = f"{Path(path_a).stem}_vs_{Path(path_b).stem}"
        else:
            out_path = Path(output)
            out_dir = out_path.parent if out_path.suffix else out_path
            out_base = out_path.stem if out_path.suffix else out_path.name

        out_dir.mkdir(parents=True, exist_ok=True)

        affine_a_adj = adjust_affine_for_crop(affine_a, (data_a.shape[0] - width) // 2 if data_a.shape[0] > width else None)
        affine_b_adj = adjust_affine_for_crop(affine_b, (data_b.shape[0] - width) // 2 if data_b.shape[0] > width else None)

        save_path_a = out_dir / f"{out_base}_A_normalized_cropped.nii.gz"
        save_path_b = out_dir / f"{out_base}_B_normalized_cropped.nii.gz"

        img_a_norm = nib.Nifti1Image(data_a_norm, affine_a_adj)
        img_b_norm = nib.Nifti1Image(data_b_norm, affine_b_adj)

        nib.save(img_a_norm, str(save_path_a))
        nib.save(img_b_norm, str(save_path_b))

        print(f"Saved normalized cropped volume A: {save_path_a}")
        print(f"Saved normalized cropped volume B: {save_path_b}")
        outputs.extend([save_path_a, save_path_b])

    for idx, pair in enumerate(pairs):
        ia, ib = pair
        if ia is None or ib is None:
            a = select_slice(data_a_matched, slice_index=None)
            b = select_slice(data_b, slice_index=None)
            z_text = "2D"
        else:
            a = select_slice_from_volume(data_a_matched, ia)
            b = select_slice_from_volume(data_b, ib)
            z_text = f"z={float((affine_a @ np.array([0.0, 0.0, float(ia), 1.0]))[2]):.2f}"
        a_crop, b_crop = crop_to_common_first_dim(a, b)
        # Scale slice using the robust volumetric bounds
        a_norm = np.clip(a_crop / vmax_reference, 0.0, 1.0)
        b_norm = np.clip(b_crop / vmax_reference, 0.0, 1.0)

        # Calculate a balanced visual difference map
        diff = b_norm - a_norm
        # Build figure output
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        im0 = axes[0].imshow(a_norm.T, cmap="gray", origin="lower", vmin=0.0, vmax=1.0)
        axes[0].set_title(f"Recon A (Matched Baseline) - {z_text}")
        axes[0].axis("off")

        im1 = axes[1].imshow(b_norm.T, cmap="gray", origin="lower", vmin=0.0, vmax=1.0)
        axes[1].set_title(f"Recon B (Original) - {z_text}")
        axes[1].axis("off")

        # Signed structural divergence plot using Blue-White-Red scheme
        im2 = axes[2].imshow(diff.T, cmap="bwr", origin="lower", vmin=-0.2, vmax=0.2)
        axes[2].set_title("Visual Difference (B - A)")
        axes[2].axis("off")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        if output is None:
            fig_path = Path(f"comparison_slice_{idx}.png")
        else:
            out_p = Path(output)
            if out_p.suffix:
                fig_path = out_p.parent / f"{out_p.stem}_slice_{idx}{out_p.suffix}"
            else:
                fig_path = out_p / f"comparison_slice_{idx}.png"

        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved visualization panel to: {fig_path}")
        outputs.append(fig_path)

    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two NIfTI images side-by-side with matched normalization.")
    parser.add_argument("img_a", help="Path to NIfTI image A")
    parser.add_argument("img_b", help="Path to NIfTI image B")
    parser.add_argument("--out", help="Output directory or file path prefix")
    parser.add_argument("--slice-index", type=int, default=None,
                         help="Index into the matched-slice list to plot a single specific slice "
                              "(ignored if --all-slices is set). Defaults to the middle slice.")
    parser.add_argument("--all-slices", action="store_true",
                         help="Plot every matched slice instead of just the middle one.")
    parser.add_argument("--slice-tol", type=float, default=1e-3,
                         help="Tolerance (mm) for matching slice z-positions between A and B.")
    parser.add_argument("--no-save-normalized", action="store_true",
                         help="Skip saving the full normalized/cropped .nii.gz volumes, only produce PNG panels.")
    parser.add_argument("--low-percentile", type=float, default=2.0,
                         help="Low percentile of each volume's tissue used as the brightness-matching anchor (default: 2).")
    parser.add_argument("--high-percentile", type=float, default=99.0,
                         help="High percentile of each volume's tissue used as the brightness-matching anchor "
                              "and display peak (default: 98).")
    args = parser.parse_args()
    compare_niftis(
        args.img_a,
        args.img_b,
        output=args.out,
        slice_index=args.slice_index,
        iterate_all_slices=args.all_slices,
        slice_tol=args.slice_tol,
        save_normalized=not args.no_save_normalized,
        low_pct=args.low_percentile,
        high_pct=args.high_percentile,
    )