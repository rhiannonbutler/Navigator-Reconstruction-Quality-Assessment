#!/usr/bin/env python3
"""
Reconstruct 4 echo volumes from a folder of single-slice NIfTIs.

Usage:
  python reconstruct_from_slices.py \
      --slices_dir SLICES_DIR \
      --refs ref_echo0.nii.gz ref_echo1.nii.gz ref_echo2.nii.gz ref_echo3.nii.gz \
      --out_dir OUT_DIR

The script matches each single-slice file to the best-fitting slice index
in the reference volumes (one NIfTI per echo) using normalized correlation.
It then assembles four output volumes (one per echo) with slices placed
at the matched indices and saves them with the affine of the corresponding
reference file.
"""
import re
import sys
from pathlib import Path
import numpy as np
try:
    import nibabel as nib
except Exception:
    print("Please install nibabel (pip install nibabel)")
    raise
import argparse


def load_nifti(path):
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    return data, img.affine, img.header


def normalize_for_match(img):
    a = img.astype(np.float32)
    a = a - np.nanmean(a)
    denom = np.sqrt(np.nansum(a * a))
    if denom == 0:
        return a, 0.0
    return a / (denom + 1e-12), denom


def best_match_index(single_slice, ref_volume):
    h, w = single_slice.shape
    if ref_volume.shape[0] != h or ref_volume.shape[1] != w:
        raise ValueError(f"Shape mismatch: single {single_slice.shape} vs ref slice {ref_volume.shape[:2]}")

    s_flat, _ = normalize_for_match(single_slice)
    best_k = 0
    best_score = -np.inf
    for k in range(ref_volume.shape[2]):
        ref_slice = ref_volume[:, :, k]
        r_flat, _ = normalize_for_match(ref_slice)
        score = float(np.nansum(s_flat * r_flat))
        if score > best_score:
            best_score = score
            best_k = k
    return best_k, best_score


def extract_echo_volumes(data):
    """Accept either 2D, (H,W,1), or (H,W,E) and return a list of E 2D images."""
    if data.ndim == 2:
        return [data.astype(np.float32)]
    if data.ndim == 3 and data.shape[2] == 1:
        return [data[:, :, 0].astype(np.float32)]
    if data.ndim == 3 and data.shape[2] >= 2:
        return [data[:, :, e].astype(np.float32) for e in range(data.shape[2])]
    raise ValueError(f"Unsupported image shape {data.shape}")


def crop_ref_volume(ref_volume, x0, x1):
    if x0 is None or x1 is None:
        return ref_volume
    return ref_volume[x0:x1, :, :]


def adjust_affine_for_crop(affine, x0):
    if x0 is None:
        return affine
    affine = affine.copy()
    offset = affine[:3, :3] @ np.array([x0, 0, 0], dtype=np.float64)
    affine[:3, 3] += offset
    return affine


def infer_echo_index(path):
    name = path.name.lower()
    # Accept 0-based labels (0,1,2,3), 1-based labels (1,2,3,4),
    # and plain numeric suffixes like "_1" or "-1".
    for idx in range(4):
        for token in [f"e{idx}", f"echo{idx}", f"echo_{idx}", f"ech{idx}"]:
            if token in name:
                return idx

    for idx in range(1, 5):
        for token in [f"e{idx}", f"echo{idx}", f"echo_{idx}", f"ech{idx}"]:
            if token in name:
                return idx - 1

        if re.search(rf'(?<![a-z]){idx}(?!\d)', name):
            return idx - 1

    return None


def discover_reference_files(refs_dir):
    refs_dir = Path(refs_dir)
    if not refs_dir.exists():
        raise FileNotFoundError(refs_dir)
    files = sorted([p for p in refs_dir.iterdir() if p.is_file() and p.suffix.lower().endswith((".nii", ".gz"))])
    if len(files) < 4:
        raise FileNotFoundError(f"Could not discover 4 reference files in {refs_dir}; found {len(files)}")

    found = {}
    for p in files:
        idx = infer_echo_index(p)
        if idx is not None:
            found[idx] = p

    if len(found) == 4:
        return [found[i] for i in range(4)]

    # Fallback: if the folder contains 4 files but only 3 distinct echo labels
    # (common when one echo is missing a label), simply use the first four files.
    if len(files) >= 4:
        print(f"Warning: could not resolve all 4 echo labels from filenames; using the first 4 files in the folder")
        return files[:4]

    raise FileNotFoundError(f"Could not discover all 4 reference files in {refs_dir}; found {sorted(found)}")


def main():
    parser = argparse.ArgumentParser(description="Reconstruct 4 echo volumes from single-slice NIfTIs")
    parser.add_argument("--slices_dir", required=True)
    parser.add_argument("--refs", nargs="*", default=None, help="Optional explicit list of four reference NIfTIs")
    parser.add_argument("--refs_dir", default=None, help="Directory containing the four reference NIfTIs; auto-discovers echo0..echo3 by filename")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--crop_x0", type=int, default=None, help="Start x index of the cropped input slices")
    parser.add_argument("--crop_x1", type=int, default=None, help="End x index (exclusive) of the cropped input slices")
    parser.add_argument("--score_thresh", type=float, default=0.1, help="Minimum match score to accept")
    args = parser.parse_args()

    slices_dir = Path(args.slices_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.refs:
        refs = [Path(p) for p in args.refs]
        if len(refs) != 4:
            raise ValueError("Provide exactly 4 reference files with --refs")
    elif args.refs_dir:
        refs = discover_reference_files(args.refs_dir)
    else:
        raise ValueError("Provide either --refs or --refs_dir")

    ref_vols = []
    affines = []
    headers = []
    for r in refs:
        data, affine, header = load_nifti(r)
        if data.ndim != 3:
            raise ValueError(f"Reference {r} must be 3D (H,W,nz)")
        ref_vols.append(data.astype(np.float32))
        affines.append(affine)
        headers.append(header)

    H, W, NZ = ref_vols[0].shape
    for rv in ref_vols:
        if rv.shape != (H, W, NZ):
            raise ValueError("All reference volumes must have identical shape")

    ref_vols_cropped = [crop_ref_volume(rv, args.crop_x0, args.crop_x1) for rv in ref_vols]
    cropped_affines = [adjust_affine_for_crop(aff, args.crop_x0) for aff in affines]

    # Keep only the slices that were actually matched from the input files.
    out_vols = [[] for _ in range(4)]
    slice_indices = [[] for _ in range(4)]

    files = sorted([p for p in slices_dir.iterdir() if p.is_file() and p.suffix.lower().endswith((".nii", ".gz"))])
    if not files:
        print("No NIfTI files found in slices_dir")
        sys.exit(1)

    print(f"Found {len(files)} input files; matching to {NZ} slices per echo")

    for f in files:
        data, _, _ = load_nifti(f)
        try:
            images = extract_echo_volumes(data)
        except Exception as e:
            print(f"Skipping {f.name}: {e}")
            continue

        if len(images) != 4:
            print(f"Skipping {f.name}: expected 4 echoes per file, got {len(images)}")
            continue

        matched_indices = []
        scores = []
        for e in range(4):
            idx, score = best_match_index(images[e], ref_vols_cropped[e])
            matched_indices.append(idx)
            scores.append(score)

        # Choose the best common slice index across echoes.
        sums = {}
        for k, score in zip(matched_indices, scores):
            sums.setdefault(k, 0.0)
            sums[k] += score
        final_k = max(sums.items(), key=lambda x: x[1])[0]
        avg_score = float(np.mean(scores))

        if avg_score < args.score_thresh:
            print(f"Low score {avg_score:.3f} for {f.name}; skipping")
            continue

        for e in range(4):
            out_vols[e].append(images[e])
            slice_indices[e].append(final_k)

        print(f"Placed {f.name} -> slice {final_k} scores={[round(s,3) for s in scores]} avg={avg_score:.3f}")

    for e in range(4):
        if len(out_vols[e]) == 0:
            print(f"Echo {e}: no slices matched; skipping output")
            continue

        ordered = sorted(zip(slice_indices[e], out_vols[e]), key=lambda x: x[0])
        volume = np.stack([img for _, img in ordered], axis=2)
        print(f"Echo {e}: matched {volume.shape[2]} slices")
        out_img = nib.Nifti1Image(volume, cropped_affines[e], header=headers[e])
        out_path = out_dir / f"reconstructed_echo{e}.nii.gz"
        nib.save(out_img, str(out_path))
        print(f"Saved {out_path}")


if __name__ == '__main__':
    main()
