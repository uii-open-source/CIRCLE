import os
from itertools import product

import numpy as np
import SimpleITK as sitk
import torch

from model.prompt_seg.rex_segmentation import RexGroundingCTSeg
from train.dataset import (
    crop_image as sitk_crop_image,
    intensity_normalize,
    get_lung_center,
)


_NORM_MEAN = -100
_NORM_STDDEV = 900
_NORM_CLIP = True

_CROP_SIZE_VOX = (384, 320, 224)
_CROP_SPACING_MM = (0.65, 0.65, 1.15)
_CROP_AXES = np.array(
    [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]],
    dtype=np.double,
)

_SLIDE_STEP_WORLD_MM = [50, 50, 50]


def _sitk_crop_norm_to_tensor(sitk_image, crop_center_world):
    """Dataset-standard crop + normalisation -> 5D CPU tensor."""
    cropped_sitk = sitk_crop_image(
        sitk_image,
        np.asarray(crop_center_world, dtype=np.float64),
        np.asarray(_CROP_SPACING_MM, dtype=np.float64),
        np.asarray(_CROP_SIZE_VOX, dtype=np.float64),
        _CROP_AXES,
        default_value=-1024,
    )
    arr = sitk.GetArrayFromImage(cropped_sitk)
    arr = intensity_normalize(arr, _NORM_MEAN, _NORM_STDDEV, _NORM_CLIP)
    arr = np.ascontiguousarray(arr)
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()
    return tensor, cropped_sitk


def _resample_prob_to_original(crop_sitk_prob, ref_sitk_image):
    """Linear-resample a crop-space prob map into the original CT grid."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitk.Cast(ref_sitk_image, sitk.sitkFloat32))
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)
    out = resampler.Execute(sitk.Cast(crop_sitk_prob, sitk.sitkFloat32))
    return sitk.GetArrayFromImage(out).astype(np.float32)


def _write_mask_like(mask_uint8, ref_sitk_image, out_path):
    """Write uint8 mask with same geom as ref_sitk_image to disk."""
    sitk_img = sitk.GetImageFromArray(mask_uint8)
    sitk_img.SetOrigin(ref_sitk_image.GetOrigin())
    sitk_img.SetSpacing(ref_sitk_image.GetSpacing())
    sitk_img.SetDirection(ref_sitk_image.GetDirection())
    sitk.WriteImage(sitk.Cast(sitk_img, sitk.sitkUInt8), out_path)


def _encode_prompt_runtime(text_model_name_or_path, prompt_text,
                           device, max_seq_len=256):
    """
    Returns:
        embed_dict with keys:
            "text_emb"   : (1, D)  pooled global vector
            "text_tokens": (1, L, D)  per-token last hidden state
            "text_mask"  : (1, L)  attention mask
    """
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        raise ValueError(
            "[run_prompt_seg] prompt_text must be a non-empty string. "
            "Dynamic embedding encoding requires the raw prompt text."
        )
    if not isinstance(text_model_name_or_path, str) or \
            not text_model_name_or_path.strip():
        raise ValueError(
            "[run_prompt_seg] text_model_name_or_path must be provided for "
            "runtime prompt encoding."
        )

    from model.prompt_seg.text_encoder import Qwen3TextEncoder

    text_encoder = Qwen3TextEncoder(
        model_name_or_path=text_model_name_or_path,
        freeze=True,
        max_seq_len=max_seq_len,
    )
    text_encoder.eval()
    text_encoder.to(device)

    with torch.no_grad():
        text_emb, text_tokens, text_mask = text_encoder(
            [prompt_text], device=device
        )

    embed_dict = {}
    for key, t in (("text_emb", text_emb),
                   ("text_tokens", text_tokens),
                   ("text_mask", text_mask)):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        t = t.detach().float()

        if key == "text_emb" and t.ndim == 1:
            t = t.unsqueeze(0)
        if key == "text_tokens" and t.ndim == 2:
            t = t.unsqueeze(0)
        if key == "text_mask" and t.ndim == 1:
            t = t.unsqueeze(0)
        t = t.requires_grad_(False).to(device)
        embed_dict[key] = t
    return embed_dict


def _sliding_window_prob(model, device, sitk_ct, center_world, embed_dict):
    orig_size = np.asarray(sitk_ct.GetSize())[::-1]   # (D, H, W) array order
    full_prob_map = np.zeros(orig_size, dtype=np.float32)
    count_map = np.zeros(orig_size, dtype=np.float32)

    center_arr = np.asarray(center_world, dtype=np.float64)
    offsets = [
        np.array([x, y, z], dtype=np.float64) * _SLIDE_STEP_WORLD_MM
        for x, y, z in product([-1, 0, 1], repeat=3)
    ]

    with torch.no_grad():
        for offset in offsets:
            win_center = center_arr + offset
            crop_tensor, crop_sitk = _sitk_crop_norm_to_tensor(sitk_ct, win_center)
            crop_tensor = crop_tensor.to(device)

            prob_logits = model.predict(crop_tensor, embed_dict=embed_dict)
            prob_arr = prob_logits.squeeze(0).squeeze(0).cpu().numpy()

            # Rebuild crop-space sitk image with correct geometry
            crop_prob_sitk = sitk.GetImageFromArray(np.ascontiguousarray(prob_arr))
            crop_prob_sitk.SetOrigin(crop_sitk.GetOrigin())
            crop_prob_sitk.SetSpacing(crop_sitk.GetSpacing())
            crop_prob_sitk.SetDirection(crop_sitk.GetDirection())

            prob_original = _resample_prob_to_original(crop_prob_sitk, sitk_ct)

            full_prob_map += prob_original
            count_map[prob_original > 0] += 1

    count_map[count_map == 0] = 1
    return full_prob_map / count_map

def run_prompt_seg(
        gpu_id,
        model_path,
        image_path,
        output_mask_path,
        text_model_name_or_path,
        prompt_text,
        center_csv,
        image_name,
        *,
        text_dim=None,
        max_seq_len=256,
        prob_threshold=0.5,
):
    """
    Run single-image / single-prompt segmentation and write one mask file.

    Args:
        gpu_id (int):               CUDA device ordinal (used for both the
                                    segmentation model AND the text encoder).
        model_path (str):           Path to RexGroundingCTSeg checkpoint.
        image_path (str):           Path to input CT (.nii/.nii.gz).
        output_mask_path (str):     Destination for binary prompt mask (.nii.gz).
        text_model_name_or_path (str): HF repo / local folder for the Qwen3
                                    AutoModel / AutoTokenizer used at runtime.
        prompt_text (str):          **Non-empty** raw prompt string.
        center_csv (str):           Lung-centre CSV used to look up the crop
                                    centre for `image_name`.
        image_name (str):           Key under which the lung centre is stored
                                    in `center_csv` (matches CIRCLE naming).
        text_dim (int | None):      Text feature dimension.  If None (default)
                                    inferred from the runtime encoder output.
        max_seq_len (int):          Tokeniser truncation / padding length for
                                    the Qwen3 text encoder (default 256).
        prob_threshold (float):     Binarisation threshold in (0, 1).
    """
    device = f"cuda:{gpu_id}"

    # --- Validate inputs early -------------------------------------------
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"[run_prompt_seg] image not found: {image_path}")
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        raise ValueError(
            "[run_prompt_seg] prompt_text must be a non-empty string. "
            "Embed is now computed at runtime from the raw prompt."
        )
    if not isinstance(text_model_name_or_path, str) or \
            not text_model_name_or_path.strip():
        raise ValueError(
            "[run_prompt_seg] text_model_name_or_path must be provided for "
            "runtime prompt embedding encoding."
        )
    if not os.path.exists(center_csv):
        raise FileNotFoundError(f"[run_prompt_seg] centre CSV not found: {center_csv}")
    if not (0.0 < prob_threshold < 1.0):
        raise ValueError(f"prob_threshold must be in (0, 1), got {prob_threshold}")
    if not (isinstance(max_seq_len, int) and max_seq_len >= 1):
        raise ValueError(f"max_seq_len must be >= 1, got {max_seq_len}")

    # --- Resolve crop centre ---------------------------------------------
    center_dict = get_lung_center(center_csv)
    if image_name not in center_dict:
        # Tolerate .nii.gz suffix in lookup key
        stripped = image_name
        if stripped.endswith(".nii.gz"):
            stripped = stripped[:-len(".nii.gz")]
        if stripped in center_dict:
            crop_center_world = center_dict[stripped]
        else:
            raise KeyError(
                f"[run_prompt_seg] '{image_name}' (nor '{stripped}') "
                f"present in centre CSV.  Available keys sample: "
                f"{list(center_dict.keys())[:5]}..."
            )
    else:
        crop_center_world = center_dict[image_name]
    crop_center_world = np.asarray(crop_center_world, dtype=np.float64)

    # --- Build embed_dict ------------------------------------
    embed_dict = _encode_prompt_runtime(
        text_model_name_or_path, prompt_text, device, max_seq_len=max_seq_len,
    )
    # --- Auto-detect text_dim from the freshly-computed embed ------------
    if text_dim is None:
        text_dim = int(embed_dict["text_emb"].shape[-1])

    # --- Load segmentation model ----------------------------------------
    model = RexGroundingCTSeg(text_dim=text_dim)
    ckpt = torch.load(model_path, map_location="cpu")
    model.load_state_dict(ckpt, strict=True)
    model = model.to(device).eval()

    # --- Run sliding-window inference ------------------------------------
    sitk_ct = sitk.ReadImage(image_path)
    prob_map = _sliding_window_prob(model, device, sitk_ct, crop_center_world,
                                     embed_dict)
    binary_mask = (prob_map >= prob_threshold).astype(np.uint8)

    # --- Write single output mask ----------------------------------------
    out_dir = os.path.dirname(os.path.abspath(output_mask_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    _write_mask_like(binary_mask, sitk_ct, output_mask_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Single-image / single-prompt prompt segmentation. "
    )
    parser.add_argument("--gpu_id",          type=int, required=True)
    parser.add_argument("--model_path",      type=str, required=True,
                        help="RexGroundingCTSeg checkpoint path")
    parser.add_argument("--image_path",      type=str, required=True,
                        help="Input CT .nii/.nii.gz file path")
    parser.add_argument("--output_mask_path", type=str, required=True,
                        help="Destination .nii(.gz) for the output binary mask")
    parser.add_argument("--text_model_name_or_path", type=str, required=True,
                        help="Qwen3 AutoModel/AutoTokenizer local folder or HF repo id "
                             "(same folder used by train.precompute_text_embeddings)")
    parser.add_argument("--prompt_text",     type=str, required=True,
                        help="**Non-empty raw prompt string** encoded at runtime")
    parser.add_argument("--center_csv",      type=str, required=True,
                        help="Lung-centre CSV (image_name, lung_center_world_x/y/z)")
    parser.add_argument("--image_name",      type=str, required=True,
                        help="Lookup key in centre CSV for this case")
    parser.add_argument("--text_dim",        type=int, default=None,
                        help="Text embedding dim.  Auto-detected by default.")
    parser.add_argument("--max_seq_len",     type=int, default=256,
                        help="Tokenizer truncation/padding length for Qwen3 (default 256)")
    parser.add_argument("--prob_threshold",  type=float, default=0.5,
                        help="Probability → binary mask threshold (0, 1)")

    args = parser.parse_args()
    run_prompt_seg(
        gpu_id=args.gpu_id,
        model_path=args.model_path,
        image_path=args.image_path,
        output_mask_path=args.output_mask_path,
        text_model_name_or_path=args.text_model_name_or_path,
        prompt_text=args.prompt_text,
        center_csv=args.center_csv,
        image_name=args.image_name,
        text_dim=args.text_dim,
        max_seq_len=args.max_seq_len,
        prob_threshold=args.prob_threshold,
    )
