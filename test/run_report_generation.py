import argparse
import os
from contextlib import nullcontext

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from model.circle_report import CIRCLEReport
from test_utils import get_lung_center, prepare_image


# Canonical list of 37 disease label names used for report prompt construction
LABEL_37_NAMES = [
    '肺部阴影', '肺部结节/肿块', '肺内钙化', '肺部透亮影', '肺实变', '肺部炎症', '肺结核', '间质改变', '肺不张', '肺水肿',
    '肺气肿', '肺大泡/肺气囊腔', '纵隔肿块', '纵隔钙化', '纵隔淋巴结肿大', '气管扩张/增厚', '气管憩室', '支气管炎', '心包积液',
    '心包增厚', '心肥大', '心脏及血管钙化', '肺动脉增粗', '肺动脉高压', '主动脉增粗', '胸腔积液', '气胸', '胸膜钙化',
    '胸膜增厚', '胸膜胸壁结节', '食管裂孔疝', '食管增厚', '骨折', '骨质破坏', '骨肿瘤', '术后', '设备植入'
]


def _build_question_from_labels(label_row, label_cols):
    """
    Build a question prompt string from a row of binary disease labels.
    Args:
        label_row: a pandas Series (one row of the label DataFrame).
        label_cols: list of column names corresponding to the 37 label classes.
    Returns:
        str: question string containing positive label names for report generation.
    """
    # Collect names of positive (value == 1) labels
    positive_labels = []
    for idx, col in enumerate(label_cols):
        if idx >= len(LABEL_37_NAMES):
            break
        if int(label_row[col]) == 1:
            positive_labels.append(LABEL_37_NAMES[idx])

    # Join positive labels or use '无' when none are present
    label_str = '，'.join(positive_labels) if positive_labels else '无'
    return f"异常标签是：{label_str}。根据图像和异常标签给出准确的影像所见和诊断结论。"


def _load_report_model(llm_model_path, vision_encoder_path, visual_mapper_path, device):
    """
    Load and assemble the CIRCLEReport model from pretrained components.
    Args:
        llm_model_path (str): Path to the pretrained LLM (tokenizer + weights).
        vision_encoder_path (str): Path to pretrained vision encoder weights; skipped if empty or missing.
        visual_mapper_path (str): Path to pretrained visual mapper weights; skipped if empty or missing.
        device: torch device to move the model onto.
    Returns:
        CIRCLEReport: assembled model in eval mode on the target device.
    """
    # Load LLM tokenizer and base model weights
    tokenizer = AutoTokenizer.from_pretrained(llm_model_path, use_fast=False, trust_remote_code=True)
    gpt_model = AutoModelForCausalLM.from_pretrained(
        llm_model_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )

    # Build CIRCLEReport with frozen LLM
    circle_report_model = CIRCLEReport(
        llm_hidden_size=gpt_model.config.hidden_size,
        gpt_tokenizer=tokenizer,
        gpt_model=gpt_model,
        train_gpt=False,
    )

    # Resolve vision encoder path; warn and skip if the file does not exist
    vision_path_to_load = None
    if vision_encoder_path:
        if os.path.exists(vision_encoder_path):
            vision_path_to_load = vision_encoder_path
        else:
            print(f"warning: vision_encoder_path not found, skip loading: {vision_encoder_path}")

    # Resolve visual mapper path; warn and skip if the file does not exist
    visual_mapper_path_to_load = None
    if visual_mapper_path:
        if os.path.exists(visual_mapper_path):
            visual_mapper_path_to_load = visual_mapper_path
        else:
            print(f"warning: visual_mapper_path not found, skip loading: {visual_mapper_path}")

    # Load pretrained vision encoder and mapper weights into the model
    circle_report_model.load(
        vision_path=vision_path_to_load,
        visual_mapper_path=visual_mapper_path_to_load,
        load_clip_weight=True,
    )

    # Move model to target device and switch to eval mode
    circle_report_model = circle_report_model.to(device)
    circle_report_model.eval()
    return circle_report_model


def run_report_generation(
    gpu_id,
    image_dir,
    lung_center_csv,
    predict_label_csv,
    output_csv,
    llm_model_path="Qwen3-8B",
    vision_encoder_path="",
    visual_mapper_path="",
):
    """
    Run CT report generation for all samples in predict_label_csv and save results.
    Args:
        gpu_id (int): GPU id for inference.
        image_dir (str): Root directory containing per-case image folders.
        lung_center_csv (str): CSV path with lung center coordinates for image cropping.
        predict_label_csv (str): CSV path with predicted binary disease labels; must contain image_name column.
        output_csv (str): Output CSV path; result contains image_name and report_generation columns.
        llm_model_path (str): Path to the pretrained LLM used for generation.
        vision_encoder_path (str): Optional path to pretrained vision encoder weights.
        visual_mapper_path (str): Optional path to pretrained visual mapper weights.
    """
    if not str(llm_model_path).strip():
        raise ValueError("llm_model_path can not be empty")

    # Select device based on CUDA availability
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"use device: {device}")

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    # Load lung center coordinates and disease label predictions
    image_to_center = get_lung_center(lung_center_csv)
    label_df = pd.read_csv(predict_label_csv, dtype={"image_name": str})
    if "image_name" not in label_df.columns:
        raise ValueError("predict_label_csv must contain column: image_name")

    # Extract label column names (all columns except image_name)
    label_cols = [col for col in label_df.columns if col != "image_name"]
    if len(label_cols) < len(LABEL_37_NAMES):
        print(
            f"warning: label column count is {len(label_cols)}, expected >= {len(LABEL_37_NAMES)}. "
            f"only first {len(label_cols)} labels will be used."
        )

    # Load CIRCLEReport model
    circle_report_model = _load_report_model(
        llm_model_path=llm_model_path,
        vision_encoder_path=vision_encoder_path,
        visual_mapper_path=visual_mapper_path,
        device=device,
    )

    # Iterate over all samples and generate reports
    records = []
    for _, row in tqdm(label_df.iterrows(), total=label_df.shape[0], desc="report_generation"):
        image_name = str(row["image_name"])
        image_path = os.path.join(image_dir, image_name, "CT.nii.gz")

        # Skip samples with missing lung center or missing image file
        if image_name not in image_to_center:
            print('warning: {} not has center'.format(image_name))
            continue
        if not os.path.exists(image_path):
            print('warning: {} not exist'.format(image_path))
            continue

        crop_center = image_to_center[image_name]
        # Build question prompt from predicted disease labels
        question = _build_question_from_labels(row, label_cols)

        # Run inference with mixed precision on GPU, plain context on CPU
        amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
        with torch.no_grad(), amp_ctx:
            image_tensor = prepare_image(image_path, crop_center)
            # Add batch dimension and move to device
            image_tensor = torch.unsqueeze(image_tensor, dim=0).to(device)
            report_text = circle_report_model.run_report_generation(image_tensor, question)

        records.append({
            "image_name": image_name,
            "report_generation": report_text,
        })

    # Save all generated reports to output CSV
    pd.DataFrame(records).to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"save done: {output_csv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='infer report generation')
    parser.add_argument('--image_dir', default='data/image')
    parser.add_argument('--lung_center_csv', default='data/lung_center.csv')
    parser.add_argument('--predict_label_csv', default='data/predict_label.csv')
    parser.add_argument('--output_csv', default='output/report_generation_output.csv')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--llm_model_path', default='Qwen3-8B')
    parser.add_argument('--vision_encoder_path', default='')
    parser.add_argument('--visual_mapper_path', default='')
    args = parser.parse_args()

    run_report_generation(
        gpu_id=args.gpu_id,
        image_dir=args.image_dir,
        lung_center_csv=args.lung_center_csv,
        predict_label_csv=args.predict_label_csv,
        output_csv=args.output_csv,
        llm_model_path=args.llm_model_path,
        vision_encoder_path=args.vision_encoder_path,
        visual_mapper_path=args.visual_mapper_path,
    )
