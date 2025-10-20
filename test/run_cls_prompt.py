import os
import pandas as pd
from tqdm import tqdm

from test_utils import model_infer, get_lung_center, load_model


def run_cls_prompt(
        gpu_id,
        vision_encoder_dir,
        text_encoder_dir,
        image_dir,
        center_csv,
        lesion_names,
        output_path
):
    """
    Runs prompt-based zero-shot classification on medical images using a trained model.

    Args:
        gpu_id (int): ID of the GPU to use for inference.
        vision_encoder_dir (str): Path to the vision encoder model directory.
        text_encoder_dir (str): Path to the text encoder model directory.
        image_dir (str): Directory containing input images.
        center_csv (str): Path to CSV file with lung center coordinates.
        lesion_names (list): List of lesion names to classify.
        output_path (str): Directory to save the results.
    """
    # Setup device and output directory
    device = 'cuda:{}'.format(gpu_id)
    os.makedirs(output_path, exist_ok=True)
    # Load lung center coordinates and prepare image list
    image_to_center = get_lung_center(center_csv)
    image_list = list(sorted(os.listdir(image_dir)))

    # Prepare text prompts for each lesion type
    text_list = [['{}'.format(lesion_name)] for lesion_name in lesion_names]

    # Load model and run prompt-based inference on each image
    circle_model, tokenizer = load_model(vision_encoder_dir, text_encoder_dir, num_classes=4)
    res = []
    for image_name in tqdm(image_list):
        image_path = os.path.join(image_dir, image_name, "CT.nii.gz")
        if not os.path.exists(image_path):
            print(f"{image_path} not existed, continue")
            continue
        if image_name not in image_to_center:
            print(f"{image_name} lung center not existed, continue")
            continue
        crop_center = image_to_center[image_name]
        clip_prob_list, _, _ = model_infer(circle_model, tokenizer, image_path, crop_center, text_list, device)
        res.append([image_name] + clip_prob_list)

    # Save prompt-based classification results to CSV file
    res_df = pd.DataFrame(res, columns=["image_name"] + lesion_names)
    res_df.to_csv(os.path.join(output_path, "result_prompt_cls.csv"), index=False, encoding="utf-8-sig")


if __name__ == '__main__':
    import argparse

    # Parse command-line arguments for prompt-based classification
    parser = argparse.ArgumentParser(description='infer prompt-based(sigmoid) zero-shot')
    parser.add_argument('--gpu_id', default=0)
    parser.add_argument('--vision_encoder_dir',
                        default="/mnt/maui/Med_VLM/project/94_paper/20251020/model/visual_transformer_4cls.bin")
    parser.add_argument('--text_encoder_dir',
                        default="/mnt/maui/Med_VLM/project/94_paper/20251020/model/nlp_roberta_backbone_base_std")
    parser.add_argument('--image_dir', default="/mnt/maui/Med_VLM/project/94_paper/CIRCLE_ZS2K/image/")
    parser.add_argument('--center_csv', default="/mnt/maui/Med_VLM/project/94_paper/CIRCLE_ZS2K/label/lung_center.csv")
    parser.add_argument('--lesion_names',
                        default=["纵隔胸腺瘤", "肝密度减低或脂肪肝", "引流管", "起搏器", "胸腺退化不全",
                                 "食管狭窄", "乳缺如", "冠脉支架", "冠脉钙化", "主动脉钙化"])
    parser.add_argument('--output_path', default="/mnt/maui/Med_VLM/project/94_paper/CIRCLE_ZS2K/output")
    args = parser.parse_args()

    # Execute prompt-based classification pipeline
    run_cls_prompt(
        args.gpu_id,
        args.vision_encoder_dir,
        args.text_encoder_dir,
        args.image_dir,
        args.center_csv,
        args.lesion_names,
        args.output_path
    )
