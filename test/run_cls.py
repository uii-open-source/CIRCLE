import os
import pandas as pd
from tqdm import tqdm

from test_utils import model_infer, get_lung_center, load_model


def run_cls(
        gpu_id,
        vision_encoder_dir,
        text_encoder_dir,
        image_dir,
        center_csv,
        output_path
):
    device = 'cuda:{}'.format(gpu_id)
    os.makedirs(output_path, exist_ok=True)
    image_to_center = get_lung_center(center_csv)
    image_list = list(sorted(os.listdir(image_dir)))
    cls_label_names = ["Pulmonary nodule/mass", "Cardiomegaly", "Pleural effusion", "Pneumothorax"]

    circle_model, tokenizer = load_model(vision_encoder_dir, text_encoder_dir, num_classes=4)
    res_cls_prob = []
    for image_name in tqdm(image_list):
        image_path = os.path.join(image_dir, image_name, "CT.nii.gz")
        if not os.path.exists(image_path):
            print(f"{image_path} not existed, continue")
            continue
        if image_name not in image_to_center:
            print(f"{image_name} lung center not existed, continue")
            continue
        crop_center = image_to_center[image_name]
        _, cls_prob, _ = model_infer(circle_model, tokenizer, image_path, crop_center, [], device)
        res_cls_prob.append([image_name] + list(cls_prob[0]))

    res_cls_prob = pd.DataFrame(res_cls_prob, columns=["image_name"] + cls_label_names)
    res_cls_prob.to_csv(os.path.join(output_path, "result_cls_prob.csv"), index=False, encoding="utf-8-sig")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='infer classification')
    parser.add_argument('--gpu_id', default=0)
    parser.add_argument('--vision_encoder_dir',
                        default="/mnt/maui/Med_VLM/project/94_paper/20251020/model/visual_transformer_4cls.bin")
    parser.add_argument('--text_encoder_dir',
                        default="/mnt/maui/Med_VLM/project/94_paper/20251020/model/nlp_roberta_backbone_base_std/")
    parser.add_argument('--image_dir', default="/mnt/maui/Med_VLM/project/94_paper/CIRCLE_ZS2K/image/")
    parser.add_argument('--center_csv', default="/mnt/maui/Med_VLM/project/94_paper/CIRCLE_ZS2K/label/lung_center.csv")
    parser.add_argument('--output_path', default="/mnt/maui/Med_VLM/project/94_paper/CIRCLE_ZS2K/output_md")
    args = parser.parse_args()

    run_cls(
        args.gpu_id,
        args.vision_encoder_dir,
        args.text_encoder_dir,
        args.image_dir,
        args.center_csv,
        args.output_path
    )
