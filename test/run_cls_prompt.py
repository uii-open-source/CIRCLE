import os
import pandas as pd
import numpy as np

from test_utils import model_infer, get_lung_center


def run_cls_prompt(
    gpu_id, 
    vision_encoder_dir, 
    text_encoder_dir, 
    image_dir,
    center_csv,
    lesion_name,
    output_path):
    
    device = 'cuda:{}'.format(gpu_id)
    os.makedirs(output_path, exist_ok=True)

    image_to_center = get_lung_center(center_csv)
    image_list = list(sorted(os.listdir(image_dir)))
    
    pathologies = [lesion_name]
    text_list = [[f"存在{pathology}.", f"不存在{pathology}."] for pathology in pathologies]
    
    # cls_label_threshold = {"肺结节肿块": 0.5, 
    #                        "气胸": 0.5, 
    #                        "胸腔积液": 0.5, 
    #                        "心脏增大": 0.5}
    # num_classes = 37
    
    res = []
    
    for image_name in image_list:
        image_path = os.path.join(image_dir, image_name, "CT.nii.gz")
        if not os.path.exists(image_path):
            print(f"{image_path} not existed, continue")
            continue
        if image_name not in image_to_center:
            print(f"{image_name} lung center not existed, continue")
            continue
        crop_center = image_to_center[image_name]
        clip_prob_list, cls_logits, feature = model_infer(
            vision_encoder_dir, text_encoder_dir, 
            image_path, crop_center, text_list, device)          
        res.append([image_name]+clip_prob_list)
        
    res = pd.DataFrame(res, columns=["image_name"]+pathologies)
    res.to_csv(os.path.join(output_path, "result_prompt_cls.csv"), index=False, encoding="utf-8-sig")
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='infer clip')
    parser.add_argument('--gpu_id', default=0)
    parser.add_argument('--vision_encoder_dir', default="/path/to/vision_encoder")
    parser.add_argument('--text_encoder_dir', default='/path/to/text_encoder')
    parser.add_argument('--image_dir', default='/path/to/images')
    parser.add_argument('--center_csv', default='/path/to/center_csv')
    parser.add_argument('--abnormality_name', default='胸腔积液')
    parser.add_argument('--output_path', default="/path/to/output")
    args = parser.parse_args()
    
    run_cls_prompt(
        args.gpu_id, 
        args.vision_encoder_dir, 
        args.text_encoder_dir,
        args.image_dir,
        args.center_csv,
        args.abnormality_name,
        args.output_path
        )