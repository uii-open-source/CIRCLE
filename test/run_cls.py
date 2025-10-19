import os
import pandas as pd
import numpy as np

from test_utils import model_infer, get_lung_center


def run_cls(gpu_id, 
        vision_encoder_dir, 
        text_encoder_dir, 
        image_dir,
        center_csv,
        output_path):
    
    device = 'cuda:{}'.format(gpu_id)
    os.makedirs(output_path, exist_ok=True)
    image_to_center = get_lung_center(center_csv)
    image_list = list(sorted(os.listdir(image_dir)))
    
    # pathologies = ['肺部阴影', '肺结节或肺部肿块', '肺部钙化', '透亮影', '肺部实变', '肺炎', '肺结核', '间质改变', '肺不张,肺压缩', '肺水肿', '肺气肿',
    #                    '纵隔肿块', '纵隔钙化,淋巴结钙化', '纵隔淋巴结肿大', '气管扩张或增厚', '气管憩室', '气管炎或支气管炎',
    #                    '心包积液', '心包增厚', '心脏增大', '冠脉钙化或主动脉钙化', '肺动脉增粗', '肺动脉高压', '主动脉增粗,主动脉扩张',
    #                    '胸腔积液', '气胸,无肺纹理区,复张不全', '胸膜钙化', '胸膜增厚', '胸膜胸壁结节', '食管裂孔疝', '食管增厚',
    #                    '骨折', '骨质破坏', '骨肿瘤,骨转移', '术后', '器械植入']
    # text_list = [[f"存在{pathology}.", f"不存在{pathology}."] for pathology in pathologies]
    
    cls_label_name = {"pulmonary nodules": "肺结节肿块", 
                    "pneumothorax": "气胸", 
                    "pleural effusion": "胸腔积液", 
                    "cardiomegaly": "心脏增大"}
    
    res_cls_prob = []
    
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
            image_path, crop_center, [], device, len(cls_label_threshold))
            
        res_cls_prob.append([image_name]+cls_logits)
    
    res_cls_prob = pd.DataFrame(res_cls_prob, columns=["image_name"]+list(cls_label_threshold.keys()))
    res_cls_prob.to_csv(os.path.join(output_path, "result_cls_prob.csv"), index=False, encoding="utf-8-sig")
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='infer clip')
    parser.add_argument('--gpu_id', default=0)
    parser.add_argument('--vision_encoder_dir', default="/path/to/vision_encoder")
    parser.add_argument('--text_encoder_dir', default='/path/to/text_encoder')
    parser.add_argument('--image_dir', default='/path/to/images')
    parser.add_argument('--center_csv', default='/path/to/center_csv')
    parser.add_argument('--output_path', default="/path/to/output")
    args = parser.parse_args()
    
    run_cls(
        args.gpu_id, 
        args.vision_encoder_dir, 
        args.text_encoder_dir,
        args.image_dir,
        args.center_csv,
        args.output_path
        )