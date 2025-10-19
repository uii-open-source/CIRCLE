import os
import pandas as pd
import numpy as np
import random

from test_utils import model_infer, get_lung_center

def normalize_features(features):
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / norms

def get_top_k_prob(image_data, text_data, text_label, k, is_random=False):
    prob_list = []
    for i in range(image_data.shape[0]):
        if is_random:
            sorted_indices = list(range(text_data.shape[0]))
            random.shuffle(sorted_indices)
        else:
            cos_sim = np.dot(text_data, image_data[i])
            sorted_indices = np.argsort(cos_sim)[::-1]
        top_k_indices = sorted_indices[:k]
        top_k_labels = text_label[top_k_indices]
        top_k_prob = np.mean(top_k_labels, axis=0)
        prob_list.append(top_k_prob)
    return np.array(prob_list)

def run_knn(
        gpu_id, 
        vision_encoder_dir, 
        text_encoder_dir,
        image_dir,
        center_csv,
        label_csv,
        test_image_dir,
        test_image_cecnter_csv,
        output_path):
    
    device = 'cuda:{}'.format(gpu_id)
    os.makedirs(output_path, exist_ok=True)
    
    # pathologies = ['肺部阴影', '肺结节或肺部肿块', '肺部钙化', '透亮影', '肺部实变', '肺炎', '肺结核', '间质改变', '肺不张,肺压缩', '肺水肿', '肺气肿',
    #                    '纵隔肿块', '纵隔钙化,淋巴结钙化', '纵隔淋巴结肿大', '气管扩张或增厚', '气管憩室', '气管炎或支气管炎',
    #                    '心包积液', '心包增厚', '心脏增大', '冠脉钙化或主动脉钙化', '肺动脉增粗', '肺动脉高压', '主动脉增粗,主动脉扩张',
    #                    '胸腔积液', '气胸,无肺纹理区,复张不全', '胸膜钙化', '胸膜增厚', '胸膜胸壁结节', '食管裂孔疝', '食管增厚',
    #                    '骨折', '骨质破坏', '骨肿瘤,骨转移', '术后', '器械植入']
    # text_list = [[f"存在{pathology}.", f"不存在{pathology}."] for pathology in pathologies]
    
    # cls_label_threshold = {"肺结节肿块": 0.5, 
    #                        "气胸": 0.5, 
    #                        "胸腔积液": 0.5, 
    #                        "心脏增大": 0.5}
    
    
    image_to_center = get_lung_center(center_csv)
    image_list = list(sorted(os.listdir(image_dir)))
    image_name_to_feature = {}
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
            image_path, crop_center, [], device)
        image_name_to_feature[image_name] = feature
    
    label_table = pd.read_csv(label_csv)
    knn_cls_labels = list(label_table.columns)
    if "image_name" in knn_cls_labels:
        knn_cls_labels.remove("image_name")
    image_name_to_label = {}
    for i in label_table.index:
        cur = []
        for label in knn_cls_labels:
            cur.append(label_table.loc[i, label])
        image_name_to_label[label_table.loc[i, "image_name"]] = cur
    
    
    image_to_center = get_lung_center(test_image_cecnter_csv)
    test_list = list(sorted(os.listdir(test_image_dir)))
    test_name_to_feature = {}
    test_feature_list = []
    test_name_list = []
    for image_name in test_list:
        image_path = os.path.join(test_image_dir, image_name, "CT.nii.gz")
        if not os.path.exists(image_path):
            print(f"{image_path} not existed, continue")
            continue
        if image_name not in image_to_center:
            print(f"{image_name} lung center not existed, continue")
            continue
        crop_center = image_to_center[image_name]
        clip_prob_list, cls_logits, feature = model_infer(
            vision_encoder_dir, text_encoder_dir, 
            image_path, crop_center, text_list, device, len(cls_label_threshold))
        test_feature_list.append(feature)
        test_name_list.append(image_name)
    
    feature_file_list = sorted(list(image_name_to_feature.keys()))
    feature_file_list = [name for name in feature_file_list if name in image_name_to_label]
    random.shuffle(feature_file_list)
    image_label = np.array([image_name_to_label[name] for name in feature_file_list])
    image_label = np.array(image_label)

    image_data = [image_name_to_feature[name] for name in feature_file_list]
    image_data = np.vstack(image_data)
    image_data = normalize_features(image_data)
    
    test_feature = np.vstack(test_feature_list)
    test_feature = normalize_features(test_feature)
    
    k = 1024
    prob = get_top_k_prob(test_feature, image_data, image_label, k)
    prob_df = pd.DataFrame(prob, columns=knn_cls_labels)
    prob_df["image_name"] = test_name_list
    prob_df.to_csv(os.path.join(output_path, "result_knn_cls.csv"), index=False, encoding="utf-8-sig")
    
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='infer clip')
    parser.add_argument('--gpu_id', default=0)
    parser.add_argument('--vision_encoder_dir', default="/path/to/vision_encoder")
    parser.add_argument('--text_encoder_dir', default='/path/to/text_encoder')
    parser.add_argument('--knn_image_dir', default='/path/to/images')
    parser.add_argument('--knn_center_csv', default='/path/to/center_csv')
    parser.add_argument('--knn_label_csv', default='/path/to/label_csv')
    parser.add_argument('--image_dir', default='/path/to/test_image')
    parser.add_argument('--center_csv', default="/path/to/test_center_csv")
    parser.add_argument('--output_path', default="/path/to/output")
    args = parser.parse_args()
    
    run_knn(
        args.gpu_id, 
        args.vision_encoder_dir, 
        args.text_encoder_dir,
        args.knn_image_dir,
        args.knn_center_csv,
        args.knn_label_csv,
        args.image_dir,
        args.center_csv,
        args.output_path
        )