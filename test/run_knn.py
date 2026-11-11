import os
import pandas as pd
import numpy as np
import random
from tqdm import tqdm

# Import custom utility functions for model inference, lung center extraction, and model loading.
from test_utils import model_infer, get_lung_center, load_model


def normalize_features(features):
    """
    Normalize feature vectors to unit norm along the first axis.

    Args:
        features (numpy.ndarray): Array of feature vectors.

    Returns:
        numpy.ndarray: Normalized feature vectors.
    """
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / norms


def get_top_k_prob(image_data, text_data, text_label, k, is_random=False):
    """
    Calculate top-K probabilities based on cosine similarity between image and text features.

    Args:
        image_data (numpy.ndarray): Image feature vectors.
        text_data (numpy.ndarray): Text feature vectors.
        text_label (numpy.ndarray): Corresponding labels for text data.
        k (int): Number of top matches to consider.
        is_random (bool): If True, select random indices instead of sorting by similarity.

    Returns:
        numpy.ndarray: Average probability for each image's top-K matches.
    """
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
        vision_encoder_path,
        text_encoder_dir,
        image_dir,
        center_csv,
        label_csv,
        test_image_dir,
        test_image_cecnter_csv,
        output_path):
    """
    Run KNN classification on medical images using pre-trained vision and text models.

    Args:
        gpu_id (int): ID of GPU to use.
        vision_encoder_path (str): Directory containing the vision encoder model.
        text_encoder_dir (str): Directory containing the text encoder model.
        image_dir (str): Directory containing training images.
        center_csv (str): Path to CSV file containing lung center coordinates for training images.
        label_csv (str): Path to CSV file containing labels for training images.
        test_image_dir (str): Directory containing test images.
        test_image_cecnter_csv (str): Path to CSV file containing lung center coordinates for test images.
        output_path (str): Directory to save the output results.
    """
    # Specify the device (GPU) to be used for computations
    device = 'cuda:{}'.format(gpu_id)
    # Ensure that the output directory exists; create it if necessary
    os.makedirs(output_path, exist_ok=True)

    # Load the pre-trained vision and text models
    circle_model, tokenizer = load_model(vision_encoder_path, text_encoder_dir)

    # Get lung centers for images from the provided CSV file
    image_to_center = get_lung_center(center_csv)
    image_list = list(sorted(os.listdir(image_dir)))
    image_name_to_feature = {}
    # Process each image in the dataset
    for image_name in tqdm(image_list):
        image_path = os.path.join(image_dir, image_name, "CT.nii.gz")
        if not os.path.exists(image_path):
            print(f"{image_path} not existed, continue")
            continue
        if image_name not in image_to_center:
            print(f"{image_name} lung center not existed, continue")
            continue
        crop_center = image_to_center[image_name]
        _, _, feature = model_infer(
            circle_model, tokenizer,
            image_path, crop_center, [], device)
        image_name_to_feature[image_name] = feature

    # Read the label CSV file into a DataFrame
    label_table = pd.read_csv(label_csv, dtype={'image_name': str})
    knn_cls_labels = list(label_table.columns)
    if "image_name" in knn_cls_labels:
        knn_cls_labels.remove("image_name")
    image_name_to_label = {}
    for i in label_table.index:
        cur = []
        for label in knn_cls_labels:
            cur.append(label_table.loc[i, label])
        image_name_to_label[label_table.loc[i, "image_name"]] = cur

    # Repeat the same process for the test dataset
    image_to_center = get_lung_center(test_image_cecnter_csv)
    test_list = list(sorted(os.listdir(test_image_dir)))
    test_feature_list = []
    test_name_list = []
    for image_name in tqdm(test_list):
        image_path = os.path.join(test_image_dir, image_name, "CT.nii.gz")
        if not os.path.exists(image_path):
            print(f"{image_path} not existed, continue")
            continue
        if image_name not in image_to_center:
            print(f"{image_name} lung center not existed, continue")
            continue
        crop_center = image_to_center[image_name]
        _, _, feature = model_infer(
            circle_model, tokenizer,
            image_path, crop_center, [], device)
        test_feature_list.append(feature)
        test_name_list.append(image_name)

    # Prepare the data for KNN classification
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

    # Calculate the probabilities for the test set using the KNN method
    k = 1024
    prob = get_top_k_prob(test_feature, image_data, image_label, k)
    prob_df = pd.DataFrame(prob, columns=knn_cls_labels)
    prob_df["image_name"] = test_name_list
    # Save the results to a CSV file
    prob_df.to_csv(os.path.join(output_path, "result_knn_cls.csv"), index=False, encoding="utf-8-sig")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='infer knn cls')
    parser.add_argument('--gpu_id', default=0)
    parser.add_argument('--vision_encoder_path',
                        default="/mnt/maui/Med_VLM/project/CIRCLE_ZS2K/model/vision_encoder.bin")
    parser.add_argument('--text_encoder_dir',
                        default='/mnt/maui/Med_VLM/project/CIRCLE_ZS2K/model/text_encoder/')
    parser.add_argument('--knn_image_dir', default='/mnt/maui/Med_VLM/project/CIRCLE_ZS2K/image/')
    parser.add_argument('--knn_center_csv',
                        default='/mnt/maui/Med_VLM/project/CIRCLE_ZS2K/label/lung_center.csv')
    parser.add_argument('--knn_label_csv',
                        default="/mnt/maui/Med_VLM/project/CIRCLE_ZS2K/label/CIRCLE_chest37.csv")
    parser.add_argument('--image_dir', default='/mnt/maui/Med_VLM/project/CIRCLE_ZS2K/image/')
    parser.add_argument('--center_csv',
                        default="/mnt/maui/Med_VLM/project/CIRCLE_ZS2K/label/lung_center.csv")
    parser.add_argument('--output_path', default="/mnt/maui/Med_VLM/project/CIRCLE_ZS2K/output")
    args = parser.parse_args()

    # Execute the main function with parsed arguments
    run_knn(
        args.gpu_id,
        args.vision_encoder_path,
        args.text_encoder_dir,
        args.knn_image_dir,
        args.knn_center_csv,
        args.knn_label_csv,
        args.image_dir,
        args.center_csv,
        args.output_path
    )
