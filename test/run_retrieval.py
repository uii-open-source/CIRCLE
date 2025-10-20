import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# Import custom utility functions for model inference, lung center extraction, text encoding, and model loading.
from test_utils import model_infer, get_lung_center, encode_text, load_model


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


def get_top_k_idx(image_data, text_data, text_name_list, k, ):
    """
    Retrieve top-K indices based on cosine similarity between image and text features.

    Args:
        image_data (numpy.ndarray): Image feature vectors.
        text_data (numpy.ndarray): Text feature vectors.
        text_name_list (list): List of names corresponding to each text feature vector.
        k (int): Number of top matches to retrieve.

    Returns:
        list: A list of lists containing the names of the top-K matches for each image.
    """
    top_k_idx_list = []
    for i in range(image_data.shape[0]):
        cos_sim = np.dot(text_data, image_data[i])
        sorted_indices = np.argsort(cos_sim)[::-1]
        top_k_indices = sorted_indices[:k]
        cur_list = []
        for idx in top_k_indices:
            cur_list.append(text_name_list[idx])
        top_k_idx_list.append(cur_list)
    return top_k_idx_list


def run_retrieval(
        gpu_id,
        vision_encoder_dir,
        text_encoder_dir,
        image_dir,
        center_csv,
        report_csv,
        retrieval_mode,
        recall_num,
        output_path):
    """
    Execute retrieval task using pre-trained models for images and texts.

    Args:
        gpu_id (int): ID of GPU to use.
        vision_encoder_dir (str): Directory containing the vision encoder model.
        text_encoder_dir (str): Directory containing the text encoder model.
        image_dir (str): Directory containing images.
        center_csv (str): Path to CSV file containing lung center coordinates for images.
        report_csv (str): Path to CSV file containing reports associated with images.
        retrieval_mode (str): Mode of retrieval ("image_to_report" or "report_to_image").
        recall_num (int): Number of top results to recall.
        output_path (str): Directory to save the output results.
    """
    # Specify the device (GPU) to be used for computations
    device = 'cuda:{}'.format(gpu_id)
    # Ensure that the output directory exists; create it if necessary
    os.makedirs(output_path, exist_ok=True)
    # Get lung centers for images from the provided CSV file
    image_to_center = get_lung_center(center_csv)
    image_list = list(sorted(os.listdir(image_dir)))

    # Load the pre-trained vision and text models
    circle_model, tokenizer = load_model(vision_encoder_dir, text_encoder_dir)

    # Encode the images into feature vectors in the dataset
    name_to_imge_feature = {}
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
        name_to_imge_feature[image_name] = feature

    # Map image names to their associated reports
    name_to_report = {}
    report_table = pd.read_csv(report_csv, dtype={'image_name': str})

    for i in report_table.index:
        name_to_report[report_table.loc[i, "image_name"]] = report_table.loc[i, "finding"] + report_table.loc[
            i, "impression"]

    # Encode the textual reports into feature vectors
    name_to_report_feature = encode_text(vision_encoder_dir, text_encoder_dir, name_to_report, device)

    # Determine query and retrieval datasets based on the retrieval mode
    query_dataset = name_to_report_feature
    retrieval_dataset = name_to_imge_feature
    if retrieval_mode == "image_to_report":
        query_dataset = name_to_imge_feature
        retrieval_dataset = name_to_report_feature
        print(f"retrieval_mode: image_to_report")
    else:
        print(f"retrieval_mode: report_to_image")

    # Prepare query and retrieval data
    query_data = []
    query_name_list = []
    for name in query_dataset:
        query_data.append(query_dataset[name])
        query_name_list.append(name)

    retrieval_data = []
    retrieval_name_list = []
    for name in retrieval_dataset:
        retrieval_data.append(retrieval_dataset[name])
        retrieval_name_list.append(name)

    # Normalize the features
    query_data = np.vstack(query_data)
    query_data = normalize_features(query_data)
    retrieval_data = np.vstack(retrieval_data)
    retrieval_data = normalize_features(retrieval_data)

    # Perform retrieval and get top-K results
    retrieval_res = get_top_k_idx(query_data, retrieval_data, retrieval_name_list, recall_num)
    retrieval_res = pd.DataFrame(retrieval_res, columns=[f"recall@{idx}" for idx in range(1, recall_num + 1)])
    retrieval_res["image_name"] = query_name_list
    retrieval_res.to_csv(os.path.join(output_path, f"result_retrieval_@{recall_num}.csv"), index=False,
                         encoding="utf-8-sig")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='infer retrieval')
    parser.add_argument('--gpu_id', default=0)
    parser.add_argument('--vision_encoder_dir',
                        default="/mnt/maui/Med_VLM/project/94_paper/20251020/model/visual_transformer.bin")
    parser.add_argument('--text_encoder_dir',
                        default='/mnt/maui/Med_VLM/project/94_paper/20251020/model/nlp_roberta_backbone_base_std')
    parser.add_argument('--image_dir', default="/mnt/maui/Med_VLM/project/94_paper/CIRCLE_ZS2K/image/")
    parser.add_argument('--center_csv', default='/mnt/maui/Med_VLM/project/94_paper/CIRCLE_ZS2K/label/lung_center.csv')
    parser.add_argument('--report_csv', default="/mnt/maui/Med_VLM/project/94_paper/CIRCLE_ZS2K/report/report.csv")
    parser.add_argument('--retrieval_mode', default="image_to_report")
    parser.add_argument('--recall_num', default=50)
    parser.add_argument('--output_path', default="/mnt/maui/Med_VLM/project/94_paper/20251020/output_retrieval")
    args = parser.parse_args()

    # Execute the main function with parsed arguments
    run_retrieval(
        args.gpu_id,
        args.vision_encoder_dir,
        args.text_encoder_dir,
        args.image_dir,
        args.center_csv,
        args.report_csv,
        args.retrieval_mode,
        args.recall_num,
        args.output_path)
