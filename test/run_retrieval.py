import os
import pandas as pd
import numpy as np


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='infer clip')
    parser.add_argument('--gpu_id', default=0)
    parser.add_argument('--vision_encoder_dir', default="/path/to/vision_encoder")
    parser.add_argument('--text_encoder_dir', default='/path/to/text_encoder')
    parser.add_argument('--image_dir', default='/mnt/maui/Med_VLM/project/01_mhd/')
    parser.add_argument('--center_csv', default='/mnt/maui/Med_VLM/project/01_mhd/')
    parser.add_argument('--report_csv', default='/mnt/maui/Med_VLM/project/01_mhd/')
    parser.add_argument('--recall_num', default=5)
    parser.add_argument('--output_path', default="/mnt/maui/ASR_MED/project_cache/czr/test/pipeline_tmp/")
    args = parser.parse_args()
    
    run_retrieval(
        args.feature_dir,
        args.recall_num,
        args.output_path)