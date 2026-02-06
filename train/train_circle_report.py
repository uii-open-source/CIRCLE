import json
import os
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM

from model.circle_report import CIRCLEReport
from train.trainer import CIRCLETrainer


llm_cache_dir = "/mnt/maui/Med_VLM/project/models/Qwen3/Qwen3-8B"
with open(os.path.join(llm_cache_dir, "config.json"), 'r') as file:
    data = json.load(file)
llm_hidden_size = data["hidden_size"]

gptTokenizer = AutoTokenizer.from_pretrained(llm_cache_dir, use_fast=False, trust_remote_code=True)
gptModel = AutoModelForCausalLM.from_pretrained(llm_cache_dir, attn_implementation="flash_attention_2",
                                                torch_dtype=torch.bfloat16)
gptModel.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=128,
    lora_alpha=256,
    lora_dropout=0.05
)
gptModel.enable_input_require_grads()
gptModel = get_peft_model(gptModel, lora_config)
gptModel.print_trainable_parameters()

circle_report_model = CIRCLEReport(
    llm_hidden_size=llm_hidden_size,
    gptTokenizer=gptTokenizer,
    gptModel=gptModel,
    train_gpt=True,
)

pretrained_vision_encoder_path = "/mnt/maui/Med_VLM/project/94_paper/CIRCLE_ZS2K/20251020/model/vision_encoder.bin"
circle_report_model.load(vision_path=pretrained_vision_encoder_path, load_clip_weight=True)

trainer = CIRCLETrainer(
    circle_report_model,
    train_gpt=True,
    data_folder='/mnt/maui/Med_VLM/project/94_paper/CIRCLE_ZS2K/image',  # folder containing all patient CT data
    label_csv="/mnt/maui/Med_VLM/project/94_paper/CIRCLE_ZS2K/20251020/label/CIRCLE_chest37.csv",
    lung_center_csv="/mnt/maui/Med_VLM/project/94_paper/CIRCLE_ZS2K/20251020/label/lung_center.csv",
    report_csv="/mnt/maui/Med_VLM/project/94_paper/CIRCLE_ZS2K/20251020/report/report.csv",
    lr=2e-5,
    max_grad_norm=1.0,
    batch_size=3,
    num_train_steps=200001,
    num_workers=6,
    save_results_every=1500,
    save_model_every=10,
    results_folder="/mnt/maui/Med_VLM/project/94_paper/code/train",
)

trainer.train()
