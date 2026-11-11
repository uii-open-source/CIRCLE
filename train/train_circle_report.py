import json
import os
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM

from model.circle_report import CIRCLEReport
from train.trainer import CIRCLETrainer


# Define the directory path for the language model cache
llm_cache_dir = "/mnt/maui/Med_VLM/project/models/Qwen3/Qwen3-8B"

# Load the configuration JSON file to get the hidden size of the language model
with open(os.path.join(llm_cache_dir, "config.json"), 'r') as file:
    data = json.load(file)
llm_hidden_size = data["hidden_size"]

# Initialize the tokenizer and model using the pretrained Qwen model
gptTokenizer = AutoTokenizer.from_pretrained(llm_cache_dir, use_fast=False, trust_remote_code=True)
gptModel = AutoModelForCausalLM.from_pretrained(llm_cache_dir, attn_implementation="flash_attention_2",
                                                torch_dtype=torch.bfloat16)

# Enable gradient checkpointing to reduce memory usage during training
gptModel.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

# Configure LoRA (Low-Rank Adaptation) parameters for efficient fine-tuning
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # Specify the task type as causal language modeling
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # Modules to apply LoRA to
    inference_mode=False,  # Set to training mode
    r=128,  # Rank parameter for LoRA
    lora_alpha=256,  # Scaling factor for LoRA
    lora_dropout=0.05  # Dropout rate for LoRA layers
)

# Enable input gradients to allow fine-tuning
gptModel.enable_input_require_grads()
# Apply LoRA configuration to the model
gptModel = get_peft_model(gptModel, lora_config)
# Print the number of trainable parameters in the model
gptModel.print_trainable_parameters()

# Initialize the CIRCLE report model with specified parameters
circle_report_model = CIRCLEReport(
    llm_hidden_size=llm_hidden_size,  # Hidden size from the loaded config
    gpt_tokenizer=gptTokenizer,  # Tokenizer instance
    gpt_model=gptModel,  # Language model with LoRA applied
    train_gpt=True,  # Flag to indicate whether to train the GPT component
)

# Path to the pretrained vision encoder weights
pretrained_vision_encoder_path = "/model/vision_encoder.bin"
# Load the vision encoder weights into the model
circle_report_model.load(vision_path=pretrained_vision_encoder_path, load_clip_weight=True)

# Initialize the trainer with model and training parameters
trainer = CIRCLETrainer(
    circle_report_model,  # The model to be trained
    train_gpt=True,  # Flag to enable training of the GPT component
    data_folder='/path/to/image',  # folder containing all patient CT data
    label_csv='/path/to/label/label.csv',  # CSV file mapping image names to labels
    lung_center_csv='/path/to/label/lung_center.csv',  # CSV file with lung center coordinates for cropping
    report_csv='/path/to/report/report.csv',  # CSV file containing report text for each image
    lr=2e-5,  # Learning rate for training
    max_grad_norm=1.0,  # Maximum gradient norm for clipping
    batch_size=3,  # Number of samples per training batch
    num_train_steps=200001,  # Total number of training steps
    num_workers=6,  # Number of worker processes for data loading
    save_results_every=1500,  # Frequency of saving intermediate results
    save_model_every=10,  # Frequency of saving model checkpoints
    results_folder="output/train_report",  # Directory to save training results
)

# Start the training process
trainer.train()
