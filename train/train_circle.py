from transformers import BertTokenizer, BertModel
# Import HuggingFace Transformers modules:
# BertTokenizer: used to convert text into token IDs suitable for BERT
# BertModel: the pre-trained BERT model for text embeddings

from model.circle import CIRCLE
# Import CIRCLE model class, which implements multi-modal (CT + text) contrastive learning

from train.trainer import CIRCLETrainer
# Import trainer class that handles the training loop, data loading, loss computation, and saving

# Initialize the BERT tokenizer from a pre-trained model path
# Tokenizer is used to convert report text (description + conclusion) into token IDs
tokenizer = BertTokenizer.from_pretrained('/path/to/circle/text_model/nlp_roberta_backbone_base_std')

# Initialize the BERT model from the same pre-trained path
# This model outputs contextualized embeddings for input tokens
text_encoder = BertModel.from_pretrained("/path/to/circle/text_model/nlp_roberta_backbone_base_std")


# Initialize the CIRCLE model
circle_model = CIRCLE(
    text_encoder=text_encoder,  # provide the text encoder to the multi-modal model
    dim_image=1792,             # dimension of the image latent features
    dim_text=768,               # dimension of the text latent features (BERT base output)
    dim_latent=512,             # dimension of the projected latent space for contrastive learning
)

# Initialize the trainer that will handle data loading, forward pass, loss computation, backprop, and checkpointing
trainer = CIRCLETrainer(
    circle_model,                              # the CIRCLE model instance
    tokenizer,                                 # tokenizer for processing text inputs
    data_folder='/path/to/image',                      # folder containing all patient CT data
    label_csv='/path/to/label/label.csv',                    # CSV file mapping image names to labels
    lung_center_csv='/path/to/label/lung_center.csv',        # CSV file with lung center coordinates for cropping
    report_csv='/path/to/report/report.csv',                  # CSV file containing report text for each image
    num_train_steps=200001,                    # total number of training steps
    batch_size=5,                              # batch size for training
    results_folder="/path/to/results",                 # folder to save training outputs and checkpoints
    num_workers=6,                             # number of data loader workers for parallel loading
    save_results_every=1000,                   # frequency (in steps) to save intermediate results
    save_model_every=1000                      # frequency (in steps) to save model checkpoints
)

# Start the training loop
# The trainer handles:
# 1) Loading batches from dataset
# 2) Forward pass through CIRCLE model (image + text)
# 3) Computing classification and contrastive losses
# 4) Backpropagation and optimizer step
# 5) Saving intermediate results and checkpoints
trainer.train()
