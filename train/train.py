from transformers import BertTokenizer, BertModel

from model.circle import CIRCLE
from train.trainer import CIRCLETrainer


tokenizer = BertTokenizer.from_pretrained('/data/circle/models/nlp_roberta_backbone_base_std')
text_encoder = BertModel.from_pretrained("/data/circle/models/nlp_roberta_backbone_base_std")


circle_model = CIRCLE(
    text_encoder=text_encoder,
    dim_image=1792,
    dim_text=768,
    dim_latent=512,
)

trainer = CIRCLETrainer(
    circle_model,
    tokenizer,
    data_folder='/image',
    label_csv='/label.csv',
    lung_center_csv='/lung_center.csv',
    report_csv='/report.csv',
    num_train_steps=200001,
    batch_size=5,
    results_folder="/results",
    num_workers=6,
    save_results_every=1000,
    save_model_every=1000
)

trainer.train()
