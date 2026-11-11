import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer, AutoConfig, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
import os
from tqdm import tqdm
import random
import chardet
import logging

# Set logging level to reduce unnecessary warnings
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.CRITICAL)

# Set random seed for reproducibility
RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

def detect_encoding(file_path):
    """Detect file encoding"""
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)
    result = chardet.detect(raw_data)
    return result['encoding']

def load_csv_with_encoding(file_path, encoding=None):
    """Try multiple encodings to load CSV file"""
    if encoding:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            pass
    
    encodings_to_try = ['utf-8-sig', 'gbk', 'gb2312', 'gb18030', 'big5', 'cp936', 'latin1', 'iso-8859-1']
    
    for enc in encodings_to_try:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except UnicodeDecodeError:
            continue
    
    detected_encoding = detect_encoding(file_path)
    if detected_encoding:
        try:
            return pd.read_csv(file_path, encoding=detected_encoding)
        except:
            pass
    
    raise UnicodeDecodeError(f"Cannot read file: {file_path}")

class TextDataset(Dataset):
    """Text dataset class"""
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class FeatureExtractionDataset(Dataset):
    """Dataset class for feature extraction"""
    def __init__(self, texts, image_names, tokenizer, max_length=512):
        self.texts = texts
        self.image_names = image_names
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        image_name = self.image_names[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'image_name': image_name
        }

class MedBERTForClassification(nn.Module):
    """MedBERT model for classification"""
    def __init__(self, model_path, num_classes=2, dropout_rate=0.1):
        super(MedBERTForClassification, self).__init__()
        
        self.config = AutoConfig.from_pretrained(model_path)
        
        # Silently load model, ignoring mismatched parameters
        print("Loading MedBERT model...")
        
        try:
            self.bert = AutoModel.from_pretrained(
                model_path, 
                config=self.config,
                ignore_mismatched_sizes=True
            )
        except:
            self.bert = AutoModel.from_pretrained(model_path, config=self.config)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask, return_features=False):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        if return_features:
            return pooled_output
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits

def train_epoch(model, data_loader, optimizer, scheduler, device, n_examples):
    """Train one epoch"""
    model.train()
    losses = []
    correct_predictions = 0
    
    progress_bar = tqdm(data_loader, desc="Training")
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(outputs, labels)
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        losses.append(loss.item())
        
        _, preds = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(preds == labels)
        
        progress_bar.set_postfix({'loss': np.mean(losses)})
    
    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, device, n_examples):
    """Evaluate model"""
    model.eval()
    losses = []
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(outputs, labels)
            
            losses.append(loss.item())
            _, preds = torch.max(outputs, dim=1)
            
            probs = torch.softmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    auc = 0.5
    if len(set(all_labels)) > 1:
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.5
    
    return np.mean(losses), auc, all_preds, all_labels, all_probs

def extract_features(model, data_loader, device):
    """Extract 768-dimensional BERT features from model"""
    model.eval()
    all_features = []
    all_image_names = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Extracting features"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            image_names = batch['image_name']
            
            features = model(input_ids=input_ids, attention_mask=attention_mask, return_features=True)
            
            all_features.append(features.cpu().numpy())
            all_image_names.extend(image_names)
    
    all_features = np.vstack(all_features)
    
    return all_features, all_image_names

def load_and_prepare_data(text_csv_path, labels_csv_path):
    """Load and prepare data"""
    print("Loading data...")
    
    try:
        text_df = load_csv_with_encoding(text_csv_path)
    except Exception as e:
        print(f"Cannot load text data: {e}")
        try:
            text_df = pd.read_csv(text_csv_path)
        except Exception as e2:
            raise Exception(f"Failed to load text data: {e2}")
    
    print(f"Text data shape: {text_df.shape}")
    
    try:
        labels_df = load_csv_with_encoding(labels_csv_path)
    except Exception as e:
        print(f"Cannot load label data: {e}")
        try:
            labels_df = pd.read_csv(labels_csv_path)
        except Exception as e2:
            raise Exception(f"Failed to load label data: {e2}")
    
    print(f"Label data shape: {labels_df.shape}")
    
    if 'file_name' in labels_df.columns:
        labels_df = labels_df.rename(columns={'file_name': 'image_name'})
    
    if 'labels' in labels_df.columns:
        labels_df = labels_df.rename(columns={'labels': 'label'})
    
    merged_df = pd.merge(text_df, labels_df, on='image_name', how='inner')
    print(f"Merged data shape: {merged_df.shape}")
    
    label_counts = merged_df['label'].value_counts()
    print(f"Label distribution: {label_counts.to_dict()}")
    
    return merged_df

def main():
    # File paths - updated with new file names
    model_path = r"C:\Users\hi\data\model\medbert_base_wwm"
    text_csv_path = r"C:\Users\hi\data\text_MCOP.csv"  # Updated file name
    labels_csv_path = r"data\labels_MCOP.csv"  # Updated file name
    
    if not os.path.exists(model_path):
        print(f"Error: Model path does not exist: {model_path}")
        return
    
    if not os.path.exists(text_csv_path):
        print(f"Error: Text data file does not exist: {text_csv_path}")
        return
    
    if not os.path.exists(labels_csv_path):
        print(f"Error: Label file does not exist: {labels_csv_path}")
        return
    
    try:
        merged_df = load_and_prepare_data(text_csv_path, labels_csv_path)
    except Exception as e:
        print(f"Failed to load data: {e}")
        return
    
    unique_labels = set(merged_df['label'])
    if len(unique_labels) < 2:
        print(f"Error: Only {len(unique_labels)} class(es)")
        return
    
    print(f"Total samples: {len(merged_df)}")
    
    # Split into training and validation sets
    train_df, val_df = train_test_split(
        merged_df, 
        test_size=0.2, 
        stratify=merged_df['label'],
        random_state=RANDOM_STATE
    )
    
    # Training data
    X_train = train_df['text'].tolist()
    y_train = train_df['label'].tolist()
    train_image_names = train_df['image_name'].tolist()
    
    # Validation data
    X_val = val_df['text'].tolist()
    y_val = val_df['label'].tolist()
    val_image_names = val_df['image_name'].tolist()
    
    # All data (for feature extraction)
    all_texts = merged_df['text'].tolist()
    all_image_names = merged_df['image_name'].tolist()
    
    print(f"Training set: {len(X_train)}")
    print(f"Validation set: {len(X_val)}")
    
    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        vocab_file = os.path.join(model_path, "vocab.txt")
        if os.path.exists(vocab_file):
            from transformers import BertTokenizer
            tokenizer = BertTokenizer(vocab_file=vocab_file)
        else:
            print("Cannot find vocab.txt file")
            return
    
    print("Creating datasets...")
    # Training dataset
    train_dataset = TextDataset(X_train, y_train, tokenizer, max_length=512)
    # Validation dataset
    val_dataset = TextDataset(X_val, y_val, tokenizer, max_length=512)
    # Dataset for all samples feature extraction
    all_feature_dataset = FeatureExtractionDataset(all_texts, all_image_names, tokenizer, max_length=512)
    
    batch_size = 8
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size)
    all_feature_data_loader = DataLoader(all_feature_dataset, batch_size=batch_size, shuffle=False)
    
    print("Creating model...")
    model = MedBERTForClassification(model_path, num_classes=len(unique_labels))
    model = model.to(device)
    
    epochs = 5
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_data_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0.1 * total_steps,
        num_training_steps=total_steps
    )
    
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    best_auc = 0.5
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 40)
        
        train_acc, train_loss = train_epoch(
            model, train_data_loader, optimizer, scheduler, device, len(X_train)
        )
        
        val_loss, val_auc, val_preds, val_labels, val_probs = eval_model(
            model, val_data_loader, device, len(X_val)
        )
        
        print(f"Train loss: {train_loss:.4f}")
        print(f"Val loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
        
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), 'best_medbert_model.bin')
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    print(f"Best AUC: {best_auc:.4f}")
    
    print("\nLoading best model for final evaluation...")
    if os.path.exists('best_medbert_model.bin'):
        model.load_state_dict(torch.load('best_medbert_model.bin'))
    
    val_loss, val_auc, val_preds, val_labels, val_probs = eval_model(
        model, val_data_loader, device, len(X_val)
    )
    
    print(f"Final - Val loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
    
    # Extract features for ALL samples
    print("\n" + "="*60)
    print("Extracting BERT features for ALL samples...")
    print("="*60)
    
    all_features, all_image_names = extract_features(model, all_feature_data_loader, device)
    
    print(f"Extracted feature shape: {all_features.shape}")
    print(f"Total samples processed: {len(all_image_names)}")
    
    feature_columns = [f'BERT_feature_{i}' for i in range(all_features.shape[1])]
    
    features_df = pd.DataFrame(all_features, columns=feature_columns)
    features_df.insert(0, 'image_name', all_image_names)
    
    output_csv_path = "all_samples_bert_features.csv"
    features_df.to_csv(output_csv_path, index=False)
    
    print(f"All samples features saved to: {output_csv_path}")
    
    # Save the fine-tuned model
    output_dir = "fine_tuned_medbert"
    os.makedirs(output_dir, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    model.config.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"Model saved to: {output_dir}")
    
    # Save training summary
    summary = {
        'best_auc': best_auc,
        'final_val_loss': val_loss,
        'final_val_auc': val_auc,
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
        'total_samples': len(all_image_names),
        'epochs_trained': epochs,
        'bert_features_extracted': all_features.shape[1],
        'all_samples_feature_file': output_csv_path,
        'model_saved_to': output_dir
    }
    
    summary_csv_path = "training_summary.csv"
    pd.DataFrame(list(summary.items()), columns=['Metric', 'Value']).to_csv(summary_csv_path, index=False)
    
    # Save validation set predictions
    predictions_df = pd.DataFrame({
        'image_name': val_image_names,
        'true_label': val_labels,
        'predicted_label': val_preds,
        'prediction_probability': val_probs
    })
    
    predictions_csv_path = "validation_set_predictions.csv"
    predictions_df.to_csv(predictions_csv_path, index=False)
    
    print("\n" + "="*60)
    print("All tasks completed!")
    print("="*60)
    print("Output files:")
    print(f"1. Fine-tuned model: {output_dir}/")
    print(f"2. All samples BERT features: {output_csv_path}")
    print(f"3. Training summary: {summary_csv_path}")
    print(f"4. Validation set predictions: {predictions_csv_path}")

if __name__ == "__main__":
    main()
