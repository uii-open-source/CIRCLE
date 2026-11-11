import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import os

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

class MCOP_Predictor:
    def __init__(self, circle_features_path, bert_features_path, labels_path):
        # Initialize file paths
        self.circle_features_path = circle_features_path
        self.bert_features_path = bert_features_path
        self.labels_path = labels_path
        
        # XGBoost model parameters as per methodology
        self.single_modal_params = {
            'n_estimators': 300,           # Number of trees for single modality
            'max_depth': 6,                # Maximum tree depth
            'learning_rate': 0.05,         # Step size shrinkage
            'subsample': 0.8,              # Subsample ratio of training instances
            'colsample_bytree': 0.8,       # Subsample ratio of columns
            'random_state': RANDOM_STATE,  # Random seed for reproducibility
            'eval_metric': 'logloss',      # Evaluation metric
            'use_label_encoder': False     # Avoid label encoder warning
        }
        
        self.hybrid_params = {
            'n_estimators': 500,           # More trees for hybrid model
            'max_depth': 8,                # Deeper trees for hybrid model
            'learning_rate': 0.05,         # Step size shrinkage
            'subsample': 0.8,              # Subsample ratio of training instances
            'colsample_bytree': 0.8,       # Subsample ratio of columns
            'random_state': RANDOM_STATE,  # Random seed for reproducibility
            'eval_metric': 'logloss',      # Evaluation metric
            'use_label_encoder': False     # Avoid label encoder warning
        }
        
        # Storage for data and models
        self.image_model = None
        self.text_model = None
        self.hybrid_model = None
        self.scaler_image = StandardScaler()
        self.scaler_text = StandardScaler()
        self.scaler_hybrid = StandardScaler()
        
    def load_labels(self):
        """Load label data"""
        labels_df = pd.read_csv(self.labels_path)
        
        # Standardize column names
        if 'file_name' in labels_df.columns:
            labels_df = labels_df.rename(columns={'file_name': 'image_name'})
        elif 'image_name' not in labels_df.columns:
            labels_df = labels_df.rename(columns={labels_df.columns[0]: 'image_name'})
        
        if 'labels' in labels_df.columns:
            labels_df = labels_df.rename(columns={'labels': 'label'})
        elif 'label' not in labels_df.columns:
            labels_df = labels_df.rename(columns={labels_df.columns[1]: 'label'})
        
        return labels_df
    
    def load_image_features(self):
        """Load CIRCLE-based CT image features"""
        circle_df = pd.read_csv(self.circle_features_path)
        
        # Standardize column names
        if 'file_name' in circle_df.columns:
            circle_df = circle_df.rename(columns={'file_name': 'image_name'})
        elif 'image_name' not in circle_df.columns:
            circle_df = circle_df.rename(columns={circle_df.columns[0]: 'image_name'})
        
        return circle_df
    
    def load_text_features(self):
        """Load BERT text features extracted from clinical text"""
        bert_df = pd.read_csv(self.bert_features_path)
        
        # Standardize column names
        if 'image_name' not in bert_df.columns:
            bert_df = bert_df.rename(columns={bert_df.columns[0]: 'image_name'})
        
        return bert_df
    
    def merge_and_split_data(self):
        """Load and merge all features, then split into train/test sets"""
        # Load labels
        labels_df = self.load_labels()
        
        # Load image features
        image_df = self.load_image_features()
        
        # Load text features
        text_df = self.load_text_features()
        
        # Merge all data
        merged_df = pd.merge(labels_df, image_df, on='image_name', how='inner')
        merged_df = pd.merge(merged_df, text_df, on='image_name', how='inner')
        
        # Random split 80/20
        train_df, test_df = train_test_split(
            merged_df, 
            test_size=0.2, 
            stratify=merged_df['label'],
            random_state=RANDOM_STATE
        )
        
        # Prepare data for different modalities
        data_dict = {}
        
        # Image-Only data
        image_features = [f for f in merged_df.columns if f.startswith('feature_')]
        data_dict['image'] = {
            'train': train_df[['image_name', 'label'] + image_features],
            'test': test_df[['image_name', 'label'] + image_features]
        }
        
        # Text-Only data
        text_features = [f for f in merged_df.columns if f.startswith('BERT_feature_')]
        data_dict['text'] = {
            'train': train_df[['image_name', 'label'] + text_features],
            'test': test_df[['image_name', 'label'] + text_features]
        }
        
        # Hybrid data (concatenation of both)
        hybrid_features = image_features + text_features
        data_dict['hybrid'] = {
            'train': train_df[['image_name', 'label'] + hybrid_features],
            'test': test_df[['image_name', 'label'] + hybrid_features]
        }
        
        return data_dict
    
    def preprocess_data(self, X_train, X_test, fit_scaler=True, scaler=None):
        """Apply Z-score normalization to features"""
        if fit_scaler:
            # Fit scaler on training data
            features_scaled_train = scaler.fit_transform(X_train)
            features_scaled_test = scaler.transform(X_test)
        else:
            # Transform test data using fitted scaler
            features_scaled_train = scaler.transform(X_train)
            features_scaled_test = scaler.transform(X_test)
        
        return features_scaled_train, features_scaled_test
    
    def train_model(self, X_train, y_train, model_type='image'):
        """Train XGBoost classifier with appropriate parameters"""
        if model_type == 'hybrid':
            params = self.hybrid_params
        else:
            params = self.single_modal_params
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        return model
    
    def predict_proba(self, model, X):
        """Predict probability of positive class"""
        return model.predict_proba(X)[:, 1]
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance - only AUC"""
        y_pred_proba = self.predict_proba(model, X_test)
        auc = roc_auc_score(y_test, y_pred_proba)
        return auc
    
    def calculate_contribution_ratio(self, model, feature_names, modality='hybrid'):
        """Calculate relative contribution of image vs text features in hybrid model"""
        if modality != 'hybrid':
            return None, None
        
        # Get feature importance scores (gain-based)
        importance_scores = model.get_booster().get_score(importance_type='gain')
        
        # Create DataFrame for importance
        importance_df = pd.DataFrame({
            'feature': list(importance_scores.keys()),
            'importance': list(importance_scores.values())
        })
        
        # Map XGBoost's internal feature names (f0, f1, ...) to actual feature names
        feature_mapping = {}
        for i, feature_name in enumerate(feature_names):
            xgb_feature_name = f'f{i}'
            feature_mapping[xgb_feature_name] = feature_name
        
        importance_df['feature_name'] = importance_df['feature'].map(
            lambda x: feature_mapping.get(x, x)
        )
        
        # Calculate total importance for image and text features
        image_importance = 0
        text_importance = 0
        
        for feature in importance_df['feature_name']:
            if isinstance(feature, str):
                if feature.startswith('feature_'):
                    # Image feature
                    image_importance += importance_df[
                        importance_df['feature_name'] == feature
                    ]['importance'].sum()
                elif feature.startswith('BERT_feature_'):
                    # Text feature
                    text_importance += importance_df[
                        importance_df['feature_name'] == feature
                    ]['importance'].sum()
        
        # Calculate relative contribution percentages
        total_importance = image_importance + text_importance
        
        if total_importance > 0:
            image_ratio = image_importance / total_importance * 100
            text_ratio = text_importance / total_importance * 100
        else:
            image_ratio = 0
            text_ratio = 0
        
        return image_ratio, text_ratio
    
    def run_pipeline(self):
        """Execute complete training and evaluation pipeline for all three modalities"""
        print("CIRCLE-based Multi-modality Clinical Outcome Prediction (MCOP)")
        
        # Load and split data
        data_dict = self.merge_and_split_data()
        
        results = {}
        
        # Train and evaluate each modality
        for modality in ['image', 'text', 'hybrid']:
            # Extract data
            train_data = data_dict[modality]['train']
            test_data = data_dict[modality]['test']
            
            # Separate features and labels
            feature_cols = [col for col in train_data.columns 
                          if col not in ['image_name', 'label']]
            
            X_train = train_data[feature_cols].values
            y_train = train_data['label'].values
            X_test = test_data[feature_cols].values
            y_test = test_data['label'].values
            
            # Preprocess data
            if modality == 'image':
                scaler = self.scaler_image
            elif modality == 'text':
                scaler = self.scaler_text
            else:  # hybrid
                scaler = self.scaler_hybrid
            
            X_train_scaled, X_test_scaled = self.preprocess_data(
                X_train, X_test, fit_scaler=True, scaler=scaler
            )
            
            # Train model
            model = self.train_model(X_train_scaled, y_train, model_type=modality)
            
            # Store model
            if modality == 'image':
                self.image_model = model
            elif modality == 'text':
                self.text_model = model
            else:
                self.hybrid_model = model
            
            # Evaluate model (only AUC)
            auc = self.evaluate_model(model, X_test_scaled, y_test)
            
            # Calculate contribution ratio for hybrid model
            if modality == 'hybrid':
                image_ratio, text_ratio = self.calculate_contribution_ratio(
                    model, feature_cols, modality=modality
                )
                results['image_contribution'] = image_ratio
                results['text_contribution'] = text_ratio
            
            # Store AUC result
            if modality == 'image':
                results['image_auc'] = auc
            elif modality == 'text':
                results['text_auc'] = auc
            else:
                results['hybrid_auc'] = auc
        
        # Output results
        print(f"\nImage-Only Model AUC: {results['image_auc']:.4f}")
        print(f"Text-Only Model AUC: {results['text_auc']:.4f}")
        print(f"Hybrid Model AUC: {results['hybrid_auc']:.4f}")
        
        if 'image_contribution' in results and 'text_contribution' in results:
            print(f"Contribution - Image features: {results['image_contribution']:.1f}%, Text features: {results['text_contribution']:.1f}%")
        
        return results


def main():
    # File paths
    circle_features_path = "data/circle_features_MCOP.csv"
    bert_features_path = "all_samples_bert_features.csv"  # Output from BERT training
    labels_path = "data/labels_MCOP.csv"
    
    # Check if required files exist
    for path in [circle_features_path, bert_features_path, labels_path]:
        if not os.path.exists(path):
            print(f"Error: File not found - {path}")
            exit(1)
    
    # Create predictor instance
    predictor = MCOP_Predictor(
        circle_features_path=circle_features_path,
        bert_features_path=bert_features_path,
        labels_path=labels_path
    )
    
    try:
        # Run complete pipeline
        results = predictor.run_pipeline()
        
    except Exception as e:
        print(f"\nError during pipeline execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
