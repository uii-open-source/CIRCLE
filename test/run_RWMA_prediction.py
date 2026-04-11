import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import joblib
import os

# Set random seed for reproducibility - ensures same results each time code runs
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

class RWMA_Predictor:
    def __init__(self, circle_features_path, radiomics_features_path, labels_path):
        # Initialize file paths
        self.circle_features_path = circle_features_path
        self.radiomics_features_path = radiomics_features_path
        self.labels_path = labels_path
        
        # XGBoost model parameters as specified in methodology
        self.model_params = {
            'n_estimators': 100,           # Number of trees
            'max_depth': 6,                # Maximum tree depth
            'learning_rate': 0.1,          # Step size shrinkage
            'subsample': 0.8,              # Subsample ratio of training instances
            'colsample_bytree': 0.8,       # Subsample ratio of columns
            'random_state': RANDOM_STATE,  # Random seed for reproducibility
            'eval_metric': 'logloss',      # Evaluation metric
            'use_label_encoder': False     # Avoid label encoder warning
        }
        
        # Storage for data
        self.feature_names = None
        self.model = None
        self.scaler = StandardScaler()  # For Z-score normalization
    
    def load_and_merge_features(self):
        """Load and merge global and local features"""
        # Load global CT features
        circle_df = pd.read_csv(self.circle_features_path)
        # Load local radiomics features
        radiomics_df = pd.read_csv(self.radiomics_features_path)
        
        # Standardize column names
        if 'file_name' in circle_df.columns:
            circle_df = circle_df.rename(columns={'file_name': 'image_name'})
        
        if 'image_name' not in circle_df.columns:
            circle_df = circle_df.rename(columns={circle_df.columns[0]: 'image_name'})
        
        if 'image_name' not in radiomics_df.columns:
            radiomics_df = radiomics_df.rename(columns={radiomics_df.columns[0]: 'image_name'})
        
        # Merge features by image name
        merged_features = pd.merge(circle_df, radiomics_df, on='image_name', how='inner')
        
        # Separate IDs from features
        image_ids = merged_features['image_name']
        features = merged_features.drop('image_name', axis=1)
        self.feature_names = list(features.columns)
        
        return image_ids, features
    
    def load_labels(self, image_ids):
        """Load labels and match with feature data"""
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
        
        # Create mapping dictionary
        label_dict = dict(zip(labels_df['image_name'], labels_df['label']))
        
        # Match labels to features
        labels = []
        valid_indices = []
        
        for i, img_id in enumerate(image_ids):
            if img_id in label_dict:
                labels.append(label_dict[img_id])
                valid_indices.append(i)
        
        return image_ids.iloc[valid_indices], labels
    
    def preprocess_data(self, features, fit_scaler=True):
        """Apply Z-score normalization to features"""
        if fit_scaler:
            # Fit scaler on training data
            features_scaled = self.scaler.fit_transform(features)
        else:
            # Transform test data using fitted scaler
            features_scaled = self.scaler.transform(features)
        
        return features_scaled
    
    def train_model(self, X_train, y_train, X_val=None, y_val=None):
        """Train XGBoost classifier"""
        self.model = xgb.XGBClassifier(**self.model_params)
        
        if X_val is not None and y_val is not None:
            # Use early stopping if validation set provided
            eval_set = [(X_train, y_train), (X_val, y_val)]
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                early_stopping_rounds=10,
                verbose=False
            )
        else:
            # Train without validation
            self.model.fit(X_train, y_train)
        
        return self.model
    
    def predict_proba(self, X):
        """Predict probability of positive class (RWMA)"""
        return self.model.predict_proba(X)[:, 1]
    
    def calculate_contribution_ratio(self):
        """Calculate relative contribution of local vs global features"""
        # Get feature importance scores (gain-based)
        importance_scores = self.model.get_booster().get_score(importance_type='gain')
        
        importance_df = pd.DataFrame({
            'feature': list(importance_scores.keys()),
            'importance': list(importance_scores.values())
        })
        
        # Map XGBoost's internal feature names (f0, f1, ...) to actual feature names
        feature_mapping = {}
        if self.feature_names is not None:
            for i, feature_name in enumerate(self.feature_names):
                xgb_feature_name = f'f{i}'
                feature_mapping[xgb_feature_name] = feature_name
        
        importance_df['feature_name'] = importance_df['feature'].map(
            lambda x: feature_mapping.get(x, x)
        )
        
        # Calculate total importance for local and global features
        local_importance = 0
        global_importance = 0
        
        for feature in importance_df['feature_name']:
            if isinstance(feature, str):
                if feature.startswith('original_'):
                    # Local radiomics feature
                    local_importance += importance_df[
                        importance_df['feature_name'] == feature
                    ]['importance'].sum()
                elif feature.startswith('feature_'):
                    # Global CT feature
                    global_importance += importance_df[
                        importance_df['feature_name'] == feature
                    ]['importance'].sum()
        
        # Calculate relative contribution percentages
        total_importance = local_importance + global_importance
        
        if total_importance > 0:
            local_ratio = local_importance / total_importance * 100
            global_ratio = global_importance / total_importance * 100
        else:
            local_ratio = 0
            global_ratio = 0
        
        return local_ratio, global_ratio
    
    def save_model(self, filepath):
        """Save trained model for later use"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }, filepath)
    
    def run_pipeline(self, test_size=0.2):
        """Execute complete training and evaluation pipeline"""
        # Load and merge features
        image_ids, features = self.load_and_merge_features()
        image_ids, labels = self.load_labels(image_ids)
        labels = np.array(labels)
        
        # Split data into training (80%) and test (20%) sets
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, 
            test_size=test_size, 
            stratify=labels,  # Preserve class distribution
            random_state=RANDOM_STATE
        )
        
        # Preprocess data: Z-score normalization
        X_train_scaled = self.preprocess_data(X_train, fit_scaler=True)
        X_test_scaled = self.preprocess_data(X_test, fit_scaler=False)
        
        # Train XGBoost model
        self.train_model(X_train_scaled, y_train)
        
        # Evaluate on test set
        y_pred_proba = self.predict_proba(X_test_scaled)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Calculate feature contribution ratio
        local_ratio, global_ratio = self.calculate_contribution_ratio()
        
        # Output results
        print(f"Model: XGBoost")
        print(f"Test set AUC: {auc:.4f}")
        print(f"Contribution - Local features: {local_ratio:.1f}%, Global features: {global_ratio:.1f}%")
        
        return {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'auc': auc,
            'local_ratio': local_ratio,
            'global_ratio': global_ratio
        }


if __name__ == "__main__":
    # File paths - updated with new file names
    circle_features_path = "data/circle_features_RWMA.csv"
    radiomics_features_path = "data/radiomics_features_RWMA.csv"
    labels_path = "data/labels_RWMA.csv"
    
    # Check if required files exist
    for path in [circle_features_path, radiomics_features_path, labels_path]:
        if not os.path.exists(path):
            print(f"Error: File not found - {path}")
            exit(1)
    
    # Create predictor instance
    predictor = RWMA_Predictor(
        circle_features_path=circle_features_path,
        radiomics_features_path=radiomics_features_path,
        labels_path=labels_path
    )
    
    try:
        # Run complete pipeline
        results = predictor.run_pipeline(test_size=0.2)
        # Save trained model
        predictor.save_model("rwma_prediction_model.pkl")
    except Exception as e:
        print(f"Error: {e}")
