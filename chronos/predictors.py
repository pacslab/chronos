import joblib
import pandas as pd
import numpy as np
import gdown
import os
import xgboost as xgb
import json


from .config import Models, Datasets, Optimizers, Devices
from .defaults import CATEGORICAL_GPU_COLS, CATEGORICAL_MLP_COLS, CATEGORICAL_CNN_COLS, CATEGORICAL_TRANSFORMER_COLS

from .mlp_probe import run_single_mlp_experiment
from .cnn_probe import run_single_cnn_experiment
from .transformer_probe import run_single_transformer_experiment


class Predictor:
    def __init__(self):
        self.script_dir = os.path.dirname(__file__)
        self.models_dir = os.path.join(self.script_dir, 'models')
        
        if not os.path.exists(self.models_dir):
            print(f"{self.models_dir} not found. Downloading and extracting models...")

            gdown.download_folder("https://drive.google.com/drive/folders/1nCRqkJYJw15sk2Zt6K0Tbfc5FUoBmSs7", output=self.models_dir, quiet=False)
    
    @staticmethod
    def predict(self, features):
        pass
    

class MLPEpochTimePredictor(Predictor):
    def __init__(self):
        super().__init__()
        model_path = f'{self.models_dir}/mlp_epoch_time_model.json'
        
        self.model = xgb.XGBRegressor()
        self.model.load_model(model_path)
        
        with open (f'{self.models_dir}/mlp_epoch_time_model_features.json', 'r') as f:
            self.feature_names = json.load(f)
        
    
    def predict(self, features):
        return self.model.predict(features)
    

class CNNEpochTimePredictor(Predictor):
    def __init__(self):
        super().__init__()
        model_path = f'{self.models_dir}/cnn_epoch_time_model.json'
        
        self.model = xgb.XGBRegressor()
        self.model.load_model(model_path)
        
        with open (f'{self.models_dir}/cnn_epoch_time_model_features.json', 'r') as f:
            self.feature_names = json.load(f)
        
    
    def predict(self, features):
        return self.model.predict(features)
   

class TransformerEpochTimePredictor(Predictor):
    def __init__(self):
        super().__init__()
        model_path = f'{self.models_dir}/transformer_epoch_time_model.json'
        
        self.model = xgb.XGBRegressor()
        self.model.load_model(model_path)
        
        with open (f'{self.models_dir}/transformer_epoch_time_model_features.json', 'r') as f:
            self.feature_names = json.load(f)
        
    
    def predict(self, features):
        return self.model.predict(features)
    

class MLPEpochConvergencePredictor(Predictor):
    def __init__(self):
        super().__init__()
        model_path = f'{self.models_dir}/mlp_epoch_convergence_model.pkl'
        
        self.model = joblib.load(model_path)
        
        with open (f'{self.models_dir}/mlp_epoch_convergence_model_features.json', 'r') as f:
            self.feature_names = json.load(f)
        
    
    def predict(self, features):
        return self.model["reg"].predict(features)
    

class CNNEpochConvergencePredictor(Predictor):
    def __init__(self):
        super().__init__()
        model_path = f'{self.models_dir}/cnn_epoch_convergence_model.pkl'
        
        self.model = joblib.load(model_path)
        
        with open (f'{self.models_dir}/cnn_epoch_convergence_model_features.json', 'r') as f:
            self.feature_names = json.load(f)
        
    
    def predict(self, features):
        return self.model["reg"].predict(features)
    

class TransformerEpochConvergencePredictor(Predictor):
    def __init__(self):
        super().__init__()
        model_path = f'{self.models_dir}/transformer_epoch_convergence_model.pkl'
        
        self.model = joblib.load(model_path)
        
        with open (f'{self.models_dir}/transformer_epoch_convergence_model_features.json', 'r') as f:
            self.feature_names = json.load(f)
        
    
    def predict(self, features):
        return self.model["reg"].predict(features)
    
    

class TrainingTimePredictor:
    """
    A class to predict training time based on model, dataset, and hardware configuration.
    """
    def __init__(self, model: Models, dataset: Datasets, batch_size: int, 
                 learning_rate: float, optimizer: Optimizers, gpu: Devices, precision = 32):
        
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.precision = precision
        self.gpu = gpu
        
        
        if self.model in {Models.Mixer, Models.Resmlp, Models.Asmlp}:
            self.epoch_time_predictor = MLPEpochTimePredictor()
            self.epoch_convergence_predictor = MLPEpochConvergencePredictor()
            self.epoch_time_features = self.model.specs.copy() | self.dataset.specs.copy() | self.gpu.specs.copy() | {
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "precision": self.precision,
                "amp": True
            }
            self.epoch_time_features = pd.DataFrame([self.epoch_time_features])
            
            self.epoch_convergence_features = self.model.specs.copy() | self.dataset.specs.copy() | {
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "optimizer": self.optimizer.value,
                "precision": self.precision,
                "model": self.model.value,
                "dataset": self.dataset.value,
                "architecture": "MLP"
            }
            self.epoch_convergence_features = pd.DataFrame([self.epoch_convergence_features])
            
            
        elif self.model in {Models.ResNet50, Models.VGG16, Models.MobileNetV2, Models.DenseNet121}:
            self.epoch_time_predictor = CNNEpochTimePredictor()
            self.epoch_convergence_predictor = CNNEpochConvergencePredictor()
            self.epoch_time_features = self.model.specs.copy() | self.dataset.specs.copy() | self.gpu.specs.copy() | {
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "precision": self.precision,
                "amp": True
            }
            self.epoch_time_features = pd.DataFrame([self.epoch_time_features])
            
            self.epoch_convergence_features = self.model.specs.copy() | self.dataset.specs.copy() | {
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "optimizer": self.optimizer.value,
                "precision": self.precision,
                "model": self.model.value,
                "dataset": self.dataset.value,
                "architecture": "CNN"
            }
            self.epoch_convergence_features = pd.DataFrame([self.epoch_convergence_features])


        elif self.model in {Models.Transformer, Models.DistilBERT, Models.ViT, Models.DeiTTiny}:
            self.epoch_time_predictor = TransformerEpochTimePredictor()
            self.epoch_convergence_predictor = TransformerEpochConvergencePredictor()
            if self.dataset not in {Datasets.Cifar100, Datasets.TinyImageNet}:
                self.epoch_time_features = self.model.specs.copy() | self.dataset.specs.copy() | self.gpu.specs.copy() | {
                    "batch_size": self.batch_size,
                    "effective_batch_size": self.batch_size,
                    "learning_rate": self.learning_rate,
                    "precision": self.precision,
                    "amp_enabled": True
                }
            else:
                dataset_features = {
                    "dataset_size": 50000,
                    "seq_len": 196,
                    "src_len": 196,
                    "tgt_len": float('nan'),
                } if self.dataset == Datasets.Cifar100 else {
                    "dataset_size": 100000,
                    "seq_len": 196,
                    "src_len": 196,
                    "tgt_len": float('nan'),
                }
                self.epoch_time_features = self.model.specs.copy() | dataset_features | self.gpu.specs.copy() | {
                    "batch_size": self.batch_size,
                    "effective_batch_size": self.batch_size,
                    "learning_rate": self.learning_rate,
                    "precision": self.precision,
                    "amp_enabled": True
                }
            self.epoch_time_features = pd.DataFrame([self.epoch_time_features])
            
            self.epoch_convergence_features = self.model.specs.copy() | self.dataset.specs.copy() | {
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "optimizer": self.optimizer.value,
                "precision": self.precision,
                "model": self.model.value,
                "dataset": self.dataset.value,
                "architecture": "Transformer"
            }
            self.epoch_convergence_features = pd.DataFrame([self.epoch_convergence_features])

        
        print(f"Predictor configured for: {self.model.value} on {self.gpu.value}")
        print(f"Optimizer: {self.optimizer.value}, Dataset: {self.dataset.value}")
        
        
        self._preprocess_features()

    
    def predict_epoch_time(self):
        return self.epoch_time_predictor.predict(self.epoch_time_features)[0] * (self.dataset.specs.get("dataset_size", 1) / self.batch_size) / 1000.0
    

    def predict_number_of_epochs(self):
        return self.epoch_convergence_predictor.predict(self.epoch_convergence_features)[0]
    
    
    def _safe_ratio(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a / np.maximum(1e-9, b)


    def _safe_add_ratio(self, df: pd.DataFrame, num: str, den: str, out: str):
        if num in df.columns and den in df.columns:
            df[out] = df[num].astype(float) / np.maximum(1e-9, df[den].astype(float))
        return df
    
    
    def _preprocess_time_features(self):
        if self.model in {Models.Mixer, Models.Resmlp, Models.Asmlp}:
            features = self.epoch_time_features.copy()
            features = self._safe_add_ratio(features, "flops_train_per_batch", "GPU Memory Bandwidth (GB/s)", "flops_to_bw_ratio")
            for peak_col in ["FP32", "FP16", "FP64"]:
                if peak_col in features.columns:
                    features = self._safe_add_ratio(features, "flops_train_per_batch", peak_col, f"flops_to_{peak_col}_ratio")
            if "GPU Memory Bandwidth (GB/s)" in features.columns and "Memory (GB)" in features.columns:
                features = self._safe_add_ratio(features, "GPU Memory Bandwidth (GB/s)", "Memory (GB)", "bw_per_gb")
            if "arithmetic_intensity_train" in features.columns and "FP32" in features.columns:
                features = self._safe_add_ratio(features, "arithmetic_intensity_train", "FP32", "ai_over_fp32")
            
            
        elif self.model in {Models.ResNet50, Models.VGG16, Models.MobileNetV2, Models.DenseNet121}:
            features = self.epoch_time_features.copy()
            self._safe_add_ratio(features, "flops_train_per_batch", "GPU Memory Bandwidth (GB/s)", "flops_to_bw_ratio")
            for peak_col in ["FP32", "FP16", "FP64"]:
                if peak_col in features.columns:
                    self._safe_add_ratio(features, "flops_train_per_batch", peak_col, f"flops_to_{peak_col}_ratio")

            if "GPU Memory Bandwidth (GB/s)" in features.columns and "Memory (GB)" in features.columns:
                self._safe_add_ratio(features, "GPU Memory Bandwidth (GB/s)", "Memory (GB)", "bw_per_gb")

            if "arithmetic_intensity_train" in features.columns and "FP32" in features.columns:
                self._safe_add_ratio(features, "arithmetic_intensity_train", "FP32", "ai_over_fp32")

            if "param_bytes" in features.columns and "param_count" in features.columns:
                denom = np.maximum(1, features["param_count"].astype(float).values)
                features["bytes_per_param"] = features["param_bytes"].astype(float).values / denom

            if "act_bytes_per_sample_proxy" in features.columns and "param_bytes" in features.columns:
                features["acts_to_params"] = self._safe_ratio(
                    features["act_bytes_per_sample_proxy"].astype(float).values,
                    features["param_bytes"].astype(float).values
                )
        
            
        elif self.model in {Models.Transformer, Models.DistilBERT, Models.ViT, Models.DeiTTiny}:
            features = self.epoch_time_features.copy()
            
            self._safe_add_ratio(features, "flops_train_per_batch", "GPU Memory Bandwidth (GB/s)", "flops_to_bw_ratio")
            for peak_col in ["FP32", "FP16", "FP64"]:
                if peak_col in features.columns:
                    self._safe_add_ratio(features, "flops_train_per_batch", peak_col, f"flops_to_{peak_col}_ratio")

            if "GPU Memory Bandwidth (GB/s)" in features.columns and "Memory (GB)" in features.columns:
                self._safe_add_ratio(features, "GPU Memory Bandwidth (GB/s)", "Memory (GB)", "bw_per_gb")

            if "arithmetic_intensity_train" in features.columns and "FP32" in features.columns:
                self._safe_add_ratio(features, "arithmetic_intensity_train", "FP32", "ai_over_fp32")
        
            if "param_bytes" in features.columns and "param_count" in features.columns:
                denom = np.maximum(1, features["param_count"].astype(float).values)
                features["bytes_per_param"] = features["param_bytes"].astype(float).values / denom

            if "act_bytes_per_seq_proxy" in features.columns and "param_bytes" in features.columns:
                features["acts_to_params"] = self._safe_ratio(
                    features["act_bytes_per_seq_proxy"].astype(float).values,
                    features["param_bytes"].astype(float).values
                )

            if "d_model" in features.columns and "n_heads" in features.columns:
                features["d_head_check"] = self._safe_ratio(features["d_model"].astype(float).values,
                                                np.maximum(1.0, features["n_heads"].astype(float).values))

            if "d_ff" in features.columns and "d_model" in features.columns:
                features["ff_multiplier"] = self._safe_ratio(features["d_ff"].astype(float).values, features["d_model"].astype(float).values)


            if "batch_size" in features.columns:
                if "src_len" in features.columns and "tgt_len" in features.columns:
                    src = features["src_len"].fillna(0).astype(float).values
                    tgt = features["tgt_len"].fillna(0).astype(float).values
                    seq_tokens = src + tgt
                elif "seq_len" in features.columns:
                    seq_tokens = features["seq_len"].fillna(0).astype(float).values
                else:
                    seq_tokens = np.zeros(len(features), dtype=float)
                features["seq_token_load"] = seq_tokens * features["batch_size"].astype(float).values


        features = pd.get_dummies(features, columns=CATEGORICAL_GPU_COLS.keys())
        for col, categories in CATEGORICAL_GPU_COLS.items():
            for cat in categories:
                dummy_col = f"{col}_{cat}"
                if dummy_col not in features.columns:
                    features[dummy_col] = 0          
        self.epoch_time_features = features[self.epoch_time_predictor.feature_names]

  
    def _preprocess_convergence_features(self):
        if self.model in {Models.Mixer, Models.Resmlp, Models.Asmlp}:
            features = self.epoch_convergence_features.copy()
            probe_features = run_single_mlp_experiment(
                dataset=self.dataset.value,
                model=self.model.value,
                batch_size=self.batch_size,
                probe_batch_size=self.batch_size,
                lr=self.learning_rate,
                optimizer=self.optimizer.value,
                precision=self.precision,
                embed_dim=self.model.specs.get("embed_dim", 256),
                depth=self.model.specs.get("depth", 8),
                patch_size=self.model.specs.get("patch_size", 8),
            )
            features = features.join(pd.DataFrame([probe_features]))
            
            features = pd.get_dummies(features, columns=CATEGORICAL_MLP_COLS.keys())
            for col, categories in CATEGORICAL_MLP_COLS.items():
                for cat in categories:
                    dummy_col = f"{col}_{cat}"
                    if dummy_col not in features.columns:
                        features[dummy_col] = 0

            
        elif self.model in {Models.ResNet50, Models.VGG16, Models.MobileNetV2, Models.DenseNet121}:
            features = self.epoch_convergence_features.copy()
            probe_features = run_single_cnn_experiment(
                dataset=self.dataset.value,
                model=self.model.value,
                batch_size=self.batch_size,
                probe_batch_size=self.batch_size,
                lr=self.learning_rate,
                optimizer=self.optimizer.value,
                precision=self.precision,
            )
            features = features.join(pd.DataFrame([probe_features]))
            
            features = pd.get_dummies(features, columns=CATEGORICAL_CNN_COLS.keys())
            for col, categories in CATEGORICAL_CNN_COLS.items():
                for cat in categories:
                    dummy_col = f"{col}_{cat}"
                    if dummy_col not in features.columns:
                        features[dummy_col] = 0
            
            
        elif self.model in {Models.Transformer, Models.DistilBERT, Models.ViT, Models.DeiTTiny}:
            features = self.epoch_convergence_features.copy()
            probe_features = run_single_transformer_experiment(
                dataset=self.dataset.value,
                model=self.model.value,
                batch_size=self.batch_size,
                lr=self.learning_rate,
                optimizer=self.optimizer.value,
                precision=self.precision,
            )
            features = features.join(pd.DataFrame([probe_features]))
            
            features = pd.get_dummies(features, columns=CATEGORICAL_TRANSFORMER_COLS.keys())
            for col, categories in CATEGORICAL_TRANSFORMER_COLS.items():
                for cat in categories:
                    dummy_col = f"{col}_{cat}"
                    if dummy_col not in features.columns:
                        features[dummy_col] = 0


        self.epoch_convergence_features = features[self.epoch_convergence_predictor.feature_names]
    

    def _preprocess_features(self):  
        self._preprocess_time_features()
        self._preprocess_convergence_features()
