"""
Deepfake Detection Model Definition
===================================

This file contains the model architecture for deepfake detection.
Use this to load pre-trained models and perform inference.

Usage Example:
    from deepfake_model import DeepfakeDetector, load_model
    
    # Load trained model
    detector = load_model('path/to/best_model.pth')
    
    # Predict on video
    result = detector.predict_video('path/to/video.mp4')
    print(f"Prediction: {result['label']} (confidence: {result['confidence']:.2f})")
"""

import torch
from torch import nn
import numpy as np
import cv2
from torchvision import models
#import albumentations as A
#from albumentations.pytorch import ToTensorV2
import torchvision.transforms as T
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer-style attention."""
    
    def __init__(self, d_model, max_seq_length=200):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class EnhancedMultiHeadAttentionLayer(nn.Module):
    """Enhanced multi-head attention layer with positional encoding."""
    
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super(EnhancedMultiHeadAttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.positional_encoding = PositionalEncoding(hidden_dim)
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, lstm_output):
        # Add positional encoding
        lstm_output = self.positional_encoding(lstm_output)
        
        # Apply multi-head attention
        attn_output, attn_weights = self.multihead_attn(
            lstm_output.transpose(0, 1),
            lstm_output.transpose(0, 1),
            lstm_output.transpose(0, 1)
        )
        
        # Residual connection and normalization
        attn_output = attn_output.transpose(0, 1)
        output = self.norm(lstm_output + self.dropout(attn_output))
        
        return output, attn_weights


class ImprovedDeepfakeModel(nn.Module):
    """
    Improved deepfake detection model using ResNeXt backbone with LSTM and attention.
    
    Architecture:
    - ResNeXt50 feature extractor
    - Bidirectional LSTM for temporal modeling
    - Multi-head attention mechanism
    - Classification head with dropout
    """
    
    def __init__(self, num_classes=2, latent_dim=2048, lstm_layers=2, hidden_dim=1024, 
                 num_heads=4, bidirectional=True, dropout=0.3):
        super(ImprovedDeepfakeModel, self).__init__()
        
        # Store configuration
        self.config = {
            'num_classes': num_classes,
            'latent_dim': latent_dim,
            'lstm_layers': lstm_layers,
            'hidden_dim': hidden_dim,
            'num_heads': num_heads,
            'bidirectional': bidirectional,
            'dropout': dropout
        }
        
        # ResNeXt50 backbone
        resnext = models.resnext50_32x4d(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnext.children())[:-2])
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            latent_dim, 
            hidden_dim, 
            lstm_layers, 
            bidirectional=bidirectional, 
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Enhanced attention mechanism
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.attention = EnhancedMultiHeadAttentionLayer(lstm_output_dim, num_heads, dropout)
        
        # Classification head
        self.fc1 = nn.Linear(lstm_output_dim, lstm_output_dim // 2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(lstm_output_dim // 2, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, channels, height, width)
            
        Returns:
            tuple: (features, logits, attention_weights)
        """
        batch_size, seq_len, c, h, w = x.shape
        
        # Extract features from each frame
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.feature_extractor(x)
        features = self.global_avg_pool(features)
        features = features.view(batch_size, seq_len, -1)
        
        # Process with LSTM
        lstm_output, _ = self.lstm(features)
        
        # Apply attention
        attention_output, attention_weights = self.attention(lstm_output)
        
        # Average pooling over sequence dimension
        pooled_output = torch.mean(attention_output, dim=1)
        
        # Classification
        x = self.dropout(pooled_output)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return features, logits, attention_weights
    
    def get_config(self):
        """Return model configuration."""
        return self.config.copy()


class DeepfakeDetector:
    """
    High-level interface for deepfake detection.
    
    This class provides an easy-to-use interface for loading models and making predictions.
    """
    
    def __init__(self, model_path=None, device=None):
        """
        Initialize the deepfake detector.
        
        Args:
            model_path: Path to the trained model weights
            device: Device to run inference on ('cuda', 'cpu', or None for auto-detect)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = self._create_transform()
        
        if model_path:
            self.load_model(model_path)
    
    def _create_transform(self):
        """Create image preprocessing pipeline."""
        return T.Compose([
            T.ToPILImage(),
            T.Resize((180, 180)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
    
    
    def load_model(self, model_path, model_config=None):
        """
        Load a trained model from file.
        
        Args:
            model_path: Path to the model weights (.pth file)
            model_config: Optional model configuration dict. If None, uses default config.
        """
        # Default configuration (should match training configuration)
        default_config = {
            'num_classes': 2,
            'latent_dim': 2048,
            'lstm_layers': 2,
            'hidden_dim': 1024,
            'num_heads': 4,
            'bidirectional': True,
            'dropout': 0.3
        }
        
        config = model_config or default_config
        
        # Initialize model
        self.model = ImprovedDeepfakeModel(**config)
        
        # Load weights
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully on {self.device}")
    
    def extract_frames(self, video_path, sequence_length=30):
        """
        Extract frames from video for inference.
        
        Args:
            video_path: Path to the video file
            sequence_length: Number of frames to extract
            
        Returns:
            torch.Tensor: Preprocessed frames tensor
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count == 0:
            raise ValueError(f"Video has no frames: {video_path}")
        
        # Uniform sampling
        frame_step = max(frame_count // sequence_length, 1)
        frame_indices = range(0, min(frame_count, frame_step * sequence_length), frame_step)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        cap.release()
        
        # Handle case with fewer frames than needed
        if len(frames) < sequence_length:
            last_frame = frames[-1] if frames else np.zeros((180, 180, 3), dtype=np.uint8)
            frames.extend([last_frame] * (sequence_length - len(frames)))
        
        # Ensure we have exactly sequence_length frames
        frames = frames[:sequence_length]
        
        # Apply transforms
        frame_tensors = []
        for frame in frames:
            transformed = self.transform(frame)   # -> tensor CxHxW
            frame_tensors.append(transformed)
        
        # Stack frames and add batch dimension
        frames_tensor = torch.stack(frame_tensors).unsqueeze(0)  # (1, seq_len, C, H, W)
        
        return frames_tensor
    
    def predict_video(self, video_path, sequence_length=30):
        """
        Predict whether a video is real or fake.
        
        Args:
            video_path: Path to the video file
            sequence_length: Number of frames to use for prediction
            
        Returns:
            dict: Prediction results with keys:
                - 'label': 'REAL' or 'FAKE'
                - 'confidence': Confidence score (0-1)
                - 'probabilities': Raw probabilities [fake_prob, real_prob]
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        # Extract and preprocess frames
        frames_tensor = self.extract_frames(video_path, sequence_length)
        frames_tensor = frames_tensor.to(self.device)
        
        # Run inference
        with torch.no_grad():
            features, logits, attention_weights = self.model(frames_tensor)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
            predicted_class = logits.argmax(dim=1).item()
        
        # Convert to interpretable results
        label = 'REAL' if predicted_class == 1 else 'FAKE'
        confidence = float(probabilities[predicted_class])
        
        return {
            'label': label,
            'confidence': confidence,
            'probabilities': {
                'FAKE': float(probabilities[0]),
                'REAL': float(probabilities[1])
            }
        }
    
    def predict_batch(self, video_paths, sequence_length=30):
        """
        Predict on multiple videos at once.
        
        Args:
            video_paths: List of video file paths
            sequence_length: Number of frames to use for each video
            
        Returns:
            list: List of prediction results for each video
        """
        results = []
        for video_path in video_paths:
            try:
                result = self.predict_video(video_path, sequence_length)
                result['video_path'] = str(video_path)
                results.append(result)
            except Exception as e:
                results.append({
                    'video_path': str(video_path),
                    'error': str(e),
                    'label': None,
                    'confidence': None,
                    'probabilities': None
                })
        return results


def load_model(model_path, model_config=None, device=None):
    """
    Convenience function to load a trained deepfake detection model.
    
    Args:
        model_path: Path to the model weights (.pth file)
        model_config: Optional model configuration dict
        device: Device to run inference on
        
    Returns:
        DeepfakeDetector: Ready-to-use detector instance
    """
    detector = DeepfakeDetector(device=device)
    detector.load_model(model_path, model_config)
    return detector


# Example usage and testing functions
if __name__ == "__main__":
    """
    Example usage of the deepfake detection model.
    """
    
    # Example 1: Load model and predict single video
    print("Loading deepfake detection model...")
    
    # Replace with your actual model path
    model_path = "models/best_model.pth"
    
    try:
        detector = load_model(model_path)
        print("Model loaded successfully!")
        
        # Example prediction (replace with actual video path)
        # result = detector.predict_video("path/to/your/video.mp4")
        # print(f"Prediction: {result['label']} (confidence: {result['confidence']:.2f})")
        
    except FileNotFoundError:
        print(f"Model file not found: {model_path}")
        print("Please ensure you have a trained model file.")
    
    # Example 2: Custom model configuration
    custom_config = {
        'num_classes': 2,
        'latent_dim': 2048,
        'lstm_layers': 3,  # Different from default
        'hidden_dim': 512,  # Different from default
        'num_heads': 8,     # Different from default
        'bidirectional': True,
        'dropout': 0.2
    }
    
    # Example 3: Batch prediction
    # video_paths = ["video1.mp4", "video2.mp4", "video3.mp4"]
    # results = detector.predict_batch(video_paths)
    # for result in results:
    #     print(f"{result['video_path']}: {result['label']} ({result['confidence']:.2f})")
