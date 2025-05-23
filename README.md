# Real-Time Violence Detection using MobileNet Bi-LSTM

This project implements a real-time violence detection system using a combination of MobileNetV2 and Bidirectional LSTM networks. The model is designed to detect violent situations in video streams efficiently.

## Overview

The system uses a hybrid deep learning architecture:
- **MobileNetV2** for feature extraction from video frames
- **Bidirectional LSTM** for temporal sequence analysis
- Real-time processing capabilities for video streams

## Project Structure

```
├── MoBiLSTM_model.h5      # Trained model weights
├── MoBiLSTM_model.ipynb   # Jupyter notebook with model development
└── MoBiLSTM_model.py      # Python script version of the model
```

## Dependencies

- TensorFlow
- OpenCV (cv2)
- NumPy
- Matplotlib
- scikit-learn
- kagglehub (for dataset access)

## Model Architecture

The model combines:
1. MobileNetV2 as the base model for efficient feature extraction
2. Bidirectional LSTM layers for temporal analysis
3. Dense layers for final classification

## Dataset

The model is trained on the "Real Life Violence Situations Dataset" from Kaggle, which contains real-world examples of violent and non-violent situations.

## Usage

1. Install the required dependencies:
```bash
pip install tensorflow opencv-python numpy matplotlib scikit-learn kagglehub
```

2. Load the model:
```python
from tensorflow.keras.models import load_model
model = load_model('MoBiLSTM_model.h5')
```

3. Use the notebook or Python script for:
   - Model training
   - Real-time violence detection
   - Performance evaluation

## Model Performance

The model is optimized for real-time detection while maintaining high accuracy. For detailed performance metrics, please refer to the notebook.

## License

This project is for research and educational purposes. Please ensure appropriate licensing and permissions when using the dataset and deploying the model.

## Acknowledgments

- Dataset: Real Life Violence Situations Dataset (Kaggle)
- Base architecture: MobileNetV2 (Google)
