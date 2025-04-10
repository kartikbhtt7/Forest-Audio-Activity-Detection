# 🌳 Forest Sound Guardian

Real-time audio classification system for detecting human activities in forest environments using deep learning.

## Features

- 🎙️ Real-time audio analysis through upload or microphone recording
- 🔍 Detection of 14 different sound classes including:
  - Human activities (footsteps, coughing, laughing)
  - Tools (chainsaws, hand saws)
  - Vehicles (car horns, engines)
  - Environmental hazards (fire, fireworks)
- 📊 Interactive audio visualizations (waveform + spectrogram)
- 🚨 Smart alert system for critical detections

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/forest-sound-guardian.git
cd forest-sound-guardian
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download pre-trained model:
```bash
mkdir -p models
wget https://your-model-host.com/best_model.pth -O models/best_model.pth
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Choose input method:
   - **Upload** existing audio files (WAV/MP3/OGG)
   - **Record** live audio directly from your microphone

3. View results:
   - Real-time audio visualization
   - Detection confidence percentages
   - Priority alerts for critical sounds

## Project Structure

```
forest-sound-guardian/
├── app.py                 # Main Streamlit application
├── predict.py             # Model loading and prediction logic
├── models/                # Pre-trained model storage
│   └── best_model.pth
├── utils/
│   └── audio_preprocessing.py  # Audio processing utilities
├── requirements.txt       # Dependency list
└── README.md              # This documentation
```

## Technical Details

### Audio Processing Pipeline
1. **Loading**: Resample to 16kHz mono
2. **Normalization**: Trim/pad to 3-second duration
3. **Feature Extraction**: Mel-spectrogram conversion
   - 128 mel bands
   - 20-8000Hz frequency range
   - STFT window: 2560 samples

### Model Architecture
```plaintext
AudioClassifier(
  (features): Sequential(
    Conv2d(1→64, kernel=3x3) → BatchNorm → ReLU
    Conv2d(64→64) → BatchNorm → ReLU → MaxPool → Dropout
    Conv2d(64→128) → BatchNorm → ReLU
    Conv2d(128→128) → BatchNorm → ReLU → MaxPool → Dropout
    Conv2d(128→256) → BatchNorm → ReLU
    Conv2d(256→256) → BatchNorm → ReLU → MaxPool → Dropout
  )
  (classifier): Sequential(
    AdaptiveAvgPool → Flatten
    Linear(256→256) → ReLU
    Linear(256→14)
  )
)