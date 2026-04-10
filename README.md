# Driver Distraction Classification Inference App

A Gradio-based web application for running inference on driver distraction detection models and visualizing training statistics.

## Features

### Inference Tab
- **Dataset Image Mode**:
  - Select from Cam2 or Cam4 datasets
  - Choose one of 22 distraction classes
  - Get random images from the selected class
  - View predictions with confidence scores
  - Compare predicted vs actual class

- **Custom Upload Mode**:
  - Upload your own images
  - Get predictions with confidence scores
  - View top 5 predicted classes

### Training Statistics Tab
- Visualize training metrics for all models:
  - Loss (Train & Validation)
  - Accuracy (Train & Validation)
  - F1 Score (Train & Validation)
  - Precision (Train & Validation)
  - Recall (Train & Validation)
  - Training Time per Epoch
- View summary statistics (best validation metrics, total/average training time)

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

Or if using the virtual environment:
```bash
venv/bin/pip install -r requirements.txt
```

## Usage

Run the application:
```bash
python app.py
```

Or with the virtual environment:
```bash
venv/bin/python app.py
```

The app will start at `http://localhost:7860`

## Model Selection

The app automatically loads the model with the highest validation F1 score on startup. You can switch between different models using the dropdown menu in both tabs.

## Supported Classes

The models are trained on 22 driver distraction classes:
- C1_Drive_Safe
- C2_Sleep
- C3_Yawning
- C4_Talk_Left
- C5_Talk_Right
- C6_Text_Left
- C7_Text_Right
- C8_Make_Up
- C9_Look_Left
- C10_Look_Right
- C11_Look_Up
- C12_Look_Down
- C13_Smoke_Left
- C14_Smoke_Right
- C15_Smoke_Mouth
- C16_Eat_Left
- C17_Eat_Right
- C18_Operate_Radio
- C19_Operate_GPS
- C20_Reach_Behind
- C21_Leave_Steering_Wheel
- C22_Talk_to_Passenger

## Technical Details

- **Model Format**: ONNX
- **Input Size**: 224x224
- **Preprocessing**: ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Inference Engine**: ONNX Runtime
