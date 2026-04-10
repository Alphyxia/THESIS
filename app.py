import os
import json
import random
import numpy as np
from PIL import Image
import onnxruntime as ort
import gradio as gr
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================
# Constants
# ============================================
MODELS_DIR = Path("models")
DATASET_DIR = Path("dataset")
IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Class names mapping
CLASS_NAMES = [
    "C1_Drive_Safe",
    "C2_Sleep",
    "C3_Yawning",
    "C4_Talk_Left",
    "C5_Talk_Right",
    "C6_Text_Left",
    "C7_Text_Right",
    "C8_Make_Up",
    "C9_Look_Left",
    "C10_Look_Right",
    "C11_Look_Up",
    "C12_Look_Down",
    "C13_Smoke_Left",
    "C14_Smoke_Right",
    "C15_Smoke_Mouth",
    "C16_Eat_Left",
    "C17_Eat_Right",
    "C18_Operate_Radio",
    "C19_Operate_GPS",
    "C20_Reach_Behind",
    "C21_Leave_Steering_Wheel",
    "C22_Talk_to_Passenger"
]

# ============================================
# Utility Functions
# ============================================

def get_available_models():
    """Get list of available models with their validation F1 scores."""
    models = []
    for model_dir in MODELS_DIR.iterdir():
        if model_dir.is_dir():
            stats_file = model_dir / "training_stats.json"
            best_model = model_dir / "best" / "model.onnx"

            if stats_file.exists() and best_model.exists():
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                    best_f1 = stats.get('summary', {}).get('best_val_f1', 0)
                    models.append({
                        'name': model_dir.name,
                        'path': str(best_model),
                        'val_f1': best_f1
                    })

    # Sort by validation F1 score (descending)
    models.sort(key=lambda x: x['val_f1'], reverse=True)
    return models

def load_onnx_model(model_path):
    """Load ONNX model using ONNX Runtime."""
    session = ort.InferenceSession(model_path)
    return session

def preprocess_image(image):
    """Preprocess image for model inference."""
    # Resize image
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert('RGB')

    image = image.resize(IMAGE_SIZE)

    # Convert to numpy array and normalize
    img_array = np.array(image, dtype=np.float32) / 255.0

    # Apply ImageNet normalization
    img_array = (img_array - IMAGENET_MEAN) / IMAGENET_STD

    # Transpose to CHW format and add batch dimension
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)

    return img_array, image

def get_class_images(dataset_name, class_name):
    """Get all image paths for a specific class in the dataset."""
    class_dir = DATASET_DIR / dataset_name / class_name
    if not class_dir.exists():
        return []

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = [
        str(img) for img in class_dir.iterdir()
        if img.suffix.lower() in image_extensions
    ]
    return images

# ============================================
# Inference Functions
# ============================================

def predict_image(model_session, image):
    """Make prediction on an image."""
    # Preprocess image
    img_array, original_image = preprocess_image(image)

    # Get input name from the model
    input_name = model_session.get_inputs()[0].name

    # Run inference
    outputs = model_session.run(None, {input_name: img_array})
    logits = outputs[0][0]

    # Apply softmax to get probabilities
    exp_logits = np.exp(logits - np.max(logits))
    probabilities = exp_logits / np.sum(exp_logits)

    # Get top prediction
    pred_idx = np.argmax(probabilities)
    pred_class = CLASS_NAMES[pred_idx]
    confidence = probabilities[pred_idx]

    # Create confidence distribution for all classes
    class_confidences = {CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))}

    return pred_class, confidence, class_confidences, original_image

# ============================================
# Gradio Interface Functions
# ============================================

# Global variable to store loaded model
current_model = None
current_model_name = None

def load_model_for_inference(model_name):
    """Load model for inference."""
    global current_model, current_model_name

    models = get_available_models()
    model_info = next((m for m in models if m['name'] == model_name), None)

    if model_info:
        current_model = load_onnx_model(model_info['path'])
        current_model_name = model_name
        return f"Model '{model_name}' loaded successfully (Val F1: {model_info['val_f1']:.4f})"
    else:
        return "Model not found"

def inference_dataset_image(model_name, dataset_name, class_name):
    """Run inference on a random image from the dataset."""
    global current_model, current_model_name

    # Load model if different from current
    if current_model is None or current_model_name != model_name:
        load_model_for_inference(model_name)

    # Get random image from class
    images = get_class_images(dataset_name, class_name)
    if not images:
        return None, "No images found in this class", None

    random_image_path = random.choice(images)

    # Make prediction
    pred_class, confidence, class_confidences, image = predict_image(current_model, random_image_path)

    # Create result text
    result_text = f"""
    **Predicted Class:** {pred_class}
    **Actual Class:** {class_name}
    **Confidence:** {confidence:.2%}
    **Match:** {'✓ Correct' if pred_class == class_name else '✗ Incorrect'}
    """

    # Sort confidences by value
    sorted_confidences = dict(sorted(class_confidences.items(), key=lambda x: x[1], reverse=True)[:5])

    return image, result_text, sorted_confidences

def inference_custom_image(model_name, image):
    """Run inference on a custom uploaded image."""
    global current_model, current_model_name

    if image is None:
        return None, "Please upload an image", None

    # Load model if different from current
    if current_model is None or current_model_name != model_name:
        load_model_for_inference(model_name)

    # Make prediction
    pred_class, confidence, class_confidences, processed_image = predict_image(current_model, image)

    # Create result text
    result_text = f"""
    **Predicted Class:** {pred_class}
    **Confidence:** {confidence:.2%}
    """

    # Sort confidences by value
    sorted_confidences = dict(sorted(class_confidences.items(), key=lambda x: x[1], reverse=True)[:5])

    return processed_image, result_text, sorted_confidences

# ============================================
# Statistics Visualization Functions
# ============================================

def load_training_stats(model_name):
    """Load training statistics for a model."""
    stats_file = MODELS_DIR / model_name / "training_stats.json"
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            return json.load(f)
    return None

def plot_training_statistics(model_name):
    """Create plots for training statistics."""
    stats = load_training_stats(model_name)

    if stats is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No statistics found for this model',
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig

    epochs = stats['epochs']
    train_metrics = stats['train_metrics']
    val_metrics = stats['val_metrics']

    # Create a figure with subplots (2 rows, 2 columns)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Training Statistics - {model_name}', fontsize=16, fontweight='bold')

    # Plot 1: Loss (Train and Validation on same plot)
    axes[0, 0].plot(epochs, train_metrics['loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, val_metrics['loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Plot 2: Accuracy (Train and Validation on same plot)
    axes[0, 1].plot(epochs, train_metrics['accuracy'], 'b-', label='Train Accuracy', linewidth=2)
    axes[0, 1].plot(epochs, val_metrics['accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[0, 1].set_title('Accuracy', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Plot 3: F1 Score (Train and Validation on same plot)
    axes[1, 0].plot(epochs, train_metrics['f1_score'], 'b-', label='Train F1', linewidth=2)
    axes[1, 0].plot(epochs, val_metrics['f1_score'], 'r-', label='Validation F1', linewidth=2)
    axes[1, 0].set_title('F1 Score', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # Plot 4: Summary statistics
    summary = stats.get('summary', {})
    summary_text = f"""
    Best Validation Accuracy: {summary.get('best_val_accuracy', 0):.4f}
    Best Validation F1: {summary.get('best_val_f1', 0):.4f}
    Total Training Time: {summary.get('total_training_time', 0):.2f}s
    Average Epoch Time: {summary.get('avg_epoch_time', 0):.2f}s
    Total Epochs: {len(epochs)}
    """
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
                    verticalalignment='center')
    axes[1, 1].set_title('Training Summary', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')

    plt.tight_layout()

    return fig

def plot_additional_metrics(model_name):
    """Create plots for precision, recall, and training time."""
    stats = load_training_stats(model_name)

    if stats is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No statistics found for this model',
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig

    epochs = stats['epochs']
    train_metrics = stats['train_metrics']
    val_metrics = stats['val_metrics']

    # Create a figure with subplots (1 row, 3 columns)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Additional Training Metrics - {model_name}', fontsize=16, fontweight='bold')

    # Plot 1: Precision (Train and Validation on same plot)
    axes[0].plot(epochs, train_metrics['precision'], 'b-', label='Train Precision', linewidth=2)
    axes[0].plot(epochs, val_metrics['precision'], 'r-', label='Validation Precision', linewidth=2)
    axes[0].set_title('Precision', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Precision')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Plot 2: Recall (Train and Validation on same plot)
    axes[1].plot(epochs, train_metrics['recall'], 'b-', label='Train Recall', linewidth=2)
    axes[1].plot(epochs, val_metrics['recall'], 'r-', label='Validation Recall', linewidth=2)
    axes[1].set_title('Recall', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Recall')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Plot 3: Training Time
    axes[2].plot(epochs, train_metrics['training_time'], 'g-', label='Training Time', linewidth=2)
    axes[2].set_title('Training Time per Epoch', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Time (seconds)')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()

    return fig

# ============================================
# Gradio Interface Setup
# ============================================

def create_app():
    """Create the Gradio application."""

    # Get available models
    models = get_available_models()
    model_names = [m['name'] for m in models]

    # Auto-load best model
    if models:
        default_model = models[0]['name']
        load_model_for_inference(default_model)
    else:
        default_model = None

    # Available datasets
    datasets = ["Cam2", "Cam4"]

    with gr.Blocks(title="Driver Distraction Classification", theme=gr.themes.Soft()) as app:
        gr.Markdown("# Driver Distraction Classification System")
        gr.Markdown("Inference and analysis tool for driver distraction detection models")

        with gr.Tabs():
            # ============================================
            # Tab 1: Inference
            # ============================================
            with gr.Tab("Inference"):
                with gr.Row():
                    model_selector = gr.Dropdown(
                        choices=model_names,
                        value=default_model,
                        label="Select Model",
                        info="Models sorted by validation F1 score (best first)"
                    )

                with gr.Row():
                    mode_selector = gr.Radio(
                        choices=["Dataset Image", "Custom Upload"],
                        value="Dataset Image",
                        label="Inference Mode"
                    )

                # Dataset Image Mode
                with gr.Group(visible=True) as dataset_group:
                    gr.Markdown("### Dataset Image Inference")
                    with gr.Row():
                        dataset_selector = gr.Dropdown(
                            choices=datasets,
                            value="Cam2",
                            label="Select Dataset"
                        )
                        class_selector = gr.Dropdown(
                            choices=CLASS_NAMES,
                            value=CLASS_NAMES[0],
                            label="Select Class"
                        )

                    random_btn = gr.Button("Get Random Image and Predict", variant="primary")

                    with gr.Row():
                        dataset_image_output = gr.Image(label="Selected Image")
                        with gr.Column():
                            dataset_result_output = gr.Markdown(label="Prediction Results")
                            dataset_confidence_output = gr.Label(label="Top 5 Predictions", num_top_classes=5)

                # Custom Upload Mode
                with gr.Group(visible=False) as custom_group:
                    gr.Markdown("### Custom Image Upload")

                    custom_image_input = gr.Image(label="Upload Image", type="numpy")
                    custom_predict_btn = gr.Button("Predict", variant="primary")

                    with gr.Row():
                        custom_image_output = gr.Image(label="Processed Image")
                        with gr.Column():
                            custom_result_output = gr.Markdown(label="Prediction Results")
                            custom_confidence_output = gr.Label(label="Top 5 Predictions", num_top_classes=5)

                # Toggle visibility based on mode
                def toggle_mode(mode):
                    if mode == "Dataset Image":
                        return gr.update(visible=True), gr.update(visible=False)
                    else:
                        return gr.update(visible=False), gr.update(visible=True)

                mode_selector.change(
                    toggle_mode,
                    inputs=[mode_selector],
                    outputs=[dataset_group, custom_group]
                )

                # Dataset image inference
                random_btn.click(
                    inference_dataset_image,
                    inputs=[model_selector, dataset_selector, class_selector],
                    outputs=[dataset_image_output, dataset_result_output, dataset_confidence_output]
                )

                # Custom image inference
                custom_predict_btn.click(
                    inference_custom_image,
                    inputs=[model_selector, custom_image_input],
                    outputs=[custom_image_output, custom_result_output, custom_confidence_output]
                )

            # ============================================
            # Tab 2: Training Statistics
            # ============================================
            with gr.Tab("Training Statistics"):
                gr.Markdown("### Visualize Training Metrics")

                stats_model_selector = gr.Dropdown(
                    choices=model_names,
                    value=default_model,
                    label="Select Model"
                )

                plot_btn = gr.Button("Generate Plots", variant="primary")

                with gr.Row():
                    main_plot_output = gr.Plot(label="Main Metrics (Loss, Accuracy, F1)")

                with gr.Row():
                    additional_plot_output = gr.Plot(label="Additional Metrics (Precision, Recall, Time)")

                # Generate plots
                def generate_all_plots(model_name):
                    main_plot = plot_training_statistics(model_name)
                    additional_plot = plot_additional_metrics(model_name)
                    return main_plot, additional_plot

                plot_btn.click(
                    generate_all_plots,
                    inputs=[stats_model_selector],
                    outputs=[main_plot_output, additional_plot_output]
                )

                # Auto-generate plots for default model
                if default_model:
                    app.load(
                        generate_all_plots,
                        inputs=[stats_model_selector],
                        outputs=[main_plot_output, additional_plot_output]
                    )

    return app

# ============================================
# Main Execution
# ============================================

if __name__ == "__main__":
    app = create_app()
    app.launch(share=False, server_name="0.0.0.0", server_port=7860)
