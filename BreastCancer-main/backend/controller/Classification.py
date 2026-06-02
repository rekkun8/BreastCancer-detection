import glob
import os

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model


def build_model(num_classes):
    IMG_SIZE = 224
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))  # RGB input
    x = inputs
    model = EfficientNetB0(include_top=False, input_tensor=x, weights=None)  # No pretrained weights

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model




def frame_splitter(video_path, output_folder, frame_rate=None):
    """
    Splits a video into frames and saves them in the specified folder.
    
    :param video_path: Path to the input video file.
    :param output_folder: Path to the folder where frames will be saved.
    :param frame_rate: Optional, specify frame extraction rate (default is all frames).
    """
    os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get original video FPS
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Stop if video ends
        
        # Save frame if frame_rate is specified (e.g., extract 1 frame per second)
        if frame_rate is None or (frame_count % int(fps / frame_rate) == 0):
            frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
            cv2.imwrite(frame_filename, frame)

        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames to {output_folder}")

def get_original_fps(video_path):
    """Retrieve the FPS of the original video."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 30  # Default to 30 if FPS can't be retrieved

def frames_to_video(frames_folder, output_video, original_video_path):
    """
    Reconstructs a video from frames and saves it with the original FPS.

    :param frames_folder: Folder containing image frames.
    :param output_video: Output video file path.
    :param original_video_path: Path to the original video to extract FPS.
    """
    # Get all frame file paths and sort them
    frame_files = sorted(glob.glob(os.path.join(frames_folder, "*.png")))

    if not frame_files:
        print("No frames found in the folder!")
        return

    # Get FPS from the original video
    frame_rate = get_original_fps(original_video_path)
    print(f"Using original FPS: {frame_rate}")

    # Read the first frame to get width and height
    first_frame = cv2.imread(frame_files[0])
    height, width, _ = first_frame.shape

    # Define the codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4 format
    out = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

    # Write frames to the video
    for file in frame_files:
        frame = cv2.imread(file)
        out.write(frame)

    # Release the video writer
    out.release()
    print(f"Video saved as {output_video}")

# Example Usage
original_video_path = r"C:\BreastCancer\ChanCode\backend\tempDB\videos\input\input_video_llm.mp4"
frames_dir = r"C:\BreastCancer\ChanCode\backend\tempDB\output_frames"
output_video_path = r"C:\BreastCancer\ChanCode\backend\tempDB\videos\output\video.mp4"

def classif(frame_path):
    from controller.GradCAM import get_gradcam_heatmap, overlay_gradcam
    image = Image.open(frame_path).resize((224, 224))
    path = r"D:\rayyan\aiml\projects\BreastCancer-main\BreastCancer-main\backend\models\Efficient_Net_Final.weights.h5"
    NUM_CLASSES = 3
    model = build_model(num_classes=NUM_CLASSES)
    model.load_weights(path)

    x = np.expand_dims(image, axis=0)
    x = preprocess_input(x)
    img_array = x  # ✅ Needed for Grad-CAM

    preds = model.predict(x)
    class_labels = ['benign', 'malignant', 'normal']
    max_index = np.argmax(preds[0])
    predicted_class = class_labels[max_index]

    heatmap = get_gradcam_heatmap(model, img_array)
    gradcam_path = overlay_gradcam(frame_path, heatmap)

    return predicted_class, gradcam_path
