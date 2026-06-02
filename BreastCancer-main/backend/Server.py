import os
import shutil
from io import BytesIO

import cv2
import numpy as np
import tensorflow as tf
from controller.Classification import *
from controller.GradCAM import get_gradcam_heatmap, overlay_gradcam
from controller.Segmentation import *
from controller.shap_explain import get_shap_visual
from flask import Flask, jsonify, render_template, request, send_file
from flask_cors import CORS
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.saving import CustomObjectScope
from tqdm import tqdm

app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)


UPLOAD_FOLDER = r"C:\Users\Rayyan\Desktop\personal\New folder\BreastCancer-main\BreastCancer-main\backend\tempDB\temp"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

frames_path=r"C:\Users\Rayyan\Desktop\personal\New folder\BreastCancer-main\BreastCancer-main\backend\tempDB\frames"
output_frames=r"C:\Users\Rayyan\Desktop\personal\New folder\BreastCancer-main\BreastCancer-main\backend\tempDB\output_frames"
input_video_path=r"C:\Users\Rayyan\Desktop\personal\New folder\BreastCancer-main\BreastCancer-main\backend\tempDB\videos"

@app.route("/test",methods=["GET"])
def test():
    return jsonify({"message":"Working"})

@app.route("/classification", methods=["POST"])
def classify_route():
    try:
        print("Received image request")

        image_data = request.files["image"]
        temp_image_path = os.path.join(app.config['UPLOAD_FOLDER'], "uploaded_img.png")
        image_data.save(temp_image_path)
        print("Image saved to:", temp_image_path)

        # Preprocess the image as RGB
        image = Image.open(temp_image_path).convert("RGB").resize((224, 224))
        image_np = np.array(image)
        img_array = preprocess_input(np.expand_dims(image_np, axis=0))

        # Build and load model
        path = r"C:\Users\Rayyan\Desktop\personal\New folder\BreastCancer-main\BreastCancer-main\backend\models\Efficient_Net_Final.weights.h5"
        NUM_CLASSES = 3
        model = build_model(num_classes=NUM_CLASSES)
        model.load_weights(path)

        # Prediction
        preds = model.predict(img_array)
        class_labels = ['benign', 'malignant', 'normal']
        max_index = np.argmax(preds[0])
        predicted_class = class_labels[max_index]

        print("Prediction Successful:", predicted_class)

        # Grad-CAM generation
        heatmap = get_gradcam_heatmap(model, tf.convert_to_tensor(img_array, dtype=tf.float32))
        gradcam_path = overlay_gradcam(temp_image_path, heatmap)
        gradcam_url = f"http://localhost:4000/{gradcam_path}"

        shap_path = get_shap_visual(model, temp_image_path)

        return jsonify({
            "prediction": predicted_class,
            "gradcam_image": gradcam_url,
            "shap_image": shap_path
        })

    except Exception as e:
        print("🔥 Error in /classification:", str(e))
        return jsonify({"error": str(e)}), 500



@app.route("/generate_video",methods=["POST"])
def generator():
    video = request.files["video"]
    video_path=os.path.join(input_video_path,video.filename)
    video.save(video_path)
    frame_splitter(video_path, frames_path)
    frame_files = os.listdir(frames_path)
    for x in frame_files:
        frame = x
        path=os.path.join(frames_path,frame)
        classification = classif(path)
        
        if classification in ["benign", "malignant"]:
            continue
        else:
            source = os.path.join(frames_path, x)
            destination = os.path.join(output_frames, x)
            shutil.move(source, destination)
    output_video_path=r"C:\BreastCancer\ChanCode\backend\tempDB\videos\output\video.mpv4"
    frames_to_video(output_frames,output_video_path,video_path)
    return send_file(output_video_path, mimetype="video/mp4")

    

@app.route("/segmentation_mask",methods=["POST"])
def seg():
    files=request.files
    image_data=files["image"]
    image = Image.open(image_data).resize((224, 224))
    image_np = np.array(image)
    model_path=r"C:\Users\Rayyan\Desktop\personal\New folder\BreastCancer-main\BreastCancer-main\backend\models\model_segmentation2.keras"
    with CustomObjectScope({"dice_coef": dice_coef, "dice_loss": dice_loss,"f1sc":f1sc}):
        model = tf.keras.models.load_model(model_path)
    x = image_np/255.0
    x = np.expand_dims(x, axis=0)
    y_pred = model.predict(x, verbose=0)[0]
    y_pred = np.squeeze(y_pred, axis=-1)
    y_pred = y_pred >= 0.5
    y_pred = y_pred.astype(np.int32)
    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)
    pred = y_pred * 255
    pred_image = Image.fromarray(pred.astype(np.uint8))
    
    # Save to in-memory file
    img_io = BytesIO()
    pred_image.save(img_io, 'PNG')
    img_io.seek(0)
    
    # Return the image
    return send_file(img_io, mimetype='image/png')
    
from io import BytesIO

from flask import send_file


@app.route("/segmentation", methods=["POST"])
def seg_over():
    try:
        # Debug: Print what we received
        print("Request files:", request.files)
        print("Request form:", request.form)
        print("Content-Type:", request.content_type)
        
        if 'image' not in request.files:
            return jsonify({"error": "No 'image' file in request. Make sure you're sending a file with key 'image'"}), 400
        
        image_data = request.files["image"]
        
        if image_data.filename == '':
            return jsonify({"error": "No file selected"}), 400
            
        image = Image.open(image_data).resize((224, 224))
    image_np = np.array(image)

    # Log to check if image is being processed
    print("Image received and resized:", image.size)

    model_path = r"C:\Users\Rayyan\Desktop\personal\New folder\BreastCancer-main\BreastCancer-main\backend\models\model_segmentation2.keras"
    with CustomObjectScope({"dice_coef": dice_coef, "dice_loss": dice_loss, "f1sc": f1sc}):
        model = tf.keras.models.load_model(model_path)

    x = image_np / 255.0
    x = np.expand_dims(x, axis=0)
    y_pred = model.predict(x, verbose=0)[0]
    y_pred = np.squeeze(y_pred, axis=-1)
    y_pred = y_pred >= 0.5
    y_pred = y_pred.astype(np.int32)

    # Create a mask
    mask = y_pred * 255

    # Log the mask's size
    print("Segmentation mask created:", mask.shape)

    # Create the colored mask and overlay
    mask_colored = np.zeros_like(image_np)
    mask_colored[:, :, 2] = mask  # Apply mask to Red channel
    overlay = cv2.addWeighted(image_np, 0.7, mask_colored, 0.3, 0)

    # Convert to PIL Image and return
    pred_image = Image.fromarray(overlay.astype(np.uint8))

    # Save to in-memory file
    img_io = BytesIO()
    pred_image.save(img_io, 'PNG')
    img_io.seek(0)

    # Log the image size before sending
    print("Returning image with size:", pred_image.size)

    # Return the image as response
    return send_file(img_io, mimetype='image/png')

from flask import send_from_directory


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify')
def classify():
    return render_template('classify.html')

@app.route('/segment')
def segment():
    return render_template('segment.html')

if(__name__=="__main__"):
    app.run(host="0.0.0.0",port=4000,debug=True)