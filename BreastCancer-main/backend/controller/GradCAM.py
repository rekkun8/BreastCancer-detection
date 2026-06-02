import os
import uuid

import cv2
import numpy as np
import tensorflow as tf


def get_gradcam_heatmap(model, img_array, layer_name='top_conv'):
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    img_tensor = tf.ensure_shape(img_tensor, [1,224,224,3])
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        tape.watch(img_array)
        conv_outputs, predictions = grad_model(img_array)
        predicted_class = tf.argmax(predictions[0])
        loss = predictions[:, predicted_class]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise ValueError("Gradients are None - check input strusture or GradientTape setup.")
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0) / tf.math.reduce_max(heatmap + 1e-10)
    return heatmap.numpy()

def overlay_gradcam(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224,224))

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    overlayed = heatmap_color * alpha + img
    overlayed = np.uint8(overlayed)
    unique_name = f"gradcam_{uuid.uuid4().hex[:8]}.jpg"
    static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
    os.makedirs(static_dir, exist_ok=True)
    static_path = os.path.join("static", unique_name)

    cv2.imwrite(static_path, overlayed)
    return static_path