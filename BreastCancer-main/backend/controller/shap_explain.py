import os

import matplotlib.pyplot as plt
import numpy as np
import shap
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input


def get_shap_visual(model, image_path, output_path=None):
    if output_path is None:
        # Use absolute path to static folder
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_path = os.path.join(base_dir, "static", "shap_output.jpg")
    
    image = Image.open(image_path).resize((224,224))
    image_np = np.array(image)
    image_batch = preprocess_input(np.expand_dims(image_np, axis=0))

    background = np.repeat(image_batch, 10, axis=0)

    explainer = shap.GradientExplainer((model.input, model.output), background)
    shap_values = explainer.shap_values(image_batch)

    shap.image_plot(shap_values, image_batch, show=False)
    plt.savefig(output_path)
    plt.close()

    return output_path
