import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
from PIL import Image
import io

st.set_page_config(page_title="Alzheimer's MRI Classifier", layout="centered")


# Load model
@st.cache_resource
def load_vgg16_model():
    return load_model("model\model_VGG16.keras")

# Load the model
model = load_vgg16_model()

class_names = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

def preprocess_image(image_file):
    image = Image.open(image_file).convert('RGB')
    image = image.resize((224, 224))
    img_array = img_to_array(image)
    img_array = tf.keras.applications.resnet.preprocess_input(img_array)
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0),image
# Grad-CAM utility
def make_gradcam_heatmap(img_array, model, last_conv_layer_name='block5_conv3', pred_index=None):
    vgg16_model = model.layers[0]
    conv_layer = vgg16_model.get_layer(last_conv_layer_name)
    conv_model = tf.keras.models.Model(inputs=vgg16_model.input, outputs=conv_layer.output)

    inputs = tf.cast(img_array, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        conv_outputs = conv_model(inputs)
        x = conv_outputs
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = model.layers[2](x)
        x = model.layers[3](x)
        x = model.layers[4](x)
        x = model.layers[5](x)
        x = model.layers[6](x)
        preds = model.layers[7](x)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def is_probably_mri(image: Image.Image, grayscale_threshold=0.95, edge_density_threshold=0.02) -> bool:
    """
    Heuristic-based MRI detection:
    - MRI scans are predominantly grayscale.
    - They contain significant structural (edge) detail in brain regions.
    
    Args:
        image (PIL.Image): Uploaded image.
        grayscale_threshold (float): Threshold for deciding if image is grayscale-like.
        edge_density_threshold (float): Threshold for deciding if image has sufficient edge detail.

    Returns:
        bool: True if image likely resembles a brain MRI, False otherwise.
    """
    try:
        # Convert to RGB and resize
        image = image.convert("RGB").resize((224, 224))
        image_np = np.array(image)

        # Check for grayscale-like images (i.e., R ≈ G ≈ B)
        diff_rg = np.abs(image_np[:, :, 0] - image_np[:, :, 1])
        diff_gb = np.abs(image_np[:, :, 1] - image_np[:, :, 2])
        diff_rb = np.abs(image_np[:, :, 0] - image_np[:, :, 2])
        grayscale_pixels = np.logical_and.reduce([
            diff_rg < 15,
            diff_gb < 15,
            diff_rb < 15
        ])
        grayscale_ratio = np.sum(grayscale_pixels) / (224 * 224)

        # Check for edge detail (MRI typically has brain structure)
        gray_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray_img, 30, 100)
        edge_density = np.sum(edges > 0) / (224 * 224)

        # Combine checks
        is_grayscale_like = grayscale_ratio >= grayscale_threshold
        has_enough_edges = edge_density >= edge_density_threshold

        print("non excepter")
        return is_grayscale_like and has_enough_edges

    except Exception as e:
        print("Error in MRI detection:", e)
        return False


def display_gradcam(image_pil, heatmap, alpha=0.4):
    image_np = np.array(image_pil)
    heatmap_resized = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    superimposed_img = heatmap_colored * alpha + image_np
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    return Image.fromarray(superimposed_img)


# Streamlit UI

st.title("🧠 Alzheimer's MRI Classification using VGG16")
st.title("Alzheimer's MRI Classification")

with st.expander("ℹ About the Model (VGG16)", expanded=False):
    st.markdown("""
*VGG16* is a deep convolutional neural network introduced by Simonyan and Zisserman in [Very Deep Convolutional Networks for Large-Scale Image Recognition (2014)](https://arxiv.org/abs/1409.1556). It has 16 layers with weights, composed of:

- *13 convolutional layers* using small 3×3 filters
- *3 fully connected layers*
- *ReLU activations* and *MaxPooling layers*

We use a *fine-tuned VGG16 model* pretrained on ImageNet, adapted for Alzheimer's classification. Transfer learning enables the model to repurpose visual knowledge from natural images for medical imaging tasks, enhancing generalization on relatively smaller datasets.

> 📘 Simonyan & Zisserman (2014): [arXiv:1409.1556](https://arxiv.org/abs/1409.1556)  
> 📚 Summary: [VGGNet – Papers with Code](https://paperswithcode.com/method/vgg)
""")

uploaded_file = st.file_uploader("Upload an MRI image (JPG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded MRI", use_container_width=True)
    with st.spinner("Analyzing..."):
        
        img_array, img_pil = preprocess_image(uploaded_file)

        if is_probably_mri(img_pil):
            preds = model.predict(img_array)[0]
            pred_idx = np.argmax(preds)
            pred_label = class_names[pred_idx]
            confidence = preds[pred_idx] * 100

            if confidence < 70:
                print("jgjfbgb")
                st.warning("⚠ Image is not an MRI scan. Please ensure the image is a valid and clear MRI scan.")
            else:
                st.subheader("Prediction Result")
                st.success(f"🧠 Prediction: *{pred_label}*")
                st.success(f"   Confidence *{confidence:.2f}%*")            

                # Confidence for all classes
                conf_dict = {class_names[i]: round(preds[i] * 100, 2) for i in range(len(class_names))}
                conf_df = pd.DataFrame(list(conf_dict.items()), columns=["Class", "Confidence (%)"])
                st.subheader("🔍 Confidence Scores for All Classes")
                st.dataframe(conf_df.set_index("Class"))

                with st.expander("🧠 How the Model Explains Itself (Grad-CAM)", expanded=False):
                    st.markdown("""
                                *Grad-CAM* (Gradient-weighted Class Activation Mapping) is a visualization technique that helps interpret CNN decisions. It highlights important regions of the input image that strongly influence the model's prediction.

                                Here’s how it works:

                                1. *Backpropagation* is used to compute gradients of the output class score with respect to feature maps in a convolutional layer (e.g., block5_conv3 in VGG16).
                                2. These gradients are *globally averaged* to obtain weights.
                                3. A *heatmap* is created by weighing the feature maps with these values and projecting them onto the input space.

                                ### 🔥 Interpreting the Heatmap

                                - *Red / Yellow areas*: High influence – the model focused here for its decision.
                                - *Green areas*: Moderate influence.
                                - *Blue / Dark areas*: Low influence – the model found them less relevant.

                                This helps users *see what the model sees*, improving trust and transparency in medical AI systems.

                                > 📘 Selvaraju et al. (2017): [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391)
                                """)
                    st.subheader("Grad-CAM Visualization")
                    
                    # Grad-CAM
                    heatmap = make_gradcam_heatmap(img_array, model, pred_index=pred_idx)
                    gradcam_img = display_gradcam(img_pil, heatmap)
                    st.image(gradcam_img, caption="Grad-CAM Visualization (showing regions influencing the prediction)", use_container_width=True)
        else:
            st.warning("⚠ Image is not an MRI scan. Please ensure the image is a valid and clear MRI scan.")