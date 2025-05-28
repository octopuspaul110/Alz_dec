# Alz_dec (🧠 Alzheimer's MRI Classifier)
An AI app built to detect alzeimers diseases using mri images.

This Streamlit web application allows users to upload an MRI brain scan and receive a predicted Alzheimer's diagnosis based on a fine-tuned *VGG16* deep learning model. It also visualizes the model's attention using *Grad-CAM* to help users understand the prediction.

<p align="center">
  <img src="https://streamlit.io/images/brand/streamlit-logo-primary-colormark-darktext.png" alt="Streamlit" width="200"/>
</p>

---

## 🚀 Features

- ✅ Upload an MRI scan and classify it into:
  - NonDemented
  - VeryMildDemented
  - MildDemented
  - ModerateDemented
- ✅ Shows *confidence scores* for each class
- ✅ Generates a *Grad-CAM heatmap* highlighting important regions
- ✅ Detects non-MRI or invalid images with warning prompts
- ✅ Provides educational insights about *VGG16* and *Grad-CAM*

---

## 🧠 Model Overview

This application uses a *VGG16* model pretrained on ImageNet, then fine-tuned on an Alzheimer's MRI dataset. The model learns to distinguish patterns in brain tissue that correspond to various dementia stages.

For more on VGG16:
- [Original Paper (Simonyan & Zisserman, 2014)](https://arxiv.org/abs/1409.1556)
- [Keras VGG16 Documentation](https://keras.io/api/applications/vgg/#vgg16-function)

### 🎯 Grad-CAM Visual Explanation

Grad-CAM (Selvaraju et al., 2017) highlights parts of the MRI that most influenced the model’s decision. Red regions indicate *strong activation* (high relevance), while blue shows *low attention*.

More:
- [Grad-CAM Paper](https://arxiv.org/abs/1610.02391)
- [Keras Grad-CAM Tutorial](https://keras.io/examples/vision/grad_cam/)

---
