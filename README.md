# 🧠 Alz_dec — Alzheimer's MRI Classifier

An AI-powered Streamlit app for detecting Alzheimer’s Disease using MRI brain scans.

This web application allows users to upload an MRI scan and receive a predicted Alzheimer’s diagnosis based on a fine-tuned **VGG16** deep learning model. It also provides interpretability via **Grad-CAM** heatmaps to help users understand what regions the model focused on during prediction.

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
- ✅ Display **confidence scores** for each class
- ✅ Generate **Grad-CAM heatmaps** highlighting important regions
- ✅ Detect **non-MRI or invalid images** and give warning prompts
- ✅ Educate users on **VGG16**, **Grad-CAM**, and Alzheimer's stages

---

## 🧠 Model Overview

This application uses a **VGG16** model pre-trained on ImageNet, then fine-tuned on a curated Alzheimer's MRI dataset. The model learns to distinguish brain patterns that correspond to various stages of dementia.

### 🔍 Resources
- [VGG16 Paper (Simonyan & Zisserman, 2014)](https://arxiv.org/abs/1409.1556)
- [Keras VGG16 Documentation](https://keras.io/api/applications/vgg/#vgg16-function)

---

## 🎯 Grad-CAM Visual Explanation

**Grad-CAM** (Selvaraju et al., 2017) is used to visualize regions of the MRI that most influenced the model's decision. 
- 🔴 **Red** = strong activation
- 🔵 **Blue** = low attention

> This adds transparency to the AI's decisions, which is vital in healthcare applications.

- [Grad-CAM Paper](https://arxiv.org/abs/1610.02391)

<p align="center">
  <img src="[images/sample_gradcam.png](https://raw.githubusercontent.com/octopuspaul110/Alz_dec/refs/heads/main/conclusion_images/download%20(1).png)" alt="Grad-CAM Example" width="500"/>
</p>

---

## 📊 Performance Summary

We compared three models: a **Custom CNN**, **SCCAN**, and **VGG16**. Results show a clear advantage in using transfer learning and attention mechanisms.

| Model         | Accuracy | Loss  | Very Mild Recall | Moderate Precision | Remarks |
|---------------|----------|-------|------------------|---------------------|---------|
| Custom CNN    | 72.7%    | 0.85  | 0.32             | 1.00                | Struggles with early detection |
| SCCAN         | 95.7%    | 0.14  | >0.93            | 1.00                | Excellent across all metrics |
| VGG16 (FT)    | 95.7%    | 0.14  | >0.93            | 1.00                | Matches SCCAN performance |

### Confusion Matrix Comparison

<p align="center">
  <img src="images/confusion_matrices.png" alt="Confusion Matrices" width="600"/>
</p>

---

## ✅ Conclusion

The results demonstrate that **VGG16** and **SCCAN** are both highly effective in diagnosing Alzheimer’s stages from MRI scans, especially early-stage detection like *Very Mild Dementia*. 

Key takeaways:

- 🔍 **Custom CNN** lacked the discriminative power to accurately classify subtle stages.
- 🧠 **SCCAN** and **VGG16** performed exceptionally well, both exceeding **95% accuracy**.
- 🎯 **Grad-CAM** helps clinicians understand what the model "sees", improving trust in AI systems.
- 🏥 These results bring us one step closer to **AI-assisted clinical diagnosis** in real-world medical environments.

---
