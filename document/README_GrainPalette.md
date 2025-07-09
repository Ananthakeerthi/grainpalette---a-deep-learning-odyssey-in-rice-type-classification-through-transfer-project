
# GrainPalette: A Deep Learning Odyssey in Rice Type Classification through Transfer Learning

## 🌾 Project Overview
GrainPalette is a deep learning-based application designed to classify different types of rice grains using transfer learning. It leverages the power of pre-trained Convolutional Neural Networks (CNNs) to automate and improve the accuracy of rice variety identification.

## 🚀 Objective
To build a scalable, accurate, and automated system that classifies rice grain images using deep learning models with transfer learning.

## 📂 Dataset
- Multiple rice grain types (e.g., Basmati, Sona Masoori, Jasmine)
- ~1000 images per class
- Preprocessing: resizing, normalization, data augmentation

## 🧠 Methodology
1. **Data Collection & Preprocessing**
2. **Model Selection**: Used pre-trained models like ResNet50 and VGG16
3. **Transfer Learning**: Fine-tuning the top layers of the model
4. **Training & Validation**: Accuracy ~95%
5. **Evaluation**: Confusion matrix, precision, recall, F1-score

## 🏗️ Model Architecture
- Base Model: Pre-trained CNN (ResNet50)
- Added Layers: GlobalAveragePooling, Dense, Dropout, Softmax
- Loss Function: Categorical Crossentropy
- Optimizer: Adam

## 📈 Results
- Accuracy: ~95%
- High generalization across all rice varieties
- Outperformed traditional ML methods (~75% accuracy)

## 🛠️ Technologies Used
- Python
- TensorFlow / Keras
- NumPy, OpenCV, Matplotlib
- Jupyter Notebook / Google Colab

## 📦 Deployment
- Can be deployed as a mobile/web app
- Compatible with TensorFlow Lite or ONNX for optimization

## 📌 Future Work
- Expand dataset with more rice types
- Deploy on edge devices for real-time use
- Add image upload interface for farmers and industries

## 📚 References
- TensorFlow & Keras Documentation
- ImageNet Pre-trained Models
- Kaggle/OpenRice dataset

---

© 2025 GrainPalette Project Team
