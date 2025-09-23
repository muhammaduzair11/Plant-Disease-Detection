# **🌱 Plant Disease Detection System**

## **📌 Overview**

This project presents a robust and explainable deep learning system for the detection and classification of diseases in plant leaves. Utilizing a Convolutional Neural Network (CNN) based on the efficient **MobileNetV2** architecture, the model is trained on a subset of the **PlantVillage** dataset, specifically focused on tomato leaf diseases.

To move beyond the "black box" nature of deep learning, we integrated **Gradient-weighted Class Activation Mapping (Grad-CAM)** to provide visual explanations of the model's predictions. The entire system is packaged within an interactive **Streamlit** web application, offering a user-friendly interface for real-time predictions, performance visualization, and model interpretability.

## **✨ Key Features**

- **Deep Learning Model:** Implements a state-of-the-art deep learning model using **MobileNetV2** with transfer learning for high-accuracy classification.
- **Model Interpretability:** Utilizes **Grad-CAM** to generate heatmaps that highlight the specific regions of a leaf image that influence the model's decision.
- **Comprehensive Evaluation:** Evaluates model performance using a full suite of metrics, including **Accuracy, Precision, Recall, F1-score**, and a **Confusion Matrix**.
- **Interactive Web Application:** Deploys a user-friendly **Streamlit** dashboard for:
  - Uploading an image and receiving a real-time disease prediction.
  - Visualizing training metrics (accuracy/loss curves).
  - Displaying the confusion matrix and classification report.
  - Showcasing Grad-CAM heatmaps for explainable predictions.

## **⚙️ Technical Stack**

- **Frameworks:** TensorFlow, Keras, Streamlit
- **Libraries:** NumPy, Matplotlib, Seaborn, scikit-learn, OpenCV, Pillow
- **Model:** MobileNetV2 (Pre-trained on ImageNet)
- **Optimizer:** Adam
- **Dataset:** PlantVillage Dataset (Tomato subset)

## **📂 Project Structure**

.  
├── .venv/ # Python virtual environment  
├── .vscode/ # VS Code configuration  
├── src/ # Source code  
│ ├── train.py # Model training script  
│ ├── predict.py # Single-image prediction script  
│ ├── evaluate.py # Model evaluation script  
│ ├── gradcam.py # Grad-CAM visualization utility  
│ └── app.py # Streamlit web application  
├── plant_model.h5 # Saved trained model weights  
├── class_indices.json # Maps class indices to labels  
├── metrics.json # Training history (accuracy & loss)  
├── report.json # Classification report JSON  
├── confusion_matrix.npy # Raw confusion matrix data  
├── requirements.txt # Project dependencies  
├── README.md # This file  

## **📊 Dataset**

This project uses the publicly available **PlantVillage Dataset (Tomato subset)**. This dataset contains images of tomato leaves classified into 10 categories, including healthy leaves and various diseases.

- **Classes:**
  - Tomato_Bacterial_spot
  - Tomato_Early_blight
  - Tomato_Late_blight
  - Tomato_Leaf_Mold
  - Tomato_Septoria_leaf_spot
  - Tomato_Spider_mites_Two_spotted_spider_mite
  - Tomato_Target_Spot
  - Tomato_YellowLeaf_Curl_Virus
  - Tomato_mosaic_virus
  - Tomato_healthy

⚠️ **Note:** The dataset itself is not included in this repository due to its large size. You can download it from the official [PlantVillage Dataset](https://www.google.com/search?q=https://www.kaggle.com/datasets/plantvillage/plant-village-dataset) page and place it in the dataset/tomato/ directory.

## **🚀 Getting Started**

### **1\. Environment Setup**

Clone the repository and install the required dependencies:

git clone \[<https://github.com/muhammaduzair11/Plant-Disease-Detection.git\>](<https://github.com/muhammaduzair11/Plant-Disease-Detection.git>)  
cd Plant-Disease-Detection  
pip install -r requirements.txt  

### **2\. Usage**

You can run the project components from the src directory:

\# To train the model  
python src/train.py  
<br/>\# To evaluate the model  
python src/evaluate.py  
<br/>\# To make a prediction on a single image  
python src/predict.py  
<br/>\# To visualize Grad-CAM for an image  
python src/gradcam.py  
<br/>\# To run the interactive web application  
streamlit run src/app.py  

## **📈 Results**

The trained MobileNetV2 model achieved a high level of performance. Key metrics on the validation set include:

- **Overall Accuracy:** ~87%
- **Precision, Recall, and F1-score:** The model demonstrated a strong balance of performance across most classes.

## **🔮 Future Work**

- **Dataset Expansion:** Expand the dataset with real-world field images to improve the model's generalization to diverse environmental conditions.
- **Mobile Deployment:** Develop a dedicated mobile application for real-time, on-site disease detection by farmers.
- **IoT Integration:** Integrate the system with drone or IoT devices for large-scale, automated crop monitoring.
- **Advanced Explainability:** Explore and implement more sophisticated explainability techniques beyond Grad-CAM.

## **🙌 Acknowledgements**

- **PlantVillage:** For providing the comprehensive dataset.
- **Selvaraju et al., (2017):** For their seminal work on Grad-CAM.
- **TensorFlow & Keras:** For the deep learning framework.
- **Streamlit:** For the web application framework.
