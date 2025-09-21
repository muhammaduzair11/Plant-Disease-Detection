🌱 Plant Disease Detection with Deep Learning

📌 Overview

This project implements a Plant Disease Detection System using MobileNetV2 (Transfer Learning) on the PlantVillage Tomato dataset.
The model classifies tomato leaf images into multiple disease categories and healthy leaves.

To enhance interpretability, we integrated Grad-CAM for visual explanations.
We also built an interactive Streamlit dashboard for real-time predictions, performance visualization, and explainability.

✨ Features

✅ Train deep learning model (MobileNetV2) on PlantVillage dataset

✅ Handle class imbalance with class weights

✅ Evaluate with multiple metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix

✅ Visualize Grad-CAM heatmaps for explainability

✅ Interactive Streamlit App with:

Upload leaf image & get predictions

Probability distribution bar chart

Grad-CAM visualization

Training metrics (Accuracy/Loss curves)

Confusion Matrix & Classification Report

📂 Project Structure
├── dataset/                # Dataset folder (⚠️ not included due to size)
│   └── tomato/             # Subfolders per class
├── plant_model.h5          # Saved trained model
├── class_indices.json      # Mapping of class indices to labels
├── metrics.json            # Training history (accuracy/loss)
├── report.json             # Classification report
├── confusion_matrix.npy    # Confusion matrix
├── src/
│   ├── train.py            # Train the model
│   ├── predict.py          # Test a single image
│   ├── evaluate.py         # Evaluate model & save metrics
│   ├── gradcam.py          # Grad-CAM visualizer
│   └── app.py              # Streamlit web app

⚙️ Environment Setup

This project uses Python 3.9+ with the following dependencies:

pip install -r requirements.txt

requirements.txt
tensorflow==2.12.0
opencv-python
numpy
matplotlib
seaborn
pandas
scikit-learn
streamlit
Pillow

📊 Dataset

Source: PlantVillage Dataset (Tomato subset).

Contains 10 classes:

Tomato_Bacterial_spot

Tomato_Early_blight

Tomato_Late_blight

Tomato_Leaf_Mold

Tomato_Septoria_leaf_spot

Tomato_Spider_mites_Two_spotted_spider_mite

Tomato_Target_Spot

Tomato_YellowLeaf_Curl_Virus

Tomato_mosaic_virus

Tomato_healthy

⚠️ Note: The dataset is not included in this repository due to size.
You can download it from PlantVillage Dataset
 or manually place it under dataset/tomato/.

🚀 Usage
1️⃣ Train the Model
python src/train.py

2️⃣ Evaluate the Model
python src/evaluate.py

3️⃣ Predict on a Single Image
python src/predict.py

4️⃣ Visualize Grad-CAM
python src/gradcam.py

5️⃣ Run the Streamlit App
streamlit run src/app.py

📈 Results

Accuracy: ~87% on validation set

Precision/Recall/F1-score: Balanced across most classes

Confusion Matrix & Grad-CAM: Show model focus regions and class-wise performance

(Insert screenshots here: training curves, confusion matrix, Grad-CAM examples, Streamlit app demo)

🔮 Future Work

Expand dataset with real-world field images

Deploy as a mobile app for farmers

Add support for more crops

Integrate with IoT devices/drones for large-scale monitoring

📜 License

This project is open-source under the MIT License.

🙌 Acknowledgements

PlantVillage dataset

TensorFlow / Keras

Streamlit

Grad-CAM (Selvaraju et al., 2017)
