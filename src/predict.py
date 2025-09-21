import tensorflow as tf
import numpy as np
import cv2
import json
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load trained model
model = tf.keras.models.load_model("plant_model.h5")

# Load class names from JSON
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Reverse mapping: index -> class name
CLASS_NAMES = {v: k for k, v in class_indices.items()}


def predict_image(img_path):
    # Read & preprocess image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)

    # Get predictions
    preds = model.predict(x)[0]
    top3_idx = preds.argsort()[-3:][::-1]

    results = {}
    for i in top3_idx:
        results[CLASS_NAMES[i]] = float(preds[i]) * 100

    return results


if __name__ == "__main__":
    test_img = "src/test.jpg"  # test image path
    results = predict_image(test_img)
    print("\nPrediction Results:")
    for cls, prob in results.items():
        print(f"{cls}: {prob:.2f}%")
