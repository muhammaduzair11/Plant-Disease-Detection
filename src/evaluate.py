import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)
import json

# Load model & class indices
model = tf.keras.models.load_model("plant_model.h5")
with open("class_indices.json") as f:
    class_indices = json.load(f)
idx_to_class = {v: k for k, v in class_indices.items()}
class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

# Validation generator
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATA_DIR = "dataset/tomato"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    validation_split=0.2,
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset="validation",
    shuffle=False,
)

# Predictions
preds = model.predict(val_gen)
y_pred = np.argmax(preds, axis=1)
y_true = val_gen.classes

# --- Confusion Matrix ---
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

np.save("confusion_matrix.npy", cm)

# --- Classification Report ---
report = classification_report(
    y_true, y_pred, target_names=class_names, output_dict=True
)
with open("report.json", "w") as f:
    json.dump(report, f, indent=4)

# --- Bar Plot of F1 scores ---
f1_scores = [report[c]["f1-score"] for c in class_names]

plt.figure(figsize=(10, 6))
sns.barplot(x=class_names, y=f1_scores, color="skyblue")
plt.xticks(rotation=45, ha="right")
plt.ylabel("F1 Score")
plt.title("Per-Class F1 Scores")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("f1_scores.png")
plt.close()

# --- Overall Evaluation JSON ---
evaluation = {
    "accuracy": float(report["accuracy"]),
    "macro avg": report["macro avg"],
    "weighted avg": report["weighted avg"],
}
with open("evaluation.json", "w") as f:
    json.dump(evaluation, f, indent=4)

print("✅ Evaluation complete. Saved:")
print(" - confusion_matrix.npy + confusion_matrix.png")
print(" - report.json (full classification report)")
print(" - f1_scores.png (per-class F1 bar chart)")
print(" - evaluation.json (summary metrics)")
