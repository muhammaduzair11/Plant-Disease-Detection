# train_improved.py (replace train.py or adapt)
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

DATA_DIR = "dataset/tomato"  # <- ensure correct path
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 12

# augment only for training
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2,
)
val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input, validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset="training",
    shuffle=True,
    seed=42,
)
val_gen = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset="validation",
    shuffle=False,
    seed=42,
)

num_classes = train_gen.num_classes
print("Classes:", train_gen.class_indices)
print("Train samples:", train_gen.samples, "Val samples:", val_gen.samples)

# compute class weights
y = train_gen.classes  # integer label per training sample
classes = np.unique(y)
cw = compute_class_weight(class_weight="balanced", classes=classes, y=y)
class_weight = {int(cls): float(weight) for cls, weight in zip(classes, cw)}
print("Class weights:", class_weight)

# build model
base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
base.trainable = False
x = base.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(256, activation="relu")(x)
out = Dense(train_gen.num_classes, activation="softmax")(x)
model = Model(inputs=base.input, outputs=out)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# callbacks
cb = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2),
    tf.keras.callbacks.ModelCheckpoint("plant_model.h5", save_best_only=True),
]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weight,
    callbacks=cb,
)

metrics = {
    "accuracy": history.history["accuracy"],
    "val_accuracy": history.history["val_accuracy"],
    "loss": history.history["loss"],
    "val_loss": history.history["val_loss"],
}
with open("metrics.json", "w") as f:
    json.dump(metrics, f)

# --- Save class indices ---
with open("class_indices.json", "w") as f:
    json.dump(train_gen.class_indices, f)
