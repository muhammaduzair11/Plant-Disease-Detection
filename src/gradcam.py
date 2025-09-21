import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load model
model = tf.keras.models.load_model("plant_model.h5")


def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:  # Conv feature maps
            return layer.name
    raise ValueError("⚠️ No convolutional layer found!")


def get_gradcam(img_path, model, layer_name=None, save_path="gradcam.jpg"):
    if layer_name is None:
        layer_name = get_last_conv_layer(model)

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (224, 224))
    x = np.expand_dims(img_resized, axis=0)
    x = preprocess_input(x)

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))

    cam = np.dot(conv_outputs[0], weights.numpy())
    cam = cv2.resize(cam, (224, 224))
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()

    heatmap = (cam * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed = cv2.addWeighted(img_resized, 0.6, heatmap, 0.4, 0)

    # Save & show
    cv2.imwrite(save_path, cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR))
    print(f"✅ Grad-CAM saved at {save_path}")

    plt.imshow(superimposed)
    plt.axis("off")
    plt.show()

    return superimposed, predictions.numpy()


# Example usage
if __name__ == "__main__":
    grad_img, preds = get_gradcam("src/test.jpg", model)
