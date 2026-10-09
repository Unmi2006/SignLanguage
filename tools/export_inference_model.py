"""Export an inference-only version of the trained CNN; run in the Colab runtime that has TensorFlow and the saved model available."""
import json
import tensorflow as tf

MODEL_PATH = "/content/drive/MyDrive/SignLanguage/TrainedModel/hand_sign_cnn.keras"
LABELS_PATH = "/content/drive/MyDrive/SignLanguage/TrainedModel/class_names.json"
OUTPUT_MODEL = "/content/signvision_inference.h5"
OUTPUT_LABELS = "/content/class_names.json"

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels = json.load(f)
rescale_index = next(i for i, layer in enumerate(model.layers) if isinstance(layer, tf.keras.layers.Rescaling))
inputs = tf.keras.Input(shape=(128, 128, 3), name="image")
x = inputs
for layer in model.layers[rescale_index:]:
    x = layer(x)
inference_model = tf.keras.Model(inputs, x, name="signvision_inference")
inference_model.save(OUTPUT_MODEL, include_optimizer=False)
with open(OUTPUT_LABELS, "w", encoding="utf-8") as f:
    json.dump(labels, f, indent=2)
print("Exported:", OUTPUT_MODEL, "Output shape:", inference_model.output_shape)
