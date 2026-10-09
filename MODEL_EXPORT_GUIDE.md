# Export your saved Keras CNN for the Vercel browser app

The current local Windows app can load `hand_sign_cnn.keras` with Python, but a static Vercel site cannot load a Python `.keras` file directly. TensorFlow.js browser inference expects `model.json` plus binary weight shards. This guide exports an inference-only model and converts it.

## Part A — Export an inference-only HDF5 model from Colab

Use the existing training notebook/runtime where TensorFlow can load your model. Run these cells there; this **does not retrain** the CNN.

### Cell 1 — Mount Drive and load your model

```python
from google.colab import drive
drive.mount('/content/drive')

import json
import tensorflow as tf
from pathlib import Path

model_path = '/content/drive/MyDrive/SignLanguage/TrainedModel/hand_sign_cnn.keras'
labels_path = '/content/drive/MyDrive/SignLanguage/TrainedModel/class_names.json'

trained_model = tf.keras.models.load_model(model_path, compile=False)
with open(labels_path, 'r', encoding='utf-8') as f:
    class_names = json.load(f)

print('Input:', trained_model.input_shape)
print('Output:', trained_model.output_shape)
print('Labels:', class_names)
print('Layers:', [layer.name for layer in trained_model.layers])
```

### Cell 2 — Remove training-only augmentation and export

The training model includes random augmentation layers. For stable browser inference we skip the augmentation submodel and preserve the `Rescaling` layer and all learned layers after it. The input should remain float pixels in the 0–255 range, matching the original webcam pipeline.

```python
import shutil
from pathlib import Path

rescale_index = next(
    i for i, layer in enumerate(trained_model.layers)
    if isinstance(layer, tf.keras.layers.Rescaling)
)

inputs = tf.keras.Input(shape=(128, 128, 3), name='image')
x = inputs
for layer in trained_model.layers[rescale_index:]:
    x = layer(x)

inference_model = tf.keras.Model(inputs, x, name='signvision_inference')
inference_model.save('/content/signvision_inference.h5', include_optimizer=False)

# Save the exact label order alongside the HDF5 export.
with open('/content/class_names.json', 'w', encoding='utf-8') as f:
    json.dump(class_names, f, indent=2)

print('Exported inference model:', '/content/signvision_inference.h5')
print('Input/output:', inference_model.input_shape, inference_model.output_shape)
```

Download `signvision_inference.h5` and `class_names.json` from the Colab Files pane (right-click each file → Download), or copy them to a Drive folder and download from Drive.

## Part B — Convert HDF5 to TensorFlow.js on Windows

TensorFlow.js's Python converter has had dependency issues in some Python 3.13 environments, so use **Python 3.11** for this conversion and keep it separate from your local app virtual environment.

1. Copy `signvision_inference.h5` into the repo's `models/` folder temporarily.
2. Open Command Prompt or PowerShell in the repository root.
3. Run:

```powershell
py -3.11 -m venv .venv-tfjs
.\.venv-tfjs\Scripts\python.exe -m pip install --upgrade pip
.\.venv-tfjs\Scripts\python.exe -m pip install tensorflowjs==4.22.0
.\.venv-tfjs\Scripts\tensorflowjs_converter.exe --input_format=keras --output_format=tfjs_layers_model models\signvision_inference.h5 models\tfjs_model
```

If the `tensorflowjs_converter.exe` command isn't created in your venv, run:

```powershell
.\.venv-tfjs\Scripts\python.exe -m tensorflowjs.converters.converter --input_format=keras --output_format=tfjs_layers_model models\signvision_inference.h5 models\tfjs_model
```

Converter packages have known version/dependency sensitivity; if installation or conversion fails, copy the complete error output before changing versions. The original `.keras` model remains safe, and this process does not train the model again.

## Part C — Put the web model in the right location

The converter output should look similar to:

```text
models/
├── class_names.json
└── tfjs_model/
    ├── model.json
    └── group1-shard1of1.bin
```

There may be more than one `group*-shard*.bin` file. Keep **all** shards in the folder and keep their filenames unchanged. Copy the original `class_names.json` generated in Colab to `models/class_names.json`, replacing the example labels.

Remove `models/tfjs_model/README_TO_REPLACE.txt`. Do not commit `signvision_inference.h5`, the original `.keras` file, or the dataset archive. The `model.json`, binary shards, and `class_names.json` are the browser inference assets needed by Vercel.

Finally, follow the GitHub and Vercel steps in `README.md`.
