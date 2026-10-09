# SignVision AI — GitHub + Vercel (Browser Inference)

SignVision AI recognizes static hand signs from a webcam using **MediaPipe Hand Landmarker** and your trained **CNN converted to TensorFlow.js**. The live camera frames and CNN inference run in the visitor's browser; this deployment does not need a Python inference server.

> **Browser model included.** This package contains `models/tfjs_model/model.json` and `group1-shard1of1.bin`, plus the A–Z label file. The MediaPipe module is dynamically loaded so initialization errors are shown in the page rather than preventing all controls from wiring up.

## Project layout

```text
SignVisionAI_GitHub_Vercel/
├── index.html
├── static/
│   ├── app.js
│   └── style.css
├── models/
│   ├── class_names.json
│   └── tfjs_model/       # Add converted model.json + group*-shard*.bin
├── tools/
│   └── export_inference_model.py
├── MODEL_EXPORT_GUIDE.md
├── vercel.json
├── .gitignore
└── README.md
```

## Before deployment: add the TensorFlow.js model

1. In Google Drive open `SignLanguage/TrainedModel/` and make sure `hand_sign_cnn.keras` and `class_names.json` are there.
2. The supplied TensorFlow.js model files are already placed under `models/tfjs_model/`. Keep `model.json` and every referenced weight shard together.
3. Confirm `models/class_names.json` matches the exact class order used during training. The supplied file lists A–Z.
5. Ensure no dataset ZIP, training images, or original `.keras` file is in the repo. The converted model files and the class-label JSON are needed for inference.

TensorFlow.js loads a Layers model using `tf.loadLayersModel('/models/tfjs_model/model.json')`; the JSON references its shard files. See the official conversion guide: <https://www.tensorflow.org/js/tutorials/conversion/import_keras>.

## Test locally before pushing

Because webcam access normally requires HTTPS or a loopback origin, do not just double-click `index.html`. With Python installed, open a terminal in the project root and run:

```bash
py -m http.server 8000
```

Open <http://localhost:8000>. Allow camera access. Use Chrome or Edge and make sure the browser can access the internet to fetch TensorFlow.js and MediaPipe assets.

If the model files haven't been exported yet, the page will show a model setup warning; the camera prediction button will stay unavailable.

## Upload to GitHub

Create a new **empty** repository on GitHub, for example `SignVisionAI`. Then from a terminal opened at the extracted project root run:

```bash
git init
git add .
git status
git commit -m "Prepare SignVision AI for Vercel"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/SignVisionAI.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username. Before committing, inspect `git status`: **do not add `Dataset.zip`, raw datasets, `.venv`, or the original `.keras` file.** The small converted browser model files should be included so Vercel can serve them with the site.

## Deploy on Vercel

1. Sign in at <https://vercel.com/> and choose **Add New → Project**.
2. Import your `SignVisionAI` GitHub repository.
3. Select **Other** as the framework preset if Vercel asks.
4. Leave Build Command empty (no build step is required) and deploy the repository root as a static site.
5. After deployment, open the HTTPS URL and allow camera access.

The included `vercel.json` adds basic response headers. Vercel serves the page and model assets; no Python server is started on Vercel. Static Vercel deployments are documented at <https://vercel.com/docs/frameworks/frontend>.

## Features

- Start/stop webcam and choose a camera
- MediaPipe hand landmarks and a padded hand crop
- Client-side TensorFlow.js CNN inference
- Recent-frame voting to reduce flickering
- Letter + confidence display
- Manual letter insertion, spaces, backspace, clear, copy and browser text-to-speech

## Limitations

- The classifier recognizes **static A–Z classes**, not complete sign-language sentences or motion-based signs.
- A confidence/softmax score is not a calibrated probability of correctness.
- Browser inference is a different image pipeline from the original dataset; real webcam accuracy must be measured separately.
- The MediaPipe WASM/model and TensorFlow.js runtime are fetched from public CDNs at runtime, so the first load requires internet access.
- The source Keras model is not directly loaded by the browser. It must be converted first as described in `MODEL_EXPORT_GUIDE.md`.

## Privacy

Camera frames are processed locally by the browser code in this repository. They are not sent to a prediction API. The page does fetch JavaScript/WASM and the CNN model files from the configured public hosting/CDNs.
