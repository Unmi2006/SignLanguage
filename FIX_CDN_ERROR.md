# Fix: MediaPipe JS could not load from jsDelivr

The app no longer imports the generated jsDelivr `+esm` endpoint. `static/app.js` now tries the published `vision_bundle.mjs` bundle from jsDelivr and falls back to unpkg, using the same CDN's WASM fileset.

## Apply the fix to your existing GitHub repository

1. Extract this ZIP.
2. In your current repository, replace only `static/app.js` with the copy from this folder. Keep your `.git` folder and your actual model files.
3. Confirm these files are present in the repository:
   - `models/class_names.json`
   - `models/tfjs_model/model.json`
   - `models/tfjs_model/group1-shard1of1.bin` (and any additional shards named in `model.json`)
4. From the root of your existing repository, run:

```bash
git add static/app.js
git commit -m "Fix MediaPipe browser module loading"
git push
```

5. Wait for Vercel to deploy. In Chrome, open the site and press `Ctrl+Shift+R`.
6. Check that the warning is gone and click **Start camera**. Grant camera permission.

If both CDNs are blocked on the network, use the browser console's latest error; the next robust option is to vendor the MediaPipe bundle and WASM assets into the repository rather than relying on external CDNs.

The MediaPipe module fix is independent of the CNN model files. Keep the `.bin` weight file in the same directory as `model.json` and preserve the exact shard filename referenced by `weightsManifest`.
