# Rebuild the public Space

Model preparation/export happens on a remote CPU Colab VM, never on the host.
Scripts use `/content/webgpu-playground` and a mode-0600 `/content/.hf_token`.
The pinned owner research archives are private; rebuilding from them requires
access. The already deployed browser weights are public in the static Space.

1. On the remote VM install `torch`, `numpy`, `onnx==1.23.2`,
   `onnxruntime==1.30.0`, `silero-vad==6.2.0`, `huggingface_hub`, `pyyaml`,
   `soundfile`, `scipy`; install `espeak` for the generated spoken example.
2. Run `prepare_remote.py` remotely. It retrieves pinned artifacts and verifies
   the original model/head hashes. Upload the parent `heads.py` to `/content/heads.py`.
3. Run `export_assets.py` remotely. It exports frontend constants, head, VAD and
   synthetic CPU-reference fixtures, and records asset hashes in the manifest.
4. Install `onnxruntime-web@1.30.0` with npm in a temporary source-only directory.
   Run `python package_site.py --runtime-dir /path/to/node_modules/onnxruntime-web --out /tmp/playground-site.tar.gz`.
   This archive contains source/runtime/licenses only, not local model assets.
5. Upload the tar to `/content/playground-site.tar.gz`, then run
   `deploy_remote.py` remotely. It combines the allowlisted UI with remote model
   assets and publishes `shinohararin/pardon-turn-webgpu`. The deployment receipt
   records exact revision, sizes and SHA256. This publication makes weights public.
6. Verify the actual Space host returned by the HF API, run its synthetic check,
   run the paced example, inspect queue/lag, verify mono/stereo selection and
   stop/restart. Keep raw microphone audio on the visitor's device.
7. Persist the exact published revision and receipt to mounted Drive, then stop
   the export CPU. Preserve separate experiment runtimes while their jobs run.

The Space README is also the HF configuration file. `BUILD.md`, Python scripts
and training artifacts are repository source, not shipped as application assets.
