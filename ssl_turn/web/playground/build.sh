#!/bin/sh
# Assemble the static Space in $1: page files, turn_stream.js, onnxruntime-web and the models.
#   sh build.sh OUT ORT_DIST MODELS     (ORT_DIST = node_modules/onnxruntime-web/dist, 1.22.0)
set -e
here=$(dirname "$0"); out=$1; ort=$2; models=$3
mkdir -p "$out/ort" "$out/models"
cp "$here"/index.html "$here"/app.js "$here"/worker.js "$here"/capture.js "$here"/style.css "$here"/README.md "$out/"
cp "$here"/../js/turn_stream.js "$out/"
cp "$ort"/ort.webgpu.bundle.min.mjs "$ort"/ort-wasm-simd-threaded.jsep.wasm "$ort"/ort-wasm-simd-threaded.jsep.mjs "$out/ort/"
for f in encoder_k2_w16.onnx encoder_k4_w16.onnx head.onnx; do cp "$models/$f" "$out/models/"; done
