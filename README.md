# Petals to the Metal TFRecord JAX Pipeline

A Kaggle competition project designed to classify 104 different classes of flowers on Google's bleeding-edge TPU v5e-8 computational hardware. This repository holds the infrastructure code that parses TFRecords at ultra-high speed and maps it directly across an 8-core TPU utilizing DataParallelism and JAX JIT compilation.

## Architecture & Framework Pivot
Historically, TensorFlow C++ `OpKernels` natively bound to earlier versions of TPU VMs via `ConfigureDistributedTPU`. On Kaggle's latest **TPU v5e-8** architectures, Google has officially prioritized JAX/PyTorch bindings, often leaving TensorFlow cluster initialization heavily prone to segmentation faults and OpKernel crashes.

**The Solution:** 
This pipeline leverages **Keras 3** by forcibly swapping the execution backend algorithm from `TensorFlow` to `JAX` via environment variables.
- `tf.data.Dataset` handles all I/O bottleneck parsing.
- Keras 3 bridges the dataset seamlessly into the neural model.
- JAX executes `.fit()` and handles native 8-Device Sharding natively using `keras.distribution.DataParallel`.

## Features
- **Algorithm Foundation:** MobileNetV3 / EfficientNetB7 Backbone Options.
- **Dataset I/O Optimization:** Ultra-fast TFRecord streaming parallelized using `AUTOTUNE`.
- **Dynamic Padding Inference:** Prevents inference-time JIT compilation crashes by guaranteeing `batch_size % NumReplicas == 0` for trailing batches.
- **Robust LR Scheduler:** Configured specifically for high-batch global scale (128 Batch per Step) incorporating Linear Warmup followed by continuous Exponential Decay.
- **Heavy Augmentations:** Hardcoded mapping implementations for Left/Right Flips, Saturation, Contrast, and Brightness dynamically injected via `tf.image`.

## Repository Structure 
- `src/` : Local fallback and test routines.
- `kaggle_submit/` : Cloud-Ready stand-alone Python scripts for direct execution.
    - `petals_test_run.py`: The finalized Version 10 Kaggle runtime code.
- `docs/` : Performance and architectural reports detailing stability scaling.
- `requirements.txt`: Virtual Environment pip bindings.

## Kaggle Deployment
To deploy to the TPU Cloud:
```bash
cd kaggle_submit
kaggle kernels push
```
*Note: Ensure your competition API mappings specifically target the Kaggle backend. If standard API calls fail, the recursive diagnostic fallback hardcoded into `petals_test_run.py` will physically locate the TFRecord mounts under `/kaggle/input/competitions/tpu-getting-started`.*
