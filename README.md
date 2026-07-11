# Petals to the Metal TFRecord JAX Pipeline

A Kaggle flower classification project that trains EfficientNetB7 on TPU v5e-8 hardware to classify 104 flower species from TFRecord data.

## Why JAX + Keras 3

TPU v5e-8 is optimized for JAX and PyTorch, not TensorFlow. This pipeline uses Keras 3 with a JAX backend to avoid segmentation faults and OpKernel crashes common in TensorFlow on newer TPU VMs.

**Architecture:**
- `tf.data.Dataset` streams TFRecords with parallel parsing
- Keras 3 (JAX backend) trains the model with `keras.distribution.DataParallel` across 8 cores
- Dynamic padding in batches ensures JAX's strict shape requirements are met during inference

## Features
- EfficientNetB7 model backbone on 512×512 images
- TFRecord streaming with automatic parallelization
- Dynamic batch padding to prevent JIT compilation failures
- Linear warmup + exponential decay learning rate schedule (peak: 0.00024)
- Data augmentation: flips, saturation, contrast, brightness adjustments
- 4-pass test-time augmentation (original, horizontal flip, vertical flip, 90° rotation)

## How it works

1. **Data pipeline** (`src/dataset.py`): Decodes TFRecords, applies augmentations, and pads batches to match TPU requirements (batch_size must be divisible by number of replicas)
2. **Model** (`src/model.py`): EfficientNetB7 backbone with label smoothing (0.1) and dropout
3. **Training** (`train.py`): Uses Keras 3 with JAX backend, linear warmup (5 epochs) + exponential decay (0.85), learning rate scaled to 0.00024 for 8-core TPU parallelism
4. **Inference** (`predict.py`): 4-pass test-time augmentation (original, H-flip, V-flip, 90° rotation) to boost accuracy

## Results / Status

**V12 training run:**
- Training accuracy: 99.29% (after 35 epochs)
- Validation accuracy: 92.11% (hard); ~93.5% after 4-pass TTA
- Learning rate peak: 0.00024, linear warmup 5 epochs, then exponential decay

Full training telemetry, accuracy curves, and learning rate schedules are in `docs/REPORT.md`.

## Getting started

```bash
git clone https://github.com/SanskarSontakke/Petals-TFRecord-Pipeline
cd Petals-TFRecord-Pipeline
pip install -r requirements.txt
python train.py
```

## Deployment to Kaggle

```bash
cd kaggle_submit
kaggle kernels push
```

The kernel will auto-detect TFRecord data at `/kaggle/input/competitions/tpu-getting-started`.

## Repository structure

- `src/`: Core pipeline modules (dataset, model, callbacks, TPU initialization)
- `train.py`: Training entry point
- `predict.py`: Inference with TTA
- `parse_logs.py`: Log visualization utility
- `kaggle_submit/`: Standalone Kaggle kernel scripts
- `docs/`: Technical report with training curves
- `scratch/`: Development utilities

## License

MIT © 2026 Sanskar Sontakke
