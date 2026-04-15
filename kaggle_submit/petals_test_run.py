import os
import re
import numpy as np
import pandas as pd

# --- ARCHITECTURAL PIVOT TO JAX ---
os.environ["KERAS_BACKEND"] = "jax"
import keras
import jax

# Leak 7: Explicitly set mixed_bfloat16 policy to prevent underflow and boost TPU throughput
keras.mixed_precision.set_global_policy("mixed_bfloat16")

# We still use TensorFlow strictly to parse the high-speed TFRecords Dataset pipeline
import tensorflow as tf
from kaggle_datasets import KaggleDatasets

# --- Configuration & Constants ---
EPOCHS = 50
# Leak 3: Upgrading resolution to 600px to match EfficientNetB7 native receptive field
IMAGE_SIZE = [600, 600]
CLASSES = 104
AUTOTUNE = tf.data.AUTOTUNE

# --- Data Pipeline (TensorFlow) ---
def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    # Fix: Raw pixel values [0, 255]. Keras backbones contain built-in Rescaling.
    image = tf.cast(image, tf.float32) 
    image = tf.reshape(image, [512, 512, 3]) 
    return image

def read_labeled_tfrecord(example):
    LABELED_FORMAT = {"image": tf.io.FixedLenFeature([], tf.string), "class": tf.io.FixedLenFeature([], tf.int64)}
    example = tf.io.parse_single_example(example, LABELED_FORMAT)
    return decode_image(example['image']), tf.one_hot(example['class'], CLASSES)

def read_unlabeled_tfrecord(example):
    UNLABELED_FORMAT = {"image": tf.io.FixedLenFeature([], tf.string), "id": tf.io.FixedLenFeature([], tf.string)}
    example = tf.io.parse_single_example(example, UNLABELED_FORMAT)
    return decode_image(example['image']), example['id']

def load_dataset(filenames, labeled=True, ordered=False):
    ignore_order = tf.data.Options()
    if not ordered: ignore_order.experimental_deterministic = False 
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
    dataset = dataset.with_options(ignore_order)
    parse_fn = read_labeled_tfrecord if labeled else read_unlabeled_tfrecord
    dataset = dataset.map(parse_fn, num_parallel_calls=AUTOTUNE)
    return dataset

def data_augment(image, label):
    # Leak 2: Replace rigid resize with a stochastic RandomResizedCrop
    # This maintains aspect ratio while providing natural diversity
    img_shape = tf.shape(image)
    h, w = img_shape[0], img_shape[1]
    
    # Select a crop size (80% to 100% of original)
    crop_size = tf.random.uniform([], minval=0.8, maxval=1.0)
    ch = tf.cast(tf.cast(h, tf.float32) * crop_size, tf.int32)
    cw = tf.cast(tf.cast(w, tf.float32) * crop_size, tf.int32)
    
    # Apply random crop and resize back to target
    image = tf.image.random_crop(image, [ch, cw, 3])
    image = tf.image.resize(image, IMAGE_SIZE)

    # Spatial augmentations (geometric invariance)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    
    # Color augmentations
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    return image, label

def sample_beta(alpha, shape):
    gamma1 = tf.random.gamma(shape, alpha=alpha)
    gamma2 = tf.random.gamma(shape, alpha=alpha)
    return gamma1 / (gamma1 + gamma2)

def batch_mixup_cutmix(images, labels, alpha_mix=0.15, alpha_cut=0.15):
    """
    Kaggle Grandmaster Refinement: Alpha reduced to 0.15 for higher model confidence
    while maintaining robust regularization against flower dataset noise.
    """
    batch_size = tf.shape(images)[0]
    
    # 50% MixUp, 50% CutMix
    do_mixup = tf.random.uniform([]) > 0.5
    
    indices = tf.random.shuffle(tf.range(batch_size))
    images2 = tf.gather(images, indices)
    labels2 = tf.gather(labels, indices)
    
    def do_mix_up():
        lam = sample_beta(alpha_mix, [batch_size, 1, 1, 1])
        mix_images = lam * images + (1 - lam) * images2
        lam_l = tf.reshape(lam, [batch_size, 1])
        mix_labels = lam_l * labels + (1 - lam_l) * labels2
        return mix_images, mix_labels
        
    def do_cut_mix():
        H = tf.shape(images)[1]
        W = tf.shape(images)[2]
        lam = sample_beta(alpha_cut, [batch_size])
        
        cut_rat = tf.math.sqrt(1.0 - lam)
        cut_w = tf.cast(tf.cast(W, tf.float32) * cut_rat, tf.int32)
        cut_h = tf.cast(tf.cast(H, tf.float32) * cut_rat, tf.int32)
        
        cx = tf.random.uniform([batch_size], minval=0, maxval=W, dtype=tf.int32)
        cy = tf.random.uniform([batch_size], minval=0, maxval=H, dtype=tf.int32)
        
        bbx1 = tf.clip_by_value(cx - cut_w // 2, 0, W)
        bby1 = tf.clip_by_value(cy - cut_h // 2, 0, H)
        bbx2 = tf.clip_by_value(cx + cut_w // 2, 0, W)
        bby2 = tf.clip_by_value(cy + cut_h // 2, 0, H)
        
        mask_w = tf.sequence_mask(bbx2, maxlen=W) ^ tf.sequence_mask(bbx1, maxlen=W)
        mask_h = tf.sequence_mask(bby2, maxlen=H) ^ tf.sequence_mask(bby1, maxlen=H)
        mask_w = tf.expand_dims(mask_w, 1) # [batch, 1, W]
        mask_h = tf.expand_dims(mask_h, 2) # [batch, H, 1]
        mask = tf.expand_dims(mask_h & mask_w, -1) # [batch, H, W, 1]
        mask = tf.cast(mask, tf.float32)
        
        mix_images = images * (1 - mask) + images2 * mask
        
        exact_lam = 1.0 - tf.cast(cut_w * cut_h, tf.float32) / tf.cast(W * H, tf.float32)
        lam_l = tf.reshape(exact_lam, [batch_size, 1])
        mix_labels = lam_l * labels + (1 - lam_l) * labels2
        return mix_images, mix_labels
        
    return tf.cond(do_mixup, do_mix_up, do_cut_mix)

def get_training_dataset(filenames, global_batch_size):
    dataset = load_dataset(filenames, ordered=False)
    dataset = dataset.map(data_augment, num_parallel_calls=AUTOTUNE)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(global_batch_size)
    dataset = dataset.map(batch_mixup_cutmix, num_parallel_calls=AUTOTUNE)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset

def get_validation_dataset(filenames, global_batch_size):
    """
    Refactor: Removed dummy zero padding to prevent validation metric poisoning.
    Evaluates only on natural dataset images for pure SOTA metrics.
    """
    def val_preprocess(image, label):
        image = tf.image.resize_with_pad(image, IMAGE_SIZE[0], IMAGE_SIZE[1])
        return image, label

    dataset = load_dataset(filenames, ordered=False)
    dataset = dataset.map(val_preprocess, num_parallel_calls=AUTOTUNE)
    # Use drop_remainder=True to keep TPU shapes static while maintaining pure metrics
    dataset = dataset.batch(global_batch_size, drop_remainder=True)
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset

def get_test_dataset(filenames, global_batch_size, pad_size=0, ordered=True):
    # Leak 2: Maintain aspect ratio consistency in test data
    def test_preprocess(image, idnum):
        image = tf.image.resize_with_pad(image, IMAGE_SIZE[0], IMAGE_SIZE[1])
        return image, idnum

    dataset = load_dataset(filenames, labeled=False, ordered=ordered)
    dataset = dataset.map(test_preprocess, num_parallel_calls=AUTOTUNE)
    if pad_size > 0:
        empty_image = tf.zeros([*IMAGE_SIZE, 3], dtype=tf.float32)
        empty_id = tf.constant("dummy_id", dtype=tf.string)
        dummy_ds = tf.data.Dataset.from_tensors((empty_image, empty_id)).repeat(pad_size)
        dataset = dataset.concatenate(dummy_ds)
    dataset = dataset.batch(global_batch_size)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset

def count_data_items(filenames):
    return np.sum([int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames])

# --- Learning Rate Scheduler ---
# --- Learning Rate Scheduler (WarmUpCosineDecay) ---
def lrfn(epoch):
    """
    SOTA WarmUpCosineDecay Strategy:
    5-Epoch linear warmup to peak LR (2.4e-4) followed by a 
    smooth cosine decay to 1e-6 over 40 epochs. No restarts.
    """
    import math
    LR_START = 1e-6
    LR_MAX = 2.4e-4 
    LR_MIN = 1e-6
    WARMUP_EPOCHS = 5
    TOTAL_EPOCHS = 45 # 5 warmup + 40 decay

    if epoch < WARMUP_EPOCHS:
        lr = (LR_MAX - LR_START) / WARMUP_EPOCHS * epoch + LR_START
    elif epoch < TOTAL_EPOCHS:
        progress = (epoch - WARMUP_EPOCHS) / (TOTAL_EPOCHS - WARMUP_EPOCHS)
        lr = LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * progress))
    else:
        lr = LR_MIN
    return lr

# --- Model Arch (Keras Core) ---
# --- Model Arch (Sequential Backbone Support) ---
def build_model(model_type="B7"):
    """
    Kaggle Grandmaster Refinement: Modular instantiation to support Sequential Offline Ensembling.
    This allows training each massive backbone independently to fit within TPU v5e HBM.
    """
    inputs = keras.Input(shape=[*IMAGE_SIZE, 3])
    
    if model_type == "B7":
        backbone = keras.applications.EfficientNetB7(
            input_shape=[*IMAGE_SIZE, 3],
            include_top=False, weights='imagenet', pooling='avg'
        )(inputs)
    elif model_type == "V2L":
        backbone = keras.applications.EfficientNetV2L(
            input_shape=[*IMAGE_SIZE, 3],
            include_top=False, weights='imagenet', pooling='avg'
        )(inputs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
        
    outputs = keras.layers.Dense(CLASSES, activation='softmax', dtype='float32')(backbone)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# --- Main Driver ---
def main():
    import gc
    # --- 1. SOTA Distribution Setup ---
    print(f"JAX Devices Detected: {jax.devices()}")
    num_replicas = len(jax.devices())
    print(f"TPU Replicas in sync: {num_replicas}")
    
    # Sequential Ensembling Batch Size Tuning: 
    # Global 64 (8 per core) for 600x600 resolution safety
    BASE_BATCH_SIZE = 8 
    GLOBAL_BATCH_SIZE = BASE_BATCH_SIZE * num_replicas
    
    if num_replicas > 1:
        print("Configuring Keras DataParallel across JAX mesh...")
        device_mesh = keras.distribution.DeviceMesh(shape=(num_replicas,), axis_names=["batch"], devices=jax.devices())
        strategy = keras.distribution.DataParallel(device_mesh=device_mesh)
        keras.distribution.set_distribution(strategy)

    # --- Path Resolution ---
    GCS_DS_PATH = "/kaggle/input/competitions/tpu-getting-started"
    TRAIN_FILENAMES = tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords-jpeg-512x512/train/*.tfrec')
    VAL_FILENAMES = tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords-jpeg-512x512/val/*.tfrec')
    TEST_FILENAMES = tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords-jpeg-512x512/test/*.tfrec')

    NUM_TRAINING_IMAGES = count_data_items(TRAIN_FILENAMES)
    NUM_VALIDATION_IMAGES = count_data_items(VAL_FILENAMES)
    
    STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // GLOBAL_BATCH_SIZE
    # Calculate steps based on natural dataset size (dropped remainder)
    VALIDATION_STEPS = NUM_VALIDATION_IMAGES // GLOBAL_BATCH_SIZE
    
    print(f"SOTA Mode: Training on {NUM_TRAINING_IMAGES} | Validating on {NUM_VALIDATION_IMAGES}")

    train_dataset = get_training_dataset(TRAIN_FILENAMES, GLOBAL_BATCH_SIZE)
    val_dataset = get_validation_dataset(VAL_FILENAMES, GLOBAL_BATCH_SIZE)

    # --- SHARED CONFIG ---
    focal_loss = keras.losses.CategoricalFocalCrossentropy(alpha=0.25, gamma=2.0, label_smoothing=0.05)
    lr_callback = keras.callbacks.LearningRateScheduler(lrfn, verbose=True)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=12, restore_best_weights=True, verbose=1)

    # --- PHASE 1: Train EfficientNetB7 ---
    print("\n[PHASE 1] Training EfficientNetB7 Pipeline...")
    model_a = build_model("B7")
    model_a.compile(optimizer=keras.optimizers.Adam(), loss=focal_loss, metrics=['categorical_accuracy'])
    
    ckpt_a = keras.callbacks.ModelCheckpoint(filepath='best_model_A.keras', monitor='val_categorical_accuracy', save_best_only=True, verbose=1)
    
    model_a.fit(
        train_dataset, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, 
        validation_data=val_dataset, validation_steps=VALIDATION_STEPS,
        callbacks=[lr_callback, early_stopping, ckpt_a], verbose=2
    )
    
    # Memory Reset A
    print("\nPhase 1 Complete. Purging Model A from TPU HBM...")
    del model_a
    keras.backend.clear_session()
    gc.collect()

    # --- PHASE 2: Train EfficientNetV2L ---
    print("\n[PHASE 2] Training EfficientNetV2-L Pipeline...")
    model_b = build_model("V2L")
    model_b.compile(optimizer=keras.optimizers.Adam(), loss=focal_loss, metrics=['categorical_accuracy'])
    
    ckpt_b = keras.callbacks.ModelCheckpoint(filepath='best_model_B.keras', monitor='val_categorical_accuracy', save_best_only=True, verbose=1)
    
    model_b.fit(
        train_dataset, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, 
        validation_data=val_dataset, validation_steps=VALIDATION_STEPS,
        callbacks=[lr_callback, early_stopping, ckpt_b], verbose=2
    )
    
    # Memory Reset B
    print("\nPhase 2 Complete. Purging Model B from TPU HBM...")
    del model_b
    keras.backend.clear_session()
    gc.collect()

    # --- PHASE 3: Weighted Ensemble Inference ---
    print("\n[PHASE 3] Reloading Models for Weighted Ensemble Inference...")
    # Instantiate models again (training session was cleared)
    model_a = build_model("B7")
    model_a.load_weights('best_model_A.keras')
    
    model_b = build_model("V2L")
    model_b.load_weights('best_model_B.keras')
    
    blend_a = 0.60
    blend_b = 0.40
    
    print(f"Ensemble Weights: B7={blend_a} | V2L={blend_b}")
    
    print("\nRunning Inference Pipeline with 2-Pass Test Time Augmentation (TTA)...")
    NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)
    remainder = NUM_TEST_IMAGES % GLOBAL_BATCH_SIZE
    pad_size = (GLOBAL_BATCH_SIZE - remainder) % GLOBAL_BATCH_SIZE
    
    print(f"Test Images: {NUM_TEST_IMAGES}. Padding: {pad_size}")
    
    test_dataset = get_test_dataset(TEST_FILENAMES, GLOBAL_BATCH_SIZE, pad_size=pad_size, ordered=True)
    
    # --- Leak 5/4: Weighted Ensemble Blending & Coordinate-Safe Transforms ---
    blend_b7 = 0.60
    blend_v2l = 0.40
    
    def predict_weighted(ds):
        probs_a = model_a.predict(ds, verbose=0)
        probs_b = model_b.predict(ds, verbose=0)
        return blend_a * probs_a + blend_b * probs_b

    # PASS 1: Original Images
    print("Inference Pass 1 (Original)...")
    test_images_ds = test_dataset.map(lambda image, idnum: image)
    probs1 = predict_weighted(test_images_ds)
    
    # PASS 2: Horizontal Flip
    print("Inference Pass 2 (Horizontal Flip)...")
    test_images_ds2 = test_dataset.map(lambda image, idnum: tf.image.flip_left_right(image))
    probs2 = predict_weighted(test_images_ds2)
    
    # PASS 3: Vertical Flip
    print("Inference Pass 3 (Vertical Flip)...")
    test_images_ds3 = test_dataset.map(lambda image, idnum: tf.image.flip_up_down(image))
    probs3 = predict_weighted(test_images_ds3)
 
    # PASS 4: Rot90 k=1
    print("Inference Pass 4 (Rot90 k=1)...")
    test_images_ds4 = test_dataset.map(lambda image, idnum: tf.image.rot90(image, k=1))
    probs4 = predict_weighted(test_images_ds4)
    
    # PASS 5: Rot90 k=3
    print("Inference Pass 5 (Rot90 k=3)...")
    test_images_ds5 = test_dataset.map(lambda image, idnum: tf.image.rot90(image, k=3))
    probs5 = predict_weighted(test_images_ds5)
    
    # PASS 6: H-Flip + V-Flip
    print("Inference Pass 6 (H-Flip + V-Flip)...")
    test_images_ds6 = test_dataset.map(lambda image, idnum: tf.image.flip_up_down(tf.image.flip_left_right(image)))
    probs6 = predict_weighted(test_images_ds6)
    
    # PASS 7: Central Crop 80% (Zoom-safe resize)
    print("Inference Pass 7 (Central Crop 80%)...")
    test_images_ds7 = test_dataset.map(lambda image, idnum: tf.image.resize_with_pad(tf.image.central_crop(image, 0.8), IMAGE_SIZE[0], IMAGE_SIZE[1]))
    probs7 = predict_weighted(test_images_ds7)
    
    # PASS 8: Central Crop + Horizontal Flip
    print("Inference Pass 8 (Central Crop + H-Flip)...")
    test_images_ds8 = test_dataset.map(lambda image, idnum: tf.image.flip_left_right(tf.image.resize_with_pad(tf.image.central_crop(image, 0.8), IMAGE_SIZE[0], IMAGE_SIZE[1])))
    probs8 = predict_weighted(test_images_ds8)
    
    # Average Probabilities across TTA passes
    predictions = (probs1 + probs2 + probs3 + probs4 + probs5 + probs6 + probs7 + probs8) / 8.0
    predicted_classes = np.argmax(predictions, axis=-1)
    
    test_ids = []
    for idnum_batch in test_dataset.map(lambda image, idnum: idnum):
        test_ids.extend([id_val.numpy().decode('utf-8') for id_val in idnum_batch])

    if pad_size > 0:
        predicted_classes = predicted_classes[:-pad_size]
        test_ids = test_ids[:-pad_size]

    submission = pd.DataFrame({'id': test_ids, 'label': predicted_classes})
    submission.to_csv('submission.csv', index=False)
    print("SUCCESS: submission.csv generated (TTA-Enhanced) for Kaggle Petals competition!")

if __name__ == "__main__":
    main()
