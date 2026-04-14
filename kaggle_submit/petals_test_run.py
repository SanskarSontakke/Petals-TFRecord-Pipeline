import os
import re
import numpy as np
import pandas as pd

# --- ARCHITECTURAL PIVOT TO JAX ---
# Kaggle's new TPU v5e-8 fleet permanently deprecated TensorFlow distributed C++ Ops natively. 
# To harness the 8-Core TPU flawlessly, we switch the Keras 3 underlying calculation backend to JAX!
os.environ["KERAS_BACKEND"] = "jax"
import keras
import jax

# We still use TensorFlow strictly to parse the high-speed TFRecords Dataset pipeline
import tensorflow as tf
from kaggle_datasets import KaggleDatasets

# --- Configuration & Constants ---
EPOCHS = 45
IMAGE_SIZE = [512, 512]
CLASSES = 104
AUTOTUNE = tf.data.AUTOTUNE

# --- Data Pipeline (TensorFlow) ---
def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
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
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    # Flower classification benefits significantly from rotational invariance
    image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    return image, label

def sample_beta(alpha, shape):
    gamma1 = tf.random.gamma(shape, alpha=alpha)
    gamma2 = tf.random.gamma(shape, alpha=alpha)
    return gamma1 / (gamma1 + gamma2)

def batch_mixup_cutmix(images, labels, alpha_mix=0.2, alpha_cut=1.0):
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

def get_validation_dataset(filenames, global_batch_size, pad_size=0):
    dataset = load_dataset(filenames, ordered=False)
    if pad_size > 0:
        empty_image = tf.zeros([*IMAGE_SIZE, 3], dtype=tf.float32)
        empty_label = tf.zeros([CLASSES], dtype=tf.float32) # One-hot zeros for padding
        dummy_ds = tf.data.Dataset.from_tensors((empty_image, empty_label)).repeat(pad_size)
        dataset = dataset.concatenate(dummy_ds)
    dataset = dataset.batch(global_batch_size)
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset

def get_test_dataset(filenames, global_batch_size, pad_size=0, ordered=True):
    dataset = load_dataset(filenames, labeled=False, ordered=ordered)
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
def lrfn(epoch, lr):
    import math
    LR_START = 0.00001
    LR_MAX = 0.00003 * 8 # Peak LR scaled strongly by 8 TPU cores
    LR_MIN = 0.00001
    LR_RAMPUP_EPOCHS = 5
    EPOCH_DECAY_RESTART = 20 # Cosine Annealing restart frequency

    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    else:
        progress = ((epoch - LR_RAMPUP_EPOCHS) % EPOCH_DECAY_RESTART) / float(EPOCH_DECAY_RESTART)
        lr = LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * progress))
    return lr

# --- Model Arch (Keras Core) ---
def build_model():
    backbone = keras.applications.EfficientNetB7(
        input_shape=[*IMAGE_SIZE, 3],
        include_top=False, weights='imagenet', pooling='avg'
    )
    model = keras.Sequential([
        backbone,
        keras.layers.Dense(CLASSES, activation='softmax', dtype='float32')
    ])
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.CategoricalFocalCrossentropy(alpha=0.25, gamma=2.0, label_smoothing=0.10),
                  metrics=['categorical_accuracy'])
    return model

# --- Main Driver ---
def main():
    print(f"JAX Devices Detected: {jax.devices()}")
    num_replicas = len(jax.devices())
    print(f"TPU Replicas in sync: {num_replicas}")
    BASE_BATCH_SIZE = 16
    GLOBAL_BATCH_SIZE = BASE_BATCH_SIZE * num_replicas
    
    # Configure Keras to natively utilize all JAX devices (TPU Multi-Core)
    if num_replicas > 1:
        print("Configuring Keras DataParallel across JAX mesh...")
        device_mesh = keras.distribution.DeviceMesh(shape=(num_replicas,), axis_names=["batch"], devices=jax.devices())
        strategy = keras.distribution.DataParallel(device_mesh=device_mesh)
        keras.distribution.set_distribution(strategy)

    try:
        # Ignore Kaggle's buggy GCS API lying about the missing /competitions/ folder
        _ = KaggleDatasets().get_gcs_path('tpu-getting-started')
    except Exception:
        pass
        
    GCS_DS_PATH = "/kaggle/input/competitions/tpu-getting-started"
        
    print(f"Bypassing API... Using hardcoded diagnostic mount path: {GCS_DS_PATH}")
    TRAIN_FILENAMES = tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords-jpeg-512x512/train/*.tfrec')
    VAL_FILENAMES = tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords-jpeg-512x512/val/*.tfrec')
    TEST_FILENAMES = tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords-jpeg-512x512/test/*.tfrec')

    if not TRAIN_FILENAMES:
        print("Dataset unmounted or path missing! Diagnostic footprint:")
        for root, dirs, files in os.walk('/kaggle/input'):
            print(f"Path: {root} | Dirs: {dirs} | File Count: {len(files)}")
        return
        
    NUM_TRAINING_IMAGES = count_data_items(TRAIN_FILENAMES)
    NUM_VALIDATION_IMAGES = count_data_items(VAL_FILENAMES)
    
    # Calculate padding for validation to ensure metric numbers are "real" (evaluated on 100% of images)
    val_remainder = NUM_VALIDATION_IMAGES % GLOBAL_BATCH_SIZE
    val_pad_size = (GLOBAL_BATCH_SIZE - val_remainder) % GLOBAL_BATCH_SIZE
    
    STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // GLOBAL_BATCH_SIZE
    VALIDATION_STEPS = (NUM_VALIDATION_IMAGES + val_pad_size) // GLOBAL_BATCH_SIZE
    
    print(f"Global Batch Size: {GLOBAL_BATCH_SIZE}")
    print(f"Training on {NUM_TRAINING_IMAGES} Images | Validating on {NUM_VALIDATION_IMAGES} (+{val_pad_size} padding).")
    print(f"Epochs: {EPOCHS}")
    
    train_dataset = get_training_dataset(TRAIN_FILENAMES, GLOBAL_BATCH_SIZE)
    val_dataset = get_validation_dataset(VAL_FILENAMES, GLOBAL_BATCH_SIZE, pad_size=val_pad_size)
    model = build_model()
    lr_callback = keras.callbacks.LearningRateScheduler(lrfn, verbose=1)
    
    print("\nTraining execution transferring to JAX JIT compiler with TPU Scheduler...")
    model.fit(train_dataset, 
              steps_per_epoch=STEPS_PER_EPOCH, 
              epochs=EPOCHS, 
              validation_data=val_dataset,
              validation_steps=VALIDATION_STEPS,
              callbacks=[lr_callback], 
              verbose=1)
    
    print("\nRunning Inference Pipeline with 2-Pass Test Time Augmentation (TTA)...")
    NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)
    remainder = NUM_TEST_IMAGES % GLOBAL_BATCH_SIZE
    pad_size = (GLOBAL_BATCH_SIZE - remainder) % GLOBAL_BATCH_SIZE
    
    print(f"Test Images: {NUM_TEST_IMAGES}. Padding: {pad_size}")
    
    test_dataset = get_test_dataset(TEST_FILENAMES, GLOBAL_BATCH_SIZE, pad_size=pad_size, ordered=True)
    
    # PASS 1: Original Images
    print("Inference Pass 1 (Original)...")
    test_images_ds = test_dataset.map(lambda image, idnum: image)
    probs1 = model.predict(test_images_ds, verbose=1)
    
    # PASS 2: Horizontal Flip
    print("Inference Pass 2 (Horizontal Flip)...")
    test_images_ds2 = test_dataset.map(lambda image, idnum: tf.image.flip_left_right(image))
    probs2 = model.predict(test_images_ds2, verbose=1)
    
    # PASS 3: Vertical Flip
    print("Inference Pass 3 (Vertical Flip)...")
    test_images_ds3 = test_dataset.map(lambda image, idnum: tf.image.flip_up_down(image))
    probs3 = model.predict(test_images_ds3, verbose=1)

    # PASS 4: Rot90 k=1
    print("Inference Pass 4 (Rot90 k=1)...")
    test_images_ds4 = test_dataset.map(lambda image, idnum: tf.image.rot90(image, k=1))
    probs4 = model.predict(test_images_ds4, verbose=1)
    
    # PASS 5: Rot90 k=3
    print("Inference Pass 5 (Rot90 k=3)...")
    test_images_ds5 = test_dataset.map(lambda image, idnum: tf.image.rot90(image, k=3))
    probs5 = model.predict(test_images_ds5, verbose=1)
    
    # PASS 6: H-Flip + V-Flip
    print("Inference Pass 6 (H-Flip + V-Flip)...")
    test_images_ds6 = test_dataset.map(lambda image, idnum: tf.image.flip_up_down(tf.image.flip_left_right(image)))
    probs6 = model.predict(test_images_ds6, verbose=1)
    
    # PASS 7: Central Crop 80% (Zoom)
    print("Inference Pass 7 (Central Crop 80%)...")
    test_images_ds7 = test_dataset.map(lambda image, idnum: tf.image.resize(tf.image.central_crop(image, 0.8), IMAGE_SIZE))
    probs7 = model.predict(test_images_ds7, verbose=1)
    
    # PASS 8: Central Crop + Horizontal Flip
    print("Inference Pass 8 (Central Crop + H-Flip)...")
    test_images_ds8 = test_dataset.map(lambda image, idnum: tf.image.flip_left_right(tf.image.resize(tf.image.central_crop(image, 0.8), IMAGE_SIZE)))
    probs8 = model.predict(test_images_ds8, verbose=1)
    
    # Average Probabilities
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
