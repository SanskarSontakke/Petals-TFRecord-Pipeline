import tensorflow as tf

# Constants
AUTOTUNE = tf.data.AUTOTUNE
IMAGE_SIZE = [512, 512]  # Usually 192, 224, 331, 512 are provided in the competition
CLASSES = 104

def decode_image(image_data):
    """
    Decodes the raw JPEG bytestring, normalizes to [0, 1], and properly reshapes.
    """
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    
    # Explicit shape is required for TPUs since they cannot dynamically reshape effectively.
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image

def data_augment(image, label):
    """
    Standard spatial and color augmentations.
    Flower classification benefits significantly from rotational invariance.
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    return image, label

def sample_beta(alpha, shape):
    gamma1 = tf.random.gamma(shape, alpha=alpha)
    gamma2 = tf.random.gamma(shape, alpha=alpha)
    return gamma1 / (gamma1 + gamma2)

def batch_mixup_cutmix(images, labels, alpha_mix=0.2, alpha_cut=0.2):
    """
    Applies MixUp and CutMix at the batch level.
    Alpha is relaxed to 0.20 to prevent over-regularization.
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

def read_labeled_tfrecord(example):
    """
    Parses a single tf.Example containing image data and label.
    """
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "class": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    label = tf.cast(example['class'], tf.int32)
    return image, label

def read_unlabeled_tfrecord(example):
    """
    Parses a single tf.Example containing image data and ID (used for testing).
    """
    UNLABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "id": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    idnum = example['id']
    return image, idnum

def load_dataset(filenames, labeled=True, ordered=False):
    """
    Reads from TFRecord files. Set labeled=False for test data.
    """
    # For optimal performance, ignore data order since it will be shuffled
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False 
        
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
    dataset = dataset.with_options(ignore_order)
    
    parse_fn = read_labeled_tfrecord if labeled else read_unlabeled_tfrecord
    dataset = dataset.map(parse_fn, num_parallel_calls=AUTOTUNE)
    return dataset

def get_test_dataset(filenames, strategy, base_batch_size=16, ordered=True):
    """
    Returns a tf.data.Dataset for test predictions. It must be ordered 
    if we want to match predictions with their IDs correctly later.
    """
    global_batch_size = base_batch_size * strategy.num_replicas_in_sync
    dataset = load_dataset(filenames, labeled=False, ordered=ordered)
    dataset = dataset.batch(global_batch_size)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset

def get_training_dataset(filenames, strategy, base_batch_size=16):
    """
    Returns a thoroughly optimized tf.data.Dataset for training.
    Computes global batch size using the strategy replicas.
    """
    global_batch_size = base_batch_size * strategy.num_replicas_in_sync
    
    dataset = load_dataset(filenames, ordered=False)
    dataset = dataset.map(data_augment, num_parallel_calls=AUTOTUNE)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(2048)
    
    # Proper batching, followed by prefetching
    dataset = dataset.batch(global_batch_size)
    dataset = dataset.map(batch_mixup_cutmix, num_parallel_calls=AUTOTUNE)
    dataset = dataset.prefetch(AUTOTUNE)
    
    return dataset

def get_validation_dataset(filenames, strategy, base_batch_size=16):
    """
    Returns an optimized tf.data.Dataset for validation.
    """
    global_batch_size = base_batch_size * strategy.num_replicas_in_sync
    
    dataset = load_dataset(filenames, ordered=False)
    dataset = dataset.batch(global_batch_size)
    dataset = dataset.cache() # Cashing val set drastically improves epoch times
    dataset = dataset.prefetch(AUTOTUNE)
    
    return dataset
