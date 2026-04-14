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
    dataset = dataset.repeat()
    dataset = dataset.shuffle(2048)
    
    # Proper batching, followed by prefetching
    dataset = dataset.batch(global_batch_size)
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
