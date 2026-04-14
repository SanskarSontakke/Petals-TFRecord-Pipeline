import tensorflow as tf
import os

def init_hardware():
    """
    Initializes the hardware strategy.
    First attempts to connect to a TPU cluster (vital for Kaggle instances).
    If that fails, it smoothly falls back to detecting GPUs (e.g. RTX 3060) and 
    uses MirroredStrategy, or the default strategy if no accelerator is found.
    """
    try:
        # First attempt Kaggle TPU VM local detection
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
        print(f"TPU VM detected: {tpu.master()}")
    except ValueError:
        try:
            # Fallback to older TPU Node network detection
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            print(f"TPU Node detected: {tpu.master()}")
        except ValueError:
            tpu = None
            print("No TPU cluster detected. Falling back to local GPU/CPU.")

    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        print("TPU Strategy Initialized.")
    else:
        # Check for GPUs
        gpus = tf.config.list_physical_devices('GPU')
        if len(gpus) > 0:
            print(f"Found {len(gpus)} GPU(s). Enabling memory growth.")
            # Set memory growth to prevent TF from hoarding all RTX 3060 VRAM right away
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    print(f"Memory growth setting failed: {e}")
            
            # Using MirroredStrategy is a good standard for multi-GPU or single-GPU setups
            strategy = tf.distribute.MirroredStrategy()
            print("MirroredStrategy Initialized.")
        else:
            strategy = tf.distribute.get_strategy()
            print("No GPU or TPU found. Default CPU Strategy Initialized.")

    print(f"Number of replicas in sync: {strategy.num_replicas_in_sync}")
    return strategy

if __name__ == "__main__":
    strategy = init_hardware()
