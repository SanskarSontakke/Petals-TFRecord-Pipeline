import tensorflow as tf
from src.dataset import CLASSES, IMAGE_SIZE

def build_model(strategy):
    """
    Initializes and compiles a model within the supplied distribution strategy block.
    This ensures that layers and variables are placed cleanly on the chosen hardware 
    (TPU workers or mirrored GPUs).
    """
    with strategy.scope():
        # Using MobileNetV3Large as a modern, lightweight backbone.
        # This can be effortlessly swapped to EfficientNetV2 for higher-end configurations.
        backbone = tf.keras.applications.MobileNetV3Large(
            input_shape=[*IMAGE_SIZE, 3],
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        
        # Optionally allow fine-tuning
        # backbone.trainable = False 
        
        # Build standard sequential classifier, outputting the 104 flower classes
        model = tf.keras.Sequential([
            backbone,
            tf.keras.layers.Dense(CLASSES, activation='softmax', dtype=tf.float32)
        ])
        
        # Compile within the scope
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['sparse_categorical_accuracy']
        )
        
    return model

if __name__ == "__main__":
    # Test compilation block
    from src.init_hardware import init_hardware
    strategy = init_hardware()
    model = build_model(strategy)
    model.summary()
