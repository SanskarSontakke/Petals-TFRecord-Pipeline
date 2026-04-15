import tensorflow as tf

def get_lr_callback(strategy, epochs=25):
    """
    Returns a tf.keras.callbacks.LearningRateScheduler configured 
    for rapid, stable TPU convergence.
    
    Implements a linear warmup followed by an exponential decay.
    The learning rate bounds scale dynamically with the hardware strategy.
    """
    # Configurable parameters
    LR_START = 1e-6
    LR_MAX = 2.4e-4 
    LR_MIN = 1e-6
    LR_RAMPUP_EPOCHS = 5
    LR_TOTAL_EPOCHS = 50

    def lrfn(epoch):
        if epoch < LR_RAMPUP_EPOCHS:
            # Linear Warmup phase: Start to Max
            lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
        else:
            # Smooth Cosine Decay phase: Max to Min over remaining epochs
            import math
            progress = (epoch - LR_RAMPUP_EPOCHS) / (LR_TOTAL_EPOCHS - LR_RAMPUP_EPOCHS)
            lr = LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * progress))
        return lr

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)
    return lr_callback
