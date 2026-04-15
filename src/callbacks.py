import tensorflow as tf

def get_lr_callback(strategy, epochs=25):
    """
    Returns a tf.keras.callbacks.LearningRateScheduler configured 
    for rapid, stable TPU convergence.
    
    Implements a 20% linear warmup followed by an exponential decay
    that stretches exactly across the remaining training period.
    """
    # Configurable parameters
    LR_START = 1e-6
    LR_MAX = 2.4e-4 
    LR_MIN = 1e-6
    
    # Dynamic Ramp-up: First 20% of epochs (min 1)
    LR_RAMPUP_EPOCHS = max(1, int(epochs * 0.2))

    def lrfn(epoch):
        if epoch < LR_RAMPUP_EPOCHS:
            # Linear Warmup phase: Start to Max
            lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
        else:
            # Exponential Decay phase: Max to Min
            # Calculation: LR starts at MAX at ramp-up completion and 
            # finishes at MIN exactly at the final epoch.
            decay_steps = (epochs - 1) - LR_RAMPUP_EPOCHS
            if decay_steps > 0:
                k = (LR_MIN / LR_MAX) ** (1 / decay_steps)
                lr = LR_MAX * (k ** (epoch - LR_RAMPUP_EPOCHS))
            else:
                lr = LR_MAX
        return lr

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)
    return lr_callback
