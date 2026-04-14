import tensorflow as tf

def get_lr_callback(strategy, epochs=25):
    """
    Returns a tf.keras.callbacks.LearningRateScheduler configured 
    for rapid, stable TPU convergence.
    
    Implements a linear warmup followed by an exponential decay.
    The learning rate bounds scale dynamically with the hardware strategy.
    """
    # Configurable parameters
    LR_START = 0.00001
    LR_MAX = 0.00005 * strategy.num_replicas_in_sync
    LR_MIN = 0.00001
    LR_RAMPUP_EPOCHS = 5
    LR_SUSTAIN_EPOCHS = 0
    LR_EXP_DECAY = 0.8

    def lrfn(epoch):
        if epoch < LR_RAMPUP_EPOCHS:
            # Linear Warmup phase
            lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
        elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
            # Sustain peak LR
            lr = LR_MAX
        else:
            # Exponential Decay phase
            lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
        return lr

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)
    return lr_callback
