import math

def test_lr_scheduler(epochs):
    LR_START = 1e-6
    LR_MAX = 2.4e-4 
    LR_MIN = 1e-6
    LR_RAMPUP_EPOCHS = max(1, int(epochs * 0.2))
    
    print(f"\n--- Testing for {epochs} epochs (Rampup: {LR_RAMPUP_EPOCHS}) ---")
    
    def lrfn(epoch):
        if epoch < LR_RAMPUP_EPOCHS:
            lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
        else:
            decay_steps = (epochs - 1) - LR_RAMPUP_EPOCHS
            if decay_steps > 0:
                k = (LR_MIN / LR_MAX) ** (1 / decay_steps)
                lr = LR_MAX * (k ** (epoch - LR_RAMPUP_EPOCHS))
            else:
                lr = LR_MAX
        return lr

    for e in range(epochs):
        lr = lrfn(e)
        if e == 0 or e == LR_RAMPUP_EPOCHS or e == epochs - 1:
            print(f"Epoch {e:2d}: LR = {lr:.8f}")

test_lr_scheduler(25)
test_lr_scheduler(50)
test_lr_scheduler(10)
