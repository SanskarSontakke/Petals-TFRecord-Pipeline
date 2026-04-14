import os
import re
import numpy as np
import tensorflow as tf

from src.init_hardware import init_hardware
from src.dataset import get_training_dataset, get_validation_dataset
from src.model import build_model
from src.callbacks import get_lr_callback

# Config parameters
EPOCHS = 25
BASE_BATCH_SIZE = 16

def count_data_items(filenames):
    """
    Extracts the number of data items embedded in Kaggle TFRecord filenames.
    E.g., 'train00-2071.tfrec' counts exactly 2071 records.
    """
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)

def main():
    # 1. Initialize Hardware Strategy
    strategy = init_hardware()
    
    # 2. Define Dataset Paths
    # Note: When uploading to Kaggle, you would use KaggleDatasets().get_gcs_path() here.
    train_dir = './data/train/*.tfrec'
    val_dir = './data/val/*.tfrec'
    
    TRAIN_FILENAMES = tf.io.gfile.glob(train_dir)
    VAL_FILENAMES = tf.io.gfile.glob(val_dir)
    
    if not TRAIN_FILENAMES:
        print(f"Warning: No training records found at {train_dir}. Adjust paths for local environment.")
        return
        
    NUM_TRAINING_IMAGES = count_data_items(TRAIN_FILENAMES)
    GLOBAL_BATCH_SIZE = BASE_BATCH_SIZE * strategy.num_replicas_in_sync
    STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // GLOBAL_BATCH_SIZE
    
    print(f"Dataset summary: {NUM_TRAINING_IMAGES} training images.")
    print(f"Global Batch Size: {GLOBAL_BATCH_SIZE} | Steps per Epoch: {STEPS_PER_EPOCH}")

    # 3. Initialize Datasets
    train_dataset = get_training_dataset(TRAIN_FILENAMES, strategy, BASE_BATCH_SIZE)
    val_dataset = get_validation_dataset(VAL_FILENAMES, strategy, BASE_BATCH_SIZE)

    # 4. Instantiate and Compile Model (Strictly Inside strategy.scope)
    print("\nBuilding model architecture...")
    model = build_model(strategy)
    
    # 5. Define Training Callbacks
    lr_schedule = get_lr_callback(strategy, epochs=EPOCHS)
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3, # Will stop if validation loss doesn't improve for 3 entire epochs
        restore_best_weights=True,
        verbose=1
    )
    
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath='best_model.weights.h5',
        monitor='val_sparse_categorical_accuracy',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )

    # 6. Run Training Fit
    print("\nBeginning Training Loop...")
    history = model.fit(
        train_dataset, 
        steps_per_epoch=STEPS_PER_EPOCH, 
        epochs=EPOCHS, 
        validation_data=val_dataset,
        callbacks=[lr_schedule, early_stopping, model_checkpoint]
    )
    
    print("\nTraining completed successfully! Best weights have been preserved to 'best_model.weights.h5'.")

if __name__ == "__main__":
    main()
