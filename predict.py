import os
import numpy as np
import pandas as pd
import tensorflow as tf

from src.init_hardware import init_hardware
from src.dataset import get_test_dataset
from src.model import build_model

def main():
    # 1. Initialize Strategy
    strategy = init_hardware()
    
    # 2. Define Test Pathways
    test_dir = './data/test/*.tfrec'
    TEST_FILENAMES = tf.io.gfile.glob(test_dir)
    
    if not TEST_FILENAMES:
        print(f"Warning: No testing records found at {test_dir}. Update path for your local setup.")
        return
    
    # Base batch size can be identical
    BASE_BATCH_SIZE = 16 
    
    # Construct an explicitly ordered dataset
    # ordered=True is crucial to match predictions row for row to Kaggle's expected IDs
    test_dataset = get_test_dataset(TEST_FILENAMES, strategy, BASE_BATCH_SIZE, ordered=True)

    # 3. Request Built Model & Load Saved State
    print("Building model architecture and loading pre-trained weights...")
    model = build_model(strategy)
    
    try:
        model.load_weights('best_model.weights.h5')
        print("Successfully loaded 'best_model.weights.h5'.")
    except Exception as e:
        print(f"Failed to load weights: {e}")
        return

    # 4. Generate Predictions
    print("Running inference over test records (this may take a moment)...")
    
    # test_dataset yields a tuple of (image, idnum). We map it to yield only images for `model.predict`
    test_images_ds = test_dataset.map(lambda image, idnum: image)
    
    # Runs batched inference quickly across either RTX 3060 or cluster TPU
    predictions = model.predict(test_images_ds, verbose=1)
    
    # Extract the winning class for each prediction via argmax
    predicted_classes = np.argmax(predictions, axis=-1)

    # 5. Extract IDs to pair them correctly
    print("Extracting IDs from TFRecords to align with predictions...")
    test_ids = []
    for idnum_batch in test_dataset.map(lambda image, idnum: idnum):
        # Decode byte strings back to simple UTF-8
        test_ids.extend([id_val.numpy().decode('utf-8') for id_val in idnum_batch])

    # 6. Export to DataFrame & CSV
    submission = pd.DataFrame({
        'id': test_ids,
        'label': predicted_classes
    })
    
    submission_path = 'submission.csv'
    submission.to_csv(submission_path, index=False)
    
    print(f"\nSubmission generated successfully: {submission_path}")
    print(submission.head())

if __name__ == "__main__":
    main()
