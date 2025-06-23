import os
import shutil
import random
from sklearn.model_selection import train_test_split

# --- Configuration ---
SOURCE_DATASET_FOLDER = "/Users/kaimikkelsen/SCNet_guitar/data/guitar_hum_dataset"
TRAIN_OUTPUT_FOLDER = "/Users/kaimikkelsen/SCNet_guitar/data/guitar_hum_dataset_split/train"
VALID_OUTPUT_FOLDER = "/Users/kaimikkelsen/SCNet_guitar/data/guitar_hum_dataset_split/valid" # New validation folder
TEST_OUTPUT_FOLDER = "/Users/kaimikkelsen/SCNet_guitar/data/guitar_hum_dataset_split/test"

# Split ratios
# Example: 70% Train, 15% Validation, 15% Test
# First split: TRAIN_SIZE for training, (1 - TRAIN_SIZE) for temp_remaining
TRAIN_SIZE = 0.70
# Second split: TEST_SIZE_RELATIVE for test, (1 - TEST_SIZE_RELATIVE) for validation
# Calculate this relative to the 'remaining' portion from the first split.
# If remaining is 30% (0.30), and you want test to be 15% of total (0.15),
# then TEST_SIZE_RELATIVE = 0.15 / 0.30 = 0.5 (meaning 50% of the remaining goes to test)
# Similarly, VALID_SIZE_RELATIVE = 0.15 / 0.30 = 0.5 (meaning 50% of the remaining goes to valid)
# So, for 70/15/15: remaining = 1 - 0.70 = 0.30. Test = 0.15. Valid = 0.15.
# Test_relative = 0.15 / 0.30 = 0.5
# Valid_relative = 0.15 / 0.30 = 0.5
VALID_TEST_RATIO_RELATIVE = 0.5 # 50% of the remaining data for validation, 50% for test

# Set a random state for reproducibility
RANDOM_SEED = 42

# --- Script ---

def split_dataset_three_ways():
    # Get a list of all subfolders (each representing a mixed audio example)
    all_examples = [
        os.path.join(SOURCE_DATASET_FOLDER, d)
        for d in os.listdir(SOURCE_DATASET_FOLDER)
        if os.path.isdir(os.path.join(SOURCE_DATASET_FOLDER, d))
    ]

    if not all_examples:
        print(f"No examples found in {SOURCE_DATASET_FOLDER}. Please run the dataset creation script first.")
        return

    print(f"Found {len(all_examples)} examples in the dataset.")

    # 1. Split into training and temporary (validation + test)
    train_examples, remaining_examples = train_test_split(
        all_examples,
        train_size=TRAIN_SIZE,
        random_state=RANDOM_SEED,
        shuffle=True
    )

    # 2. Split the remaining into validation and test
    valid_examples, test_examples = train_test_split(
        remaining_examples,
        test_size=VALID_TEST_RATIO_RELATIVE, # This 'test_size' is relative to 'remaining_examples'
        random_state=RANDOM_SEED,
        shuffle=True
    )

    print(f"Splitting into {len(train_examples)} training examples,")
    print(f"               {len(valid_examples)} validation examples, and")
    print(f"               {len(test_examples)} testing examples.")

    # Create target directories
    os.makedirs(TRAIN_OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(VALID_OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(TEST_OUTPUT_FOLDER, exist_ok=True)

    # Function to copy files for a given example
    def copy_example(source_folder, destination_base_folder):
        example_name = os.path.basename(source_folder)
        destination_folder = os.path.join(destination_base_folder, example_name)
        os.makedirs(destination_folder, exist_ok=True) # Create the track folder

        # Copy each file within the example folder
        for file_name in os.listdir(source_folder):
            source_file_path = os.path.join(source_folder, file_name)
            destination_file_path = os.path.join(destination_folder, file_name)
            try:
                shutil.copy2(source_file_path, destination_file_path) # copy2 preserves metadata
            except Exception as e:
                print(f"Error copying {source_file_path} to {destination_file_path}: {e}")

    print("Copying training examples...")
    for example_folder in train_examples:
        copy_example(example_folder, TRAIN_OUTPUT_FOLDER)
    print("Training examples copied.")

    print("Copying validation examples...")
    for example_folder in valid_examples:
        copy_example(example_folder, VALID_OUTPUT_FOLDER)
    print("Validation examples copied.")

    print("Copying testing examples...")
    for example_folder in test_examples:
        copy_example(example_folder, TEST_OUTPUT_FOLDER)
    print("Testing examples copied.")

    print(f"\nDataset split complete.")
    print(f"Train set in {TRAIN_OUTPUT_FOLDER}")
    print(f"Valid set in {VALID_OUTPUT_FOLDER}")
    print(f"Test set in {TEST_OUTPUT_FOLDER}")

if __name__ == "__main__":
    split_dataset_three_ways()