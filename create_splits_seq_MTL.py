import os
import numpy as np
import pandas as pd
import argparse
from datasets.dataset_generic import save_splits

parser = argparse.ArgumentParser(description='Creating splits for multi-task learning with Cox regression')
parser.add_argument('--label_frac', type=float, default= 1.0,
                    help='fraction of labels (default: 1)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--k', type=int, default=5,
                    help='number of splits (default: 5)')
parser.add_argument('--val_frac', type=float, default= 0.2,
                    help='fraction of labels for validation (default: 0.2)')
parser.add_argument('--test_frac', type=float, default= 0.2,
                    help='fraction of labels for test (default: 0.2)')
parser.add_argument('--dataset', type=str, default='Yifuyuan')

args = parser.parse_args()

# Load the MTL dataset
csv_path = 'dataset_csv/MTL182.csv'
data = pd.read_csv(csv_path, encoding='utf-8-sig')

# Get all unique patients
patients = data['case_id'].unique()
n_patients = len(patients)

# Calculate split sizes
total_val = int(np.round(n_patients * args.val_frac))
total_test = int(np.round(n_patients * args.test_frac))
total_train = n_patients - total_val - total_test

print(f"Total patients: {n_patients}")
print(f"Train patients: {total_train}")
print(f"Val patients: {total_val}")
print(f"Test patients: {total_test}")

# Function to create balanced splits considering Cox duration
def create_balanced_splits(patients, data, n_splits=5, seed=1):
    np.random.seed(seed)
    
    # Get duration values for each patient
    patient_durations = {}
    for patient in patients:
        patient_data = data[data['case_id'] == patient]
        # Use the first occurrence's duration (assuming one slide per patient)
        patient_durations[patient] = patient_data['duration'].values[0]
    
    # Sort patients by duration to ensure balanced splits
    sorted_patients = sorted(patients, key=lambda x: patient_durations[x])
    
    # Create splits
    splits = []
    for i in range(n_splits):
        # Shuffle the sorted patients with a seed based on split index
        shuffled = sorted_patients.copy()
        split_seed = seed + i
        np.random.seed(split_seed)
        np.random.shuffle(shuffled)
        
        # Split into train, val, test
        train_end = total_train
        val_end = train_end + total_val
        
        train_patients = shuffled[:train_end]
        val_patients = shuffled[train_end:val_end]
        test_patients = shuffled[val_end:]
        
        splits.append((train_patients, val_patients, test_patients))
    
    return splits

# Create splits
splits = create_balanced_splits(patients, data, n_splits=args.k, seed=args.seed)

# Function to get slide IDs for patients
def get_slide_ids(patient_list, data):
    slide_ids = []
    for patient in patient_list:
        patient_data = data[data['case_id'] == patient]
        slide_ids.extend(patient_data['slide_id'].values)
    return slide_ids

# Save splits
task_name = 'task_mtl182'
split_dir = f'splits/{args.dataset}/{task_name}_{int(args.label_frac * 100)}'
os.makedirs(split_dir, exist_ok=True)

for i, (train_patients, val_patients, test_patients) in enumerate(splits):
    # Get slide IDs
    train_slides = get_slide_ids(train_patients, data)
    val_slides = get_slide_ids(val_patients, data)
    test_slides = get_slide_ids(test_patients, data)
    
    # Create split dataframes
    train_df = pd.DataFrame({'train': train_slides})
    val_df = pd.DataFrame({'val': val_slides})
    test_df = pd.DataFrame({'test': test_slides})
    
    # Combine and save
    combined_df = pd.concat([train_df, val_df, test_df], axis=1)
    split_file = os.path.join(split_dir, f'splits_{i}.csv')
    combined_df.to_csv(split_file, index=False)
    
    # Save boolean style split
    all_slides = data['slide_id'].values
    bool_df = pd.DataFrame(index=all_slides, columns=['train', 'val', 'test'], dtype=bool)
    bool_df.fillna(False, inplace=True)
    
    for slide in train_slides:
        if slide in bool_df.index:
            bool_df.loc[slide, 'train'] = True
    for slide in val_slides:
        if slide in bool_df.index:
            bool_df.loc[slide, 'val'] = True
    for slide in test_slides:
        if slide in bool_df.index:
            bool_df.loc[slide, 'test'] = True
    
    bool_file = os.path.join(split_dir, f'splits_{i}_bool.csv')
    bool_df.to_csv(bool_file)
    
    # Create and save descriptor
    descriptor_df = pd.DataFrame(
        {
            'train': [len(train_slides)],
            'val': [len(val_slides)],
            'test': [len(test_slides)]
        },
        index=['count']
    )
    
    # Add duration statistics
    train_durations = data[data['slide_id'].isin(train_slides)]['duration'].values
    val_durations = data[data['slide_id'].isin(val_slides)]['duration'].values
    test_durations = data[data['slide_id'].isin(test_slides)]['duration'].values
    
    descriptor_df.loc['duration_mean'] = [
        np.mean(train_durations) if len(train_durations) > 0 else 0,
        np.mean(val_durations) if len(val_durations) > 0 else 0,
        np.mean(test_durations) if len(test_durations) > 0 else 0
    ]
    
    descriptor_df.loc['duration_std'] = [
        np.std(train_durations) if len(train_durations) > 0 else 0,
        np.std(val_durations) if len(val_durations) > 0 else 0,
        np.std(test_durations) if len(test_durations) > 0 else 0
    ]
    
    descriptor_file = os.path.join(split_dir, f'splits_{i}_descriptor.csv')
    descriptor_df.to_csv(descriptor_file)
    
    print(f"Split {i} saved:")
    print(f"  Train: {len(train_slides)} slides")
    print(f"  Val: {len(val_slides)} slides")
    print(f"  Test: {len(test_slides)} slides")
    print(f"  Duration mean - Train: {np.mean(train_durations):.2f}, Val: {np.mean(val_durations):.2f}, Test: {np.mean(test_durations):.2f}")
    print()

print(f"All splits saved to {split_dir}")
