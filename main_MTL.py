from __future__ import print_function
import argparse
import os
from utils.file_utils import save_pkl
from utils.utils import *
from utils.core_utils import train_mtl
from datasets.dataset_generic import Generic_MIL_Dataset, Generic_MTL_Dataset
import torch
import pandas as pd
import numpy as np

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Multi-Task Learning Training')
parser.add_argument('--data_root_dir', type=str, default=None, help='data directory')
parser.add_argument('--data_folder_s', type=str, default=None, help='dir under data directory' )
parser.add_argument('--data_folder_l', type=str, default=None, help='dir under data directory' )
parser.add_argument('--max_epochs', type=int, default=20, help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 0.0001)')
parser.add_argument('--label_frac', type=float, default=1.0, help='fraction of training labels (default: 1.0)')
parser.add_argument('--seed', type=int, default=0, help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=5, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--resume', action='store_true', default=False, help='resume from unfinished fold by checking existing result files')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default=None)
parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--early_stopping_metric', type=str, choices=['loss', 'concordance_time_dependent', 'ipw_concordance_time_dependent', 'brier_score', 'c_index'], default='loss', help='metric to use for early stopping')
parser.add_argument('--early_stopping_mode', type=str, choices=['min', 'max'], default=None, help='mode for early stopping (min: smaller is better, max: larger is better). If not specified, mode is automatically determined based on metric type')
parser.add_argument('--early_stopping_patience', type=int, default=20, help='number of epochs to wait before stopping if no improvement (default: 20)')
parser.add_argument('--early_stopping_stop_epoch', type=int, default=0, help='earliest epoch possible for stopping (default: 80)')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--scheduler_type', type=str, choices=['plateau', 'step'], default='plateau', 
                    help='Scheduler type: plateau (ReduceLROnPlateau) or step (StepLR)')
parser.add_argument('--scheduler_step_size', type=int, default=3, 
                    help='Step size for StepLR scheduler (default: 3)')
parser.add_argument('--scheduler_gamma', type=float, default=0.9, 
                    help='Gamma (decay factor) for StepLR scheduler (default: 0.9)')
parser.add_argument('--drop_out', action='store_true', default=False, help='enabel dropout (p=0.25)')
parser.add_argument('--model_type', type=str, choices=['ViLa_MTL_MIL'], default='ViLa_MTL_MIL', help='type of model')
parser.add_argument('--mode', type=str, choices=['transformer'], default='transformer')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--reg', type=float, default=1e-5, help='weight decay (default: 1e-5)')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce', 'focal'], default='ce')
parser.add_argument('--batch_size', type=int, default=4, help='batch size (default: 4)')
parser.add_argument('--task', type=str)
parser.add_argument("--text_prompt_file", type=str, default=None, 
                    help='CSV file with multiple columns, each column is text prompts for one auxiliary task')
parser.add_argument("--prototype_number", type=int, default=16, help='number of prototypes')
parser.add_argument("--cox_weight", type=float, default=1.0, help='weight for Cox regression loss')
parser.add_argument("--aux_task_weights", type=str, default=None, 
                    help='comma-separated weights for auxiliary tasks (e.g., "1.0,1.0,0.5")')
parser.add_argument("--cox_duration_col", type=str, default='duration', 
                    help='column name for Cox regression duration in CSV')
parser.add_argument("--cox_event_col", type=str, default='event', 
                    help='column name for Cox regression event indicator in CSV')
parser.add_argument("--aux_label_cols", type=str, default=None,
                    help='comma-separated column names for auxiliary task labels in CSV (e.g., "label1,label2")')
parser.add_argument("--mil_aggregator", type=str, choices=['amil', 'prototype', 'baseline'], default='amil',
                    help='MIL aggregator type: amil (traditional attention-based), prototype (original prototype-guided), or baseline (from MIL_BASELINE-main12.9)')
parser.add_argument("--text_extractor", type=str, choices=['clip', 'conch'], default='clip',
                    help='Text extractor type: clip (using CLIP model) or conch (using Conch pathology model)')
parser.add_argument("--conch_checkpoint", type=str, default='/home/hzxu/CONCH/checkpoints/CONCH/',
                    help='Path to Conch model checkpoint directory')
parser.add_argument("--freeze_conch", type=lambda x: x.lower() in ('true', '1', 't', 'y', 'yes'), default=True,
                    help='Whether to freeze the Conch text encoder (default: True). Set to False to fine-tune the last layer of Conch.')
parser.add_argument("--conch_unfreeze_layers", type=int, default=1,
                    help='Number of last layers to unfreeze in Conch when freeze_conch is False (default: 1)')
parser.add_argument("--aux_task_losses", type=str, default=None, 
                    help='comma-separated loss functions for auxiliary tasks (e.g., "ce,ce,ordinal")')
parser.add_argument("--record_loss", action='store_true', default=False, help='enable detailed loss recording')
parser.add_argument("--mil_method", type=str, default='AB_MIL',
                    help='MIL method to use when mil_aggregator is set to baseline (e.g., AB_MIL, CLAM_MB_MIL)')
parser.add_argument("--mil_config", type=str, default=None,
                    help='Path to YAML config file for MIL baseline methods (optional)')
parser.add_argument("--main_task", type=str, choices=['cox', 'binary', 'mdn', 'none'], default='cox',
                    help='Main task type: cox (survival prediction), binary (排斥反应预测), mdn (mixture density network for survival), or none (no main task)')
parser.add_argument("--main_task_weight", type=float, default=1.0,
                    help='Weight for main task loss')
parser.add_argument("--mdn_K", type=int, default=50,
                    help='Number of mixture components for MDN, 100')
parser.add_argument("--mdn_mu_max", type=float, default=200.0,
                    help='Maximum value for mu initialization in MDN (default: 200)')
parser.add_argument("--mdn_mu_min", type=float, default=0.0,
                    help='Minimum value for mu initialization in MDN (default: 0)')
parser.add_argument("--mdn_sigma_init", type=float, default=1.0,
                    help='Initial value for sigma in MDN (default: 1.0)')
parser.add_argument("--mdn_sigma_min", type=float, default=4.0,
                    help='Minimum value for sigma in MDN (default: 4.0). Sigma will be clamped to >= this value.')
parser.add_argument("--mdn_hidden_size", type=int, default=256,
                    help='Hidden size for MDN MLP')
parser.add_argument("--mdn_param_share", type=str, default='mu,sigma',
                    help='Comma-separated list of parameters to share across samples for MDN (e.g., "mu,sigma")')
parser.add_argument("--mdn_dropout", type=float, default=0.1,
                    help='Dropout rate for MDN')
parser.add_argument("--mdn_mode", type=str, choices=['separate', 'unified', 'shared_weight'], default='separate',
                    help='MDN mode: separate (original: two independent MDNs), unified (fusion then single MDN), shared_weight (shared weight head + separate mu/sigma)')
parser.add_argument("--mdn_fixed_mu", type=int, default=0,
                    help='Whether to fix mu in MDN (1=fixed, 0=trainable). When fixed, mu is uniformly distributed in [mdn_mu_min, mdn_mu_max].')
parser.add_argument("--mdn_fixed_sigma", type=int, default=0,
                    help='Whether to fix sigma in MDN (1=fixed, 0=trainable). When fixed, sigma is set to mdn_sigma_init.')
parser.add_argument("--zero_shot_tasks", type=str, default=None,
                    help='comma-separated list of zero-shot task indices (0-based) and their zero-shot classes (e.g., "1:3,2:3" means task 1 and task 2 have zero-shot class 3)')
parser.add_argument("--use_text_prompts", type=lambda x: x.lower() in ('true', '1', 't', 'y', 'yes'), default=True,
                    help='whether to use text prompts for auxiliary tasks (default: True)')

# Text Prompt related hyperparameters
parser.add_argument("--n_ctx", type=int, default=16,
                    help='Number of learnable context tokens in PromptLearner (default: 16)')
parser.add_argument("--class_token_position", type=str, choices=['end', 'middle', 'front'], default='end',
                    help='Position of class token in text prompts (default: end)')
parser.add_argument("--ctx_init_std", type=float, default=0.02,
                    help='Standard deviation for initializing context vectors (default: 0.02)')
parser.add_argument("--prototype_init_std", type=float, default=0.02,
                    help='Standard deviation for initializing prototype vectors (default: 0.02)')

# Model architecture hyperparameters
parser.add_argument("--num_heads", type=int, default=1,
                    help='Number of attention heads in multihead attention layers (default: 1)')
parser.add_argument("--input_size", type=int, default=512,
                    help='Input feature dimension (default: 512)')
parser.add_argument("--hidden_size", type=int, default=192,
                    help='Hidden layer dimension (default: 192)')

# Cross-scale contrastive learning hyperparameters
parser.add_argument("--use_cross_scale_contrastive", type=lambda x: x.lower() in ('true', '1', 't', 'y', 'yes'), default=False,
                    help='Enable cross-scale contrastive learning (default: False)')
parser.add_argument("--cross_scale_weight", type=float, default=0.1,
                    help='Weight for cross-scale contrastive loss (default: 0.1)')
parser.add_argument("--contrast_temperature", type=float, default=0.5,
                    help='Temperature parameter for contrastive loss (default: 0.5)')
parser.add_argument("--contrast_intra_weight", type=float, default=1.0,
                    help='Weight for intra-image contrast (MSE loss) (default: 1.0)')
parser.add_argument("--contrast_inter_weight", type=float, default=0.5,
                    help='Weight for inter-image contrast (InfoNCE loss) (default: 0.5)')

# Consistency loss hyperparameters
parser.add_argument("--use_consistency", type=lambda x: x.lower() in ('true', '1', 't', 'y', 'yes'), default=False,
                    help='Enable consistency loss (default: False)')
parser.add_argument("--consistency_weight", type=float, default=0.1,
                    help='Weight for consistency loss (default: 0.1)')
parser.add_argument("--consistency_noise_std", type=float, default=0.1,
                    help='Noise standard deviation for consistency loss (default: 0.1)')

args = parser.parse_args()

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

# Load multi-task text prompts
num_aux_tasks = 0
task_text_prompts = []
aux_task_weights = []
aux_task_losses = []
aux_task_num_classes = []
zero_shot_config = {}

if args.aux_label_cols is not None:
    aux_label_cols_list = [col.strip() for col in args.aux_label_cols.split(',')]
    num_aux_tasks = len(aux_label_cols_list)
    
    if args.use_text_prompts:
        print('\nLoading multi-task text prompts...')
        if args.text_prompt_file is None:
            raise ValueError("--text_prompt_file must be provided for MTL training with text prompts")

        # Read text prompt file (CSV format: each column is text prompts for one task)
        # text_prompt_df = pd.read_csv(args.text_prompt_file, header=None)
        text_prompt_df = pd.read_csv(args.text_prompt_file, header=0)                                   # Use header=0 to treat first row as column names
        
        # Parse aux_label_cols to determine which columns to use from text_prompt_file
        # Format: aux_label_4,aux_label_5 -> should select task4, task5 columns
        if args.aux_label_cols is not None:
            aux_label_cols_list = [col.strip() for col in args.aux_label_cols.split(',')]
            # Extract task indices from aux_label_cols (e.g., "aux_label_4" -> 4)
            task_indices = []
            for col in aux_label_cols_list:
                # Extract number from column name like "aux_label_4"
                if '_' in col:
                    try:
                        task_idx = int(col.split('_')[-1])
                        task_indices.append(task_idx)
                    except ValueError:
                        # If no number found, use sequential index
                        task_indices.append(len(task_indices))
            print(f"Using task indices from aux_label_cols: {task_indices}")
        else:
            task_indices = list(range(num_aux_tasks))
        
        # Parse text prompts for each task based on selected task indices
        # Map task index to column name: 0 -> task1, 1 -> task2, etc.
        for task_order_idx, task_idx in enumerate(task_indices):
            col_name = f'task{task_idx + 1}'  # Convert to 1-based column name
            if col_name in text_prompt_df.columns:
                prompts = np.array(text_prompt_df[col_name]).squeeze()
                # Remove NaN values
                prompts = prompts[~pd.isna(prompts)]
                prompts = prompts.tolist()
                task_text_prompts.append(prompts)
                print(f'Task {task_order_idx + 1} (from {col_name}): {len(prompts)} text prompts loaded')
            else:
                # Fallback to column index if column name not found
                if task_idx < len(text_prompt_df.columns):
                    prompts = np.array(text_prompt_df.iloc[:, task_idx]).squeeze()
                    prompts = prompts[~pd.isna(prompts)]
                    prompts = prompts.tolist()
                    task_text_prompts.append(prompts)
                    print(f'Task {task_order_idx + 1} (col {task_idx}): {len(prompts)} text prompts loaded')
                else:
                    raise ValueError(f"Task index {task_idx} out of range for text_prompt_file with {len(text_prompt_df.columns)} columns")
    else:
        # When not using text prompts, create dummy prompts for compatibility
        print('\nNot using text prompts for auxiliary tasks')
        for task_idx in range(num_aux_tasks):
            # Create empty prompts list for compatibility
            task_text_prompts.append([])
            print(f'Task {task_idx + 1}: no text prompts (using traditional classification)')

    # Parse auxiliary task weights
    if args.aux_task_weights is not None:
        aux_task_weights = [float(w.strip()) for w in args.aux_task_weights.split(',')]
        if len(aux_task_weights) != num_aux_tasks:
            raise ValueError(f"Number of auxiliary task weights ({len(aux_task_weights)}) "
                            f"does not match number of tasks ({num_aux_tasks})")
    else:
        aux_task_weights = [1.0] * num_aux_tasks

    # Parse auxiliary task losses
    if args.aux_task_losses is not None:
        aux_task_losses = [loss.strip() for loss in args.aux_task_losses.split(',')]
        if len(aux_task_losses) != num_aux_tasks:
            raise ValueError(f"Number of auxiliary task losses ({len(aux_task_losses)}) "
                            f"does not match number of tasks ({num_aux_tasks})")
    else:
        aux_task_losses = ['ce'] * num_aux_tasks

    print(f'Number of auxiliary tasks: {num_aux_tasks}')
    print(f'Auxiliary task weights: {aux_task_weights}')
    print(f'Auxiliary task losses: {aux_task_losses}')

    # Parse zero-shot task configuration
    # Format: "task_idx:class_idx,task_idx:class_idx" (e.g., "1:3,2:3")
    if args.zero_shot_tasks is not None:
        zero_shot_items = [item.strip() for item in args.zero_shot_tasks.split(',')]
        for item in zero_shot_items:
            if ':' in item:
                task_idx, class_idx = item.split(':')
                task_idx = int(task_idx.strip())
                class_idx = int(class_idx.strip())
                if task_idx not in zero_shot_config:
                    zero_shot_config[task_idx] = []
                zero_shot_config[task_idx].append(class_idx)
        
        print(f'Zero-shot task configuration: {zero_shot_config}')
        for task_idx, zero_shot_classes in zero_shot_config.items():
            print(f'  Task {task_idx}: zero-shot classes {zero_shot_classes}')
    else:
        zero_shot_config = {}

    # Determine number of classes for each auxiliary task
    if args.use_text_prompts:
        # When using text prompts, infer from text prompts
        for prompts in task_text_prompts:
            # Number of classes = number of unique prompts / 2
            num_classes = len(set(prompts)) // 2
            aux_task_num_classes.append(num_classes)
            print(f'Task with {num_classes} classes (from text prompts)')
    else:
        # When not using text prompts, infer from data labels
        # Read the CSV file to get actual label values
        csv_path = None
        if args.task == 'MTL':
            csv_path = 'dataset_csv/MTL182.csv'
        elif args.task == 'MTLtj':
            csv_path = 'dataset_csv/tj.csv'
        elif args.task == 'crcos':
            csv_path = 'dataset_csv/crc_outcomesOS.csv'
        elif args.task == 'crcpfs':
            csv_path = 'dataset_csv/crc_outcomesPFS.csv'
        elif args.task == 'sr386':
            csv_path = 'dataset_csv/sr386OS.csv'
        else:
            raise NotImplementedError(f"Unknown task: {args.task}")
        
        if csv_path and os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            for task_idx, aux_label_col in enumerate(aux_label_cols_list):
                if aux_label_col in df.columns:
                    unique_labels = df[aux_label_col].dropna().unique()
                    num_classes = len(unique_labels)
                    aux_task_num_classes.append(num_classes)
                    print(f'Task {task_idx} ({aux_label_col}): {num_classes} classes (from data)')
                else:
                    # Fallback to default if column not found
                    aux_task_num_classes.append(2)
                    print(f'Task {task_idx}: defaulting to 2 classes')
        else:
            # Fallback to default if CSV not found
            for task_idx in range(num_aux_tasks):
                aux_task_num_classes.append(2)
                print(f'Task {task_idx}: defaulting to 2 classes')
else:
    print('\nNo auxiliary tasks specified, running only main task')
    num_aux_tasks = 0
    task_text_prompts = []
    aux_task_weights = []
    aux_task_losses = []
    aux_task_num_classes = []
    zero_shot_config = {}

settings = {'num_splits': args.k,
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs,
            'results_dir': args.results_dir,
            'lr': args.lr,
            'experiment': args.exp_code,
            'label_frac': args.label_frac,
            'seed': args.seed,
            'model_type': args.model_type,
            'mode': args.mode,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'opt': args.opt,
            'num_aux_tasks': num_aux_tasks,
            'aux_task_num_classes': aux_task_num_classes,
            'cox_weight': args.cox_weight,
            'aux_task_weights': aux_task_weights,
            'aux_task_losses': aux_task_losses,
            'main_task': args.main_task,
            'main_task_weight': args.main_task_weight,
            'zero_shot_config': zero_shot_config}

# Store task configuration in args
args.task_text_prompts = task_text_prompts
args.aux_task_num_classes = aux_task_num_classes
args.num_aux_tasks = num_aux_tasks
args.aux_task_losses = aux_task_losses
args.main_task = args.main_task
args.main_task_weight = args.main_task_weight
args.zero_shot_config = zero_shot_config

# Parse auxiliary label columns
if args.aux_label_cols is not None:
    args.aux_label_cols = [col.strip() for col in args.aux_label_cols.split(',')]
    if num_aux_tasks > 0 and len(args.aux_label_cols) != num_aux_tasks:
        raise ValueError(f"Number of auxiliary label columns ({len(args.aux_label_cols)}) "
                        f"does not match number of tasks ({num_aux_tasks})")
else:
    # Default: assume labels are in columns named 'aux_label_0', 'aux_label_1', etc.
    args.aux_label_cols = [f'aux_label_{i}' for i in range(num_aux_tasks)]

print('\nLoad Dataset')

# Load dataset
csv_path = None
if args.task == 'MTL':
    csv_path = 'dataset_csv/MTL182.csv'
elif args.task == 'MTLtj':
    csv_path = 'dataset_csv/tj.csv'
elif args.task == 'crcos':
    csv_path = 'dataset_csv/crc_outcomesOS.csv'
elif args.task == 'crcpfs':
    csv_path = 'dataset_csv/crc_outcomesPFS.csv'
elif args.task == 'sr386':
    csv_path = 'dataset_csv/sr386OS.csv'
else:
    raise NotImplementedError(f"Unknown task: {args.task}")

dataset = Generic_MTL_Dataset(csv_path = csv_path,
                              mode = args.mode,
                              data_dir_s = os.path.join(args.data_root_dir, args.data_folder_s),
                              data_dir_l = os.path.join(args.data_root_dir, args.data_folder_l),
                              shuffle = False,
                              print_info = True,
                              aux_label_cols=args.aux_label_cols)

# Identify zero-shot samples (samples belonging to zero-shot classes)
# These samples will be forced into the test set
custom_test_ids = []
if zero_shot_config:
    print("\nIdentifying zero-shot samples...")
    for task_idx, zero_shot_classes in zero_shot_config.items():
        for class_idx in zero_shot_classes:
            indices = dataset.get_task_class_indices(task_idx, class_idx)
            if len(indices) > 0:
                print(f"  Task {task_idx}, Class {class_idx}: {len(indices)} samples found")
                custom_test_ids.extend(indices.tolist())
    
    # Remove duplicates
    custom_test_ids = list(set(custom_test_ids))
    print(f"Total zero-shot samples to be placed in test set: {len(custom_test_ids)}")
    
    # Print sample IDs for verification
    if len(custom_test_ids) > 0:
        sample_ids = dataset.slide_data.iloc[custom_test_ids]['slide_id'].tolist()
        print(f"Sample IDs: {sample_ids}")
else:
    custom_test_ids = None

if not os.path.exists(args.results_dir):
    os.makedirs(args.results_dir)

args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.exists(args.results_dir):
    os.makedirs(args.results_dir)

if args.split_dir is None:
    args.split_dir = os.path.join('splits', args.task+'_{}'.format(int(args.label_frac*100)))
else:
    args.split_dir = os.path.join('splits', args.split_dir)

print('split_dir: ', args.split_dir)
assert os.path.isdir(args.split_dir)

settings.update({'split_dir': args.split_dir})

with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
    import sys
    print('Command line: ' + ' '.join(sys.argv), file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))


def main(args):
    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    all_test_metrics = []
    all_val_metrics = []
    folds = np.arange(start, end)
    
    # Store zero-shot sample IDs for reference
    zero_shot_sample_ids = []
    if custom_test_ids is not None:
        zero_shot_sample_ids = dataset.slide_data.iloc[custom_test_ids]['slide_id'].tolist()
    
    for i in folds:
        # Check if this fold is already completed (when resume mode is enabled)
        if args.resume:
            result_file = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
            if os.path.exists(result_file):
                print(f"Fold {i} already completed, skipping...")
                # Load existing metrics if available
                try:
                    import pickle
                    with open(result_file, 'rb') as f:
                        results = pickle.load(f)
                    if 'test_metrics' in results:
                        all_test_metrics.append(results['test_metrics'])
                    if 'val_metrics' in results:
                        all_val_metrics.append(results['val_metrics'])
                except:
                    pass
                continue
        
        seed_torch(args.seed)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(
            from_id=False, csv_path='{}/splits_{}.csv'.format(args.split_dir, i)
        )
        
        # If zero-shot samples are not in test set, move them there
        if custom_test_ids is not None:
            # Get current test set slide IDs
            current_test_ids = test_dataset.slide_data['slide_id'].tolist()
            
            # Check if zero-shot samples are in train or val sets
            train_data = train_dataset.slide_data
            val_data = val_dataset.slide_data
            test_data = test_dataset.slide_data
            
            # Find zero-shot samples in train or val sets
            train_zero_shot_mask = train_data['slide_id'].isin(zero_shot_sample_ids)
            val_zero_shot_mask = val_data['slide_id'].isin(zero_shot_sample_ids)
            
            if train_zero_shot_mask.any() or val_zero_shot_mask.any():
                # Move zero-shot samples from train/val to test
                train_zero_shot_samples = train_data[train_zero_shot_mask]
                val_zero_shot_samples = val_data[val_zero_shot_mask]
                
                # Remove from train and val
                train_data = train_data[~train_zero_shot_mask].reset_index(drop=True)
                val_data = val_data[~val_zero_shot_mask].reset_index(drop=True)
                
                # Add to test
                test_data = pd.concat([test_data, train_zero_shot_samples, val_zero_shot_samples], 
                                     ignore_index=True)
                
                # Update datasets
                train_dataset.slide_data = train_data
                val_dataset.slide_data = val_data
                test_dataset.slide_data = test_data
                
                print(f"Moved {len(train_zero_shot_samples)} zero-shot samples from train to test")
                print(f"Moved {len(val_zero_shot_samples)} zero-shot samples from val to test")
        
        datasets = (train_dataset, val_dataset, test_dataset)
        results, test_metrics, val_metrics = train_mtl(datasets, i, args)

        all_test_metrics.append(test_metrics)
        all_val_metrics.append(val_metrics)
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)

    # Aggregate results across folds
    # Calculate mean and std for each metric
    metric_names = list(all_test_metrics[0].keys())
    summary_dict = {}
    for metric_name in metric_names:
        values = [metrics[metric_name] for metrics in all_test_metrics]
        summary_dict[f'{metric_name}_mean'] = np.mean(values)
        summary_dict[f'{metric_name}_std'] = np.std(values)
    
    result_df = pd.DataFrame([summary_dict])
    result_df.to_csv(os.path.join(args.results_dir, 'result.csv'), index=False)
    
    # Save detailed results
    detailed_df = pd.DataFrame(all_test_metrics)
    detailed_df.insert(0, 'fold', folds)
    detailed_df.to_csv(os.path.join(args.results_dir, 'summary.csv'), index=False)

    print("Results saved to:", args.results_dir)


if __name__ == "__main__":
    results = main(args)
    print("finished!")
    print("end script")
