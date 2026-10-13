import numpy as np
import torch
from utils.utils import *
import os
from datasets.dataset_generic import save_splits
from models.model_mil import MIL_fc, MIL_fc_mc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, balanced_accuracy_score
from utils.loss_utils import FocalLoss
import pandas as pd
import yaml
from utils.survival_metrics import CIndexMeter, IPWCIndexMeter, BrierScoreMeter, ConcordanceMeter


def load_mil_config(mil_config_path):
    """
    Load MIL baseline configuration from YAML file
    
    Args:
        mil_config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing MIL configuration parameters
    """
    if mil_config_path is None:
        return None
    
    try:
        with open(mil_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract model-specific parameters
        mil_config = {}
        if 'Model' in config:
            model_config = config['Model']
            
            # Common parameters
            mil_config['in_dim'] = model_config.get('in_dim', 1024)
            mil_config['L'] = model_config.get('L', 512)
            mil_config['D'] = model_config.get('D', 128)
            mil_config['dropout'] = model_config.get('dropout', 0.1)
            mil_config['act'] = model_config.get('act', 'relu')
            
            # CLAM-specific parameters
            mil_config['k_sample'] = model_config.get('k_sample', 8)
            mil_config['size_arg'] = model_config.get('size_arg', 'small')
            mil_config['instance_loss_fn'] = model_config.get('instance_loss_fn', 'ce')
            mil_config['subtyping'] = model_config.get('subtyping', True)
            mil_config['instance_eval'] = model_config.get('instance_eval', True)
            mil_config['gate'] = model_config.get('gate', True)
            mil_config['bag_weight'] = model_config.get('bag_weight', 0.7)
            
            # Other method-specific parameters can be added here
            mil_config.update({k: v for k, v in model_config.items() 
                             if k not in ['in_dim', 'L', 'D', 'dropout', 'act', 
                                         'k_sample', 'size_arg', 'instance_loss_fn', 
                                         'subtyping', 'instance_eval', 'gate', 'bag_weight',
                                         'optimizer', 'criterion', 'scheduler']})
        
        return mil_config
        
    except Exception as e:
        print(f"Warning: Failed to load MIL config from {mil_config_path}: {str(e)}")
        return None

def save_loss_records(loss_records, save_path):
    """
    Save loss records to a CSV file
    
    Args:
        loss_records (list): List of loss record dictionaries
        save_path (str): Path to save the CSV file
    """
    if not loss_records:
        return
    
    # Create DataFrame from loss records
    df = pd.DataFrame(loss_records)
    
    # Check if file exists to decide whether to write header
    file_exists = os.path.exists(save_path)
    
    # Save to CSV
    df.to_csv(save_path, mode='a' if file_exists else 'w', index=False, header=not file_exists)


class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

def save_model_state(model, ckpt_name):
    """
    Save model state dict, using trainable parameters only for ViLa_MIL models
    to reduce model size from ~1.5GB to ~10MB
    """
    if hasattr(model, 'get_trainable_state_dict'):
        # Use trainable state dict for ViLa_MIL models to save space
        torch.save(model.get_trainable_state_dict(), ckpt_name)
        print(f"Model saved using trainable state dict (reduced size)")
    else:
        torch.save(model.state_dict(), ckpt_name)

def load_model_state(model, ckpt_name):
    """
    Load model state dict, handling both full and trainable-only state dicts
    """
    state_dict = torch.load(ckpt_name, map_location='cpu')
    
    if hasattr(model, 'load_trainable_state_dict'):
        # ViLa_MIL models use trainable state dict
        model.load_trainable_state_dict(state_dict)
        print(f"Model loaded using trainable state dict")
    else:
        model.load_state_dict(state_dict)
    
    return model


class EarlyStopping:
    """Early stops the training if validation metric doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False, mode='min'):
        """
        Args:
            patience (int): How long to wait after last time validation metric improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation metric improvement. 
                            Default: False
            mode (str): 'min' for metrics that should decrease (like loss), 'max' for metrics that should increase (like accuracy, concordance)
                            Default: 'min'
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        if mode == 'min':
            self.best_value = np.Inf
        else:
            self.best_value = -np.Inf

    def __call__(self, epoch, val_metric, model, ckpt_name = 'checkpoint.pt'):

        if self.mode == 'min':
            score = -val_metric
        else:
            score = val_metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_metric, model, ckpt_name)
        elif score <= self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_metric, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_metric, model, ckpt_name):
        '''Saves model when validation metric improves.'''
        if self.verbose:
            if self.mode == 'min':
                print(f'Validation metric decreased ({self.best_value:.6f} --> {val_metric:.6f}).  Saving model ...')
            else:
                print(f'Validation metric increased ({self.best_value:.6f} --> {val_metric:.6f}).  Saving model ...')
        save_model_state(model, ckpt_name)
        self.best_value = val_metric

def train(datasets, cur, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes = args.n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    elif args.bag_loss == 'focal':
        loss_fn = FocalLoss().cuda()
    else:
        loss_fn = nn.CrossEntropyLoss()
    print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}

    if args.model_type == 'ViLa_MIL':
        import ml_collections
        from models.model_ViLa_MIL import ViLa_MIL_Model
        config = ml_collections.ConfigDict()
        config.text_extractor = getattr(args, 'text_extractor', 'clip')
        config.input_size = 512 if config.text_extractor == 'conch' else 1024
        config.hidden_size = 192
        config.text_prompt = args.text_prompt
        config.prototype_number = args.prototype_number
        config.conch_checkpoint = getattr(args, 'conch_checkpoint', '/home/hzxu/CONCH/checkpoints/CONCH/')
        config.freeze_conch = getattr(args, 'freeze_conch', True)
        config.conch_unfreeze_layers = getattr(args, 'conch_unfreeze_layers', 1)
        model_dict = {'config': config, 'num_classes':args.n_classes}
        model = ViLa_MIL_Model(**model_dict)

    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)


    if hasattr(model, "relocate"):
        model.relocate()
    else:
        model = model.to(torch.device('cuda:0'))
    print('Done!')
    # print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')

    print('\nInit scheduler ...', end=' ')
    if args.scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)
        scheduler_type = 'plateau'
    else:  # step
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma, verbose=True)
        scheduler_type = 'step'
    print(f'Done! (using {scheduler_type} scheduler)')

    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample, mode=args.mode)
    val_loader = get_split_loader(val_split,  testing = args.testing, mode=args.mode)
    test_loader = get_split_loader(test_split, testing = args.testing, mode=args.mode)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience=20, stop_epoch=80, verbose=True)
    else:
        early_stopping = None
    print('Done!')

    for epoch in range(args.max_epochs):
        train_loop(args, epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn)
        stop = validate(cur, epoch, model, val_loader, args.n_classes, 
            early_stopping, writer, loss_fn, args.results_dir)
        
        # Update learning rate scheduler
        if scheduler_type == 'step':
            scheduler.step()
        
        if stop: 
            break

    if args.early_stopping: 
        load_model_state(model, os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
    else:
        save_model_state(model, os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    _, val_error, val_auc, _, val_f1, val_bacc = summary(args.mode, model, val_loader, args.n_classes)
    print('Val error: {:.4f}, ROC AUC: {:.4f}, F1: {:.4f}, BACC: {:.4f}'.format(val_error, val_auc, val_f1, val_bacc))

    results_dict, test_error, test_auc, acc_logger, test_f1, test_bacc = summary(args.mode, model, test_loader, args.n_classes)
    print('Test error: {:.4f}, ROC AUC: {:.4f}, F1: {:.4f}, BACC: {:.4f}'.format(test_error, test_auc, test_f1, test_bacc))

    each_class_acc = []
    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        each_class_acc.append(acc)
        print('class {}: acc {:.4f}, correct {}/{}'.format(i, acc, correct, count))

        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
        writer.add_scalar('final/val_bacc', val_bacc, 0)
        writer.add_scalar('final/test_bacc', test_bacc, 0)
        writer.close()
    return results_dict, test_auc, val_auc, 1-test_error, 1-val_error, each_class_acc, test_f1, test_bacc


def train_loop(args, epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None):

    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.
    all_pred = []
    all_label = []

    print('\n')
    for batch_idx, (data_s, coord_s, data_l, coords_l, label) in enumerate(loader):
        data_s, coord_s, data_l, coords_l, label = data_s.to(device), coord_s.to(device), data_l.to(device), coords_l.to(device), label.to(device)
        _, Y_hat, loss = model(data_s, coord_s, data_l, coords_l, label)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        acc_logger.log(Y_hat, label)
        loss_value = loss.item()
        train_loss += loss_value
        error = calculate_error(Y_hat, label)
        train_error += error
        all_pred.append(Y_hat.cpu())
        all_label.append(label.cpu())

    train_loss /= len(loader)
    train_error /= len(loader)
    all_label_np = np.array([l.item() if torch.is_tensor(l) else l for l in all_label]).astype(int)
    all_pred_np = np.array([p.item() if torch.is_tensor(p) else p for p in all_pred]).astype(int)
    train_f1 = f1_score(all_label_np, all_pred_np, average='macro')
    train_bacc = balanced_accuracy_score(all_label_np, all_pred_np)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}, train_f1: {:.4f}, train_bacc: {:.4f}'.format(epoch, train_loss, train_error, train_f1, train_bacc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {:.4f}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/f1', train_f1, epoch)
        writer.add_scalar('train/bacc', train_bacc, epoch)
   
def validate(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    all_pred = []
    all_label = []
    with torch.no_grad():
        for batch_idx, (data_s, coord_s, data_l, coords_l, label) in enumerate(loader):
            data_s, coord_s, data_l, coords_l, label = data_s.to(device, non_blocking=True), coord_s.to(device, non_blocking=True), \
                                                                  data_l.to(device, non_blocking=True), coords_l.to(device, non_blocking=True), \
                                                                  label.to(device, non_blocking=True)
            Y_prob, Y_hat, loss = model(data_s, coord_s, data_l, coords_l, label)

            acc_logger.log(Y_hat, label)
            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
            all_pred.append(Y_hat.cpu())
            all_label.append(label.cpu())

    val_error /= len(loader)
    val_loss /= len(loader)
    all_label_np = np.array([l.item() if torch.is_tensor(l) else l for l in all_label]).astype(int)
    all_pred_np = np.array([p.item() if torch.is_tensor(p) else p for p in all_pred]).astype(int)
    val_f1 = f1_score(all_label_np, all_pred_np, average='macro')
    val_bacc = balanced_accuracy_score(all_label_np, all_pred_np)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])

    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')

    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/f1', val_f1, epoch)
        writer.add_scalar('val/bacc', val_bacc, epoch)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}, f1: {:.4f}, bacc: {:.4f}'.format(val_loss, val_error, auc, val_f1, val_bacc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {:.4f}, correct {}/{}'.format(i, acc, correct, count))

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_error, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def summary(mode, model, loader, n_classes):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.
    test_f1 = 0.
    all_pred = []
    all_label = []

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    if(mode == 'transformer'):
        for batch_idx, (data_s, coord_s, data_l, coords_l, label) in enumerate(loader):
            data_s, coord_s, data_l, coords_l, label = data_s.to(device), coord_s.to(device), data_l.to(device), coords_l.to(device), label.to(device)
            slide_id = slide_ids.iloc[batch_idx]
            with torch.no_grad():
                Y_prob, Y_hat, loss = model(data_s, coord_s, data_l, coords_l, label)

            acc_logger.log(Y_hat, label)
            probs = Y_prob.cpu().numpy()
            all_probs[batch_idx] = probs
            all_labels[batch_idx] = label.item()

            patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
            error = calculate_error(Y_hat, label)
            test_error += error

            all_pred.append(Y_hat.cpu())
            all_label.append(label.cpu())

        test_error /= len(loader)
        all_label_np = np.array([l.item() if torch.is_tensor(l) else l for l in all_label]).astype(int)
        all_pred_np = np.array([p.item() if torch.is_tensor(p) else p for p in all_pred]).astype(int)
        test_f1 = f1_score(all_label_np, all_pred_np, average='macro')
        test_bacc = balanced_accuracy_score(all_label_np, all_pred_np)

        if n_classes == 2:
            auc = roc_auc_score(all_labels, all_probs[:, 1])
            aucs = []
        else:
            aucs = []
            binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
            for class_idx in range(n_classes):
                if class_idx in all_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                    aucs.append(calc_auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))

            auc = np.nanmean(np.array(aucs))

        print('\nTest Set, test_error: {:.4f}, auc: {:.4f}, f1: {:.4f}, bacc: {:.4f}'.format(test_error, auc, test_f1, test_bacc))
        return patient_results, test_error, auc, acc_logger, test_f1, test_bacc


def train_mtl(datasets, cur, args):
    """
    Multi-task learning training function
    
    Main task: Cox regression (survival prediction)
    Auxiliary tasks: Classification tasks with text prompts
    """
    print('\nTraining Fold {} (Multi-Task Learning)!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)
    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    
    # Convert to MTL splits
    from datasets.dataset_generic import Generic_MTL_Split
    
    # Read the original CSV to get additional columns
    csv_path = None
    if args.task == 'task_tcga_rcc_subtyping':
        csv_path = 'dataset_csv/TCGA_RCC_subtyping.csv'
    elif args.task == 'task_tcga_lung_subtyping':
        csv_path = 'dataset_csv/TCGA_Lung_subtyping.csv'
    elif args.task == 'debug_MTL':
        csv_path = 'dataset_csv/debug_MTL.csv'
    elif args.task == 'sr386':
        csv_path = 'dataset_csv/sr386OS.csv'

    if csv_path and os.path.exists(csv_path):
        full_data = pd.read_csv(csv_path, encoding='utf-8-sig')
        # Merge with split data to get additional columns
        train_data = train_split.slide_data.merge(full_data, on='slide_id', how='left', suffixes=('', '_full'))
        
        # 处理没有验证集的情况
        if val_split is not None:
            val_data = val_split.slide_data.merge(full_data, on='slide_id', how='left', suffixes=('', '_full'))
        else:
            val_data = None
            
        test_data = test_split.slide_data.merge(full_data, on='slide_id', how='left', suffixes=('', '_full'))
    else:
        train_data = train_split.slide_data
        
        # 处理没有验证集的情况
        if val_split is not None:
            val_data = val_split.slide_data
        else:
            val_data = None
            
        test_data = test_split.slide_data
    
    # Determine main task label column based on task type
    main_task_label_col = 'final_label' if args.main_task == 'binary' else None
    
    train_split_mtl = Generic_MTL_Split(
        train_data,
        data_dir_s=train_split.data_dir_s,
        data_dir_l=train_split.data_dir_l,
        mode=args.mode,
        cox_duration_col=args.cox_duration_col,
        cox_event_col=args.cox_event_col,
        aux_label_cols=args.aux_label_cols,
        main_task_type=args.main_task,
        main_task_label_col=main_task_label_col
    )
    # 只有在有验证集时才创建 val_split_mtl
    val_split_mtl = None
    if val_data is not None and val_split is not None:
        val_split_mtl = Generic_MTL_Split(
            val_data,
            data_dir_s=val_split.data_dir_s,
            data_dir_l=val_split.data_dir_l,
            mode=args.mode,
            cox_duration_col=args.cox_duration_col,
            cox_event_col=args.cox_event_col,
            aux_label_cols=args.aux_label_cols,
            main_task_type=args.main_task,
            main_task_label_col=main_task_label_col
        )
    test_split_mtl = Generic_MTL_Split(
        test_data,
        data_dir_s=test_split.data_dir_s,
        data_dir_l=test_split.data_dir_l,
        mode=args.mode,
        cox_duration_col=args.cox_duration_col,
        cox_event_col=args.cox_event_col,
        aux_label_cols=args.aux_label_cols,
        main_task_type=args.main_task,
        main_task_label_col=main_task_label_col
    )
    
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split_mtl)))
    
    # 处理没有验证集的情况
    if val_split_mtl is not None:
        print("Validating on {} samples".format(len(val_split_mtl)))
    else:
        print("Validating on 0 samples (no validation set)")
        
    print("Testing on {} samples".format(len(test_split_mtl)))

    print('\nInit Model...', end=' ')
    if args.model_type == 'ViLa_MTL_MIL':
        import ml_collections
        from models.model_ViLa_MIL_MTL import ViLa_MTL_MIL_Model
        
        config = ml_collections.ConfigDict()
        config.input_size = getattr(args, 'input_size', 512)
        config.hidden_size = getattr(args, 'hidden_size', 192)
        config.prototype_number = args.prototype_number
        config.main_task = args.main_task
        config.main_task_weight = args.main_task_weight
        config.aux_task_weights = args.aux_task_weights  # This is a list
        config.aux_task_losses = getattr(args, 'aux_task_losses', ['ce'] * args.num_aux_tasks)
        config.mil_aggregator = getattr(args, 'mil_aggregator', 'amil')
        config.text_extractor = getattr(args, 'text_extractor', 'clip')
        config.conch_checkpoint = getattr(args, 'conch_checkpoint', '/home/hzxu/CONCH/checkpoints/CONCH/')
        config.freeze_conch = getattr(args, 'freeze_conch', True)
        config.conch_unfreeze_layers = getattr(args, 'conch_unfreeze_layers', 1)
        config.use_text_prompts = getattr(args, 'use_text_prompts', True)
        
        # Text prompt related hyperparameters
        config.n_ctx = getattr(args, 'n_ctx', 16)
        config.class_token_position = getattr(args, 'class_token_position', 'end')
        config.ctx_init_std = getattr(args, 'ctx_init_std', 0.02)
        config.prototype_init_std = getattr(args, 'prototype_init_std', 0.02)
        config.num_heads = getattr(args, 'num_heads', 1)
        
        # Cross-scale contrastive learning hyperparameters
        config.use_cross_scale_contrastive = getattr(args, 'use_cross_scale_contrastive', False)
        config.cross_scale_weight = getattr(args, 'cross_scale_weight', 0.1)
        config.contrast_temperature = getattr(args, 'contrast_temperature', 0.5)
        config.contrast_intra_weight = getattr(args, 'contrast_intra_weight', 1.0)
        config.contrast_inter_weight = getattr(args, 'contrast_inter_weight', 0.5)
        
        # Consistency loss hyperparameters
        config.use_consistency = getattr(args, 'use_consistency', False)
        config.consistency_weight = getattr(args, 'consistency_weight', 0.1)
        config.consistency_noise_std = getattr(args, 'consistency_noise_std', 0.1)
        
        # MDN configuration
        if args.main_task == 'mdn':
            config.mdn_K = getattr(args, 'mdn_K', 100)
            config.mdn_mu_max = getattr(args, 'mdn_mu_max', 200.0)
            config.mdn_mu_min = getattr(args, 'mdn_mu_min', 0.0)
            config.mdn_sigma_init = getattr(args, 'mdn_sigma_init', 1.0)
            config.mdn_sigma_min = getattr(args, 'mdn_sigma_min', 4.0)
            config.mdn_hidden_size = getattr(args, 'mdn_hidden_size', 256)
            mdn_param_share_str = getattr(args, 'mdn_param_share', 'mu,sigma')
            config.mdn_param_share = [p.strip() for p in mdn_param_share_str.split(',')]
            config.mdn_dropout = getattr(args, 'mdn_dropout', 0.1)
            config.mdn_mode = getattr(args, 'mdn_mode', 'separate')
            # 新增：独立固定mu和sigma模式
            config.mdn_fixed_mu = bool(getattr(args, 'mdn_fixed_mu', 0))
            config.mdn_fixed_sigma = bool(getattr(args, 'mdn_fixed_sigma', 0))
        
        # Load MIL baseline configuration if provided
        mil_config_path = getattr(args, 'mil_config', None)
        config.mil_config = load_mil_config(mil_config_path)
        config.mil_method = getattr(args, 'mil_method', 'AB_MIL')
        
        # Create task configurations
        task_configs = []
        if args.num_aux_tasks > 0:
            for task_idx in range(args.num_aux_tasks):
                task_config = {
                    'text_prompts': args.task_text_prompts[task_idx],
                    'num_classes': args.aux_task_num_classes[task_idx]
                }
                task_configs.append(task_config)
        config.task_configs = task_configs
        
        # Add zero-shot configuration
        config.zero_shot_config = args.zero_shot_config
        
        model = ViLa_MTL_MIL_Model(config)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    if hasattr(model, "relocate"):
        model.relocate()
    else:
        model = model.to(torch.device('cuda:0'))
    print('Done!')
    # print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')

    print('\nInit scheduler ...', end=' ')
    if args.scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)
        scheduler_type = 'plateau'
    else:  # step
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma, verbose=True)
        scheduler_type = 'step'
    print(f'Done! (using {scheduler_type} scheduler)')

    print('\nInit Loaders...', end=' ')
    # Create custom collate function for MTL
    def mtl_collate_fn(batch):
        features_s_list = []
        coords_s_list = []
        features_l_list = []
        coords_l_list = []
        cox_durations = []
        cox_events = []
        main_task_labels = []
        aux_labels_list = [[] for _ in range(args.num_aux_tasks)]
        
        for item in batch:
            # Handle case with and without aux_labels
            if len(item) == 7:
                # No auxiliary tasks
                features_s, coords_s, features_l, coords_l, cox_duration, cox_event, main_task_label = item
                aux_labels = []
            else:
                # With auxiliary tasks
                features_s, coords_s, features_l, coords_l, cox_duration, cox_event, main_task_label, aux_labels = item
            
            features_s_list.append(features_s)
            coords_s_list.append(coords_s)
            features_l_list.append(features_l)
            coords_l_list.append(coords_l)
            
            if cox_duration is not None:
                cox_durations.append(cox_duration)
            else:
                cox_durations.append(torch.tensor(0.0))
            
            if cox_event is not None:
                cox_events.append(cox_event)
            else:
                cox_events.append(torch.tensor(0.0))
            
            if main_task_label is not None:
                main_task_labels.append(main_task_label)
            else:
                main_task_labels.append(None)
            
            # Only process auxiliary labels if there are auxiliary tasks
            if args.num_aux_tasks > 0:
                for task_idx, label in enumerate(aux_labels):
                    if task_idx < len(aux_labels_list):
                        if label is not None:
                            aux_labels_list[task_idx].append(label)
                        else:
                            aux_labels_list[task_idx].append(None)
        
        # Stack features (they should have same size, but handle variable sizes)
        # For now, return as list and handle in training loop
        cox_durations = torch.stack(cox_durations) if len(cox_durations) > 0 else None
        cox_events = torch.stack(cox_events) if len(cox_events) > 0 else None
        
        # Stack main task labels
        if len(main_task_labels) > 0:
            valid_main_labels = [l for l in main_task_labels if l is not None]
            if len(valid_main_labels) > 0:
                main_task_labels = torch.stack(valid_main_labels) if len(valid_main_labels) == len(batch) else None
            else:
                main_task_labels = None
        else:
            main_task_labels = None
        
        aux_labels_stacked = []
        for task_labels in aux_labels_list:
            valid_labels = [l for l in task_labels if l is not None]
            if len(valid_labels) > 0:
                aux_labels_stacked.append(torch.stack(valid_labels) if len(valid_labels) == len(batch) else None)
            else:
                aux_labels_stacked.append(None)
        
        return (features_s_list, coords_s_list, features_l_list, coords_l_list, 
                cox_durations, cox_events, main_task_labels, aux_labels_stacked)
    
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_split_mtl, batch_size=args.batch_size, shuffle=True, collate_fn=mtl_collate_fn)
    
    # 只有在有验证集时才创建 val_loader
    val_loader = None
    if val_split_mtl is not None:
        val_loader = DataLoader(val_split_mtl, batch_size=args.batch_size, shuffle=False, collate_fn=mtl_collate_fn)
        
    test_loader = DataLoader(test_split_mtl, batch_size=args.batch_size, shuffle=False, collate_fn=mtl_collate_fn)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        # Use user-specified mode if provided
        if hasattr(args, 'early_stopping_mode') and args.early_stopping_mode is not None:
            early_stopping_mode = args.early_stopping_mode
            print(f'Using user-specified early stopping mode: {early_stopping_mode}')
        else:
            # Determine early stopping mode based on metric type
            if args.early_stopping_metric in ['loss', 'brier_score']:
                early_stopping_mode = 'min'
            else:  # concordance_time_dependent, ipw_concordance_time_dependent
                early_stopping_mode = 'max'
            print(f'Automatically determined early stopping mode: {early_stopping_mode}')
        
        # Get patience and stop_epoch from args (with defaults)
        patience = getattr(args, 'early_stopping_patience', 20)
        stop_epoch = getattr(args, 'early_stopping_stop_epoch', 80)
        print(f'Patience: {patience}, Stop epoch: {stop_epoch}')
        
        early_stopping = EarlyStopping(patience=patience, stop_epoch=stop_epoch, verbose=True, mode=early_stopping_mode)
    else:
        early_stopping = None
    print('Done!')

    # Initialize loss records list if recording is enabled
    # loss_records = []
    # loss_csv_path = os.path.join(args.results_dir, f'loss_records_fold_{cur}.csv')

    # for epoch in range(args.max_epochs):
    #     # Get train loss record
    #     train_loss_record = train_loop_mtl(args, epoch, model, train_loader, optimizer, writer)
    #     if train_loss_record:
    #         loss_records.append(train_loss_record)
        
    #     # Get val loss record
    #     stop, val_loss_record = validate_mtl(cur, epoch, model, val_loader, args,
    #         early_stopping, writer, args.results_dir)
    #     if val_loss_record:
    #         loss_records.append(val_loss_record)
        
    #     # Save loss records every epoch if recording is enabled
    #     if hasattr(args, 'record_loss') and args.record_loss:
    #         save_loss_records(loss_records, loss_csv_path)
    #         # Clear the list to avoid duplicate entries
    #         loss_records = []
        
    #     if stop: 
    #         break
    # 初始化空列表来存储每个 epoch 的完整记录
    loss_records = []
    loss_csv_path = os.path.join(args.results_dir, f'loss_records_fold_{cur}.csv')

    for epoch in range(args.max_epochs):
        # 获取训练损失字典: e.g., {'total_loss': ..., 'cox_loss': ..., ...}
        train_loss_record = train_loop_mtl(args, epoch, model, train_loader, optimizer, writer)
        
        # 获取验证损失字典，并接收 early stopping 信号（如果有验证集）
        stop = False
        val_loss_record = None
        val_loss_for_scheduler = None
        if val_loader is not None:
            stop, val_loss_record = validate_mtl(cur, epoch, model, val_loader, args,
                                                early_stopping, writer, args.results_dir)
            # 获取验证损失用于scheduler（通常是total_loss_val）
            if val_loss_record and 'total_loss' in val_loss_record:
                val_loss_for_scheduler = val_loss_record['total_loss']

        # Update learning rate scheduler
        if scheduler_type == 'plateau':
            if val_loss_for_scheduler is not None:
                scheduler.step(val_loss_for_scheduler)
            else:
                # 如果没有验证集，使用训练损失
                if train_loss_record and 'total_loss' in train_loss_record:
                    scheduler.step(train_loss_record['total_loss'])
        else:  # step
            scheduler.step()

        # 构造合并的 record
        merged_record = {'epoch': epoch}

        if train_loss_record:
            for k, v in train_loss_record.items():
                merged_record[f'{k}_train'] = v

        if val_loss_record:
            for k, v in val_loss_record.items():
                merged_record[f'{k}_val'] = v

        # 添加到总记录
        loss_records.append(merged_record)

        # 【可选】每轮都保存一次（实时监控用）
        if hasattr(args, 'record_loss') and args.record_loss:
            # 转成 DataFrame 并保存
            df = pd.DataFrame(loss_records)
            df.to_csv(loss_csv_path, index=False)

        # Early stopping
        if stop:
            print(f"Early stopping at epoch {epoch}")
            break

    if args.early_stopping and val_loader is not None: 
        load_model_state(model, os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
    else:
        save_model_state(model, os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    # Save Gaussian model parameters (mu, sigma) before testing for MDN task
    if args.main_task == 'mdn' and hasattr(model, 'get_gaussian_params'):
        gaussian_params = model.get_gaussian_params()
        if gaussian_params['mu'] is not None and gaussian_params['sigma'] is not None:
            # Create DataFrame with mu and sigma
            gaussian_df = pd.DataFrame({
                'mu': gaussian_params['mu'],
                'sigma': gaussian_params['sigma']
            })
            # Save to CSV
            gaussian_csv_path = os.path.join(args.results_dir, f'gaussian_fold_{cur}.csv')
            gaussian_df.to_csv(gaussian_csv_path, index=False)
            print(f"Gaussian parameters (mu, sigma) saved to: {gaussian_csv_path}")

    # 只有在有验证集时才计算验证指标
    val_metrics = None
    if val_loader is not None:
        val_metrics = summary_mtl(args.mode, model, val_loader, args, dataset_type='val')
        print('Val metrics:', val_metrics[1])  # Only print metrics, not patient_results or mdn_survival_data

    # Pass fold index to summary_mtl for saving sample weights
    results_dict, test_metrics, mdn_survival_data = summary_mtl(args.mode, model, test_loader, args, dataset_type='test', fold_idx=cur)
    print('Test metrics:', test_metrics)  # test_metrics is already just the metrics dict

    # Save MDN survival data if available
    if mdn_survival_data is not None:
        import json
        mdn_survival_path = os.path.join(writer_dir, 'test_distribution_data.json')
        with open(mdn_survival_path, 'w') as f:
            json.dump(mdn_survival_data, f)
        print(f"MDN survival data saved to: {mdn_survival_path}")

    if writer:
        for metric_name, metric_value in test_metrics.items():
            writer.add_scalar(f'final/test_{metric_name}', metric_value, 0)
        writer.close()
    
    # Generate and save prediction results table for test set
    if results_dict:
        # Prepare data for DataFrame
        data = []
        for slide_id, info in results_dict.items():
            row = {'slide_id': slide_id}
            
            # Add main task results
            if 'main_task' in info and info['main_task']:
                main_task = info['main_task']
                if args.main_task == 'binary':
                    if 'label' in main_task:
                        row['main_task_label'] = main_task['label']
                    if 'pred' in main_task:
                        row['main_task_pred'] = main_task['pred']
                        # Mark if prediction is correct
                        if 'label' in main_task:
                            row['main_task_correct'] = (main_task['pred'] == main_task['label'])
                    if 'prob' in main_task:
                        row['main_task_prob'] = main_task['prob']
                elif args.main_task == 'cox':
                    if 'risk_score' in main_task:
                        row['main_task_risk_score'] = main_task['risk_score']
                    if 'duration' in main_task:
                        row['main_task_duration'] = main_task['duration']
                    if 'event' in main_task:
                        row['main_task_event'] = main_task['event']
            
            # Add auxiliary tasks results
            if 'aux_tasks' in info:
                for task_idx, task_info in enumerate(info['aux_tasks']):
                    if task_info:
                        if 'label' in task_info:
                            row[f'aux_task_{task_idx}_label'] = task_info['label']
                        if 'pred' in task_info:
                            row[f'aux_task_{task_idx}_pred'] = task_info['pred']
                            # Mark if prediction is correct
                            if 'label' in task_info:
                                row[f'aux_task_{task_idx}_correct'] = (task_info['pred'] == task_info['label'])
                        if 'prob' in task_info:
                            row[f'aux_task_{task_idx}_prob'] = task_info['prob']
                        if 'zero_shot' in task_info:
                            row[f'aux_task_{task_idx}_zero_shot'] = task_info['zero_shot']
            
            data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Save to CSV file
        results_dir = args.results_dir
        csv_path = os.path.join(results_dir, f'prediction_results_{cur}.csv')
        df.to_csv(csv_path, index=False)
        print(f'Prediction results saved to: {csv_path}')
    
    return results_dict, test_metrics, val_metrics


def train_loop_mtl(args, epoch, model, loader, optimizer, writer=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.train()
    train_loss = 0.
    main_task_loss_sum = 0.
    aux_loss_sum = [0.] * args.num_aux_tasks

    print('\n')
    # Initialize loss sums for low and high resolution
    main_task_loss_low_sum = 0.
    main_task_loss_high_sum = 0.
    aux_loss_low_sum = [0.] * args.num_aux_tasks
    aux_loss_high_sum = [0.] * args.num_aux_tasks
    contrastive_loss_sum = 0.  # 初始化移到循环外部
    consistency_loss_sum = 0.

    for batch_idx, (features_s_list, coords_s_list, features_l_list, coords_l_list, 
                    cox_durations, cox_events, main_task_labels, aux_labels_list) in enumerate(loader):
        
        # Forward pass with the entire batch
        outputs = model(features_s_list, coords_s_list, features_l_list, coords_l_list,
                       cox_durations=cox_durations,
                       cox_events=cox_events,
                       main_task_labels=main_task_labels,
                       aux_labels=aux_labels_list)
        
        # Get loss
        loss = outputs['total_loss']
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_value = loss.item()
        train_loss += loss_value
        
        if 'main_task' in outputs and 'loss' in outputs['main_task']:
            main_task_loss_sum += outputs['main_task']['loss'].item()
            # Collect low and high resolution losses
            if 'loss_low' in outputs['main_task']:
                main_task_loss_low_sum += outputs['main_task']['loss_low'].item()
            if 'loss_high' in outputs['main_task']:
                main_task_loss_high_sum += outputs['main_task']['loss_high'].item()
        
        for task_idx, task_output in enumerate(outputs['aux_tasks']):
            if 'loss' in task_output:
                aux_loss_sum[task_idx] += task_output['loss'].item()
                # Collect low and high resolution losses
                if 'loss_low' in task_output:
                    aux_loss_low_sum[task_idx] += task_output['loss_low'].item()
                if 'loss_high' in task_output:
                    aux_loss_high_sum[task_idx] += task_output['loss_high'].item()
        
        # Collect cross-scale contrastive loss
        if 'contrastive_loss' in outputs:
            cl_value = outputs['contrastive_loss'].item()
            contrastive_loss_sum += cl_value
            if batch_idx == 0:  # 只打印第一个batch用于调试
                print(f"[DEBUG] Batch {batch_idx}: contrastive_loss from outputs = {cl_value:.6f}")
        
        # Collect consistency loss
        if 'consistency_loss' in outputs:
            consistency_loss_sum += outputs['consistency_loss'].item()

    train_loss /= len(loader)
    main_task_loss_avg = main_task_loss_sum / len(loader) if main_task_loss_sum > 0 else 0.
    main_task_loss_low_avg = main_task_loss_low_sum / len(loader) if main_task_loss_low_sum > 0 else 0.
    main_task_loss_high_avg = main_task_loss_high_sum / len(loader) if main_task_loss_high_sum > 0 else 0.
    aux_loss_avg = [loss / len(loader) if loss > 0 else 0. for loss in aux_loss_sum]
    aux_loss_low_avg = [loss / len(loader) if loss > 0 else 0. for loss in aux_loss_low_sum]
    aux_loss_high_avg = [loss / len(loader) if loss > 0 else 0. for loss in aux_loss_high_sum]
    contrastive_loss_avg = contrastive_loss_sum / len(loader) if contrastive_loss_sum > 0 else 0.
    consistency_loss_avg = consistency_loss_sum / len(loader) if consistency_loss_sum > 0 else 0.

    print('Epoch: {}, train_loss: {:.4f}, main_task_loss: {:.4f} (low: {:.4f}, high: {:.4f})'.format(
        epoch, train_loss, main_task_loss_avg, main_task_loss_low_avg, main_task_loss_high_avg))
    if contrastive_loss_avg > 0:
        print('  contrastive_loss: {:.6f} (sum={:.6f}, batches={})'.format(
            contrastive_loss_avg, contrastive_loss_sum, len(loader)))
    if consistency_loss_avg > 0:
        print('  consistency_loss: {:.6f}'.format(consistency_loss_avg))
    for task_idx, aux_loss in enumerate(aux_loss_avg):
        if aux_loss > 0:
            print('  aux_task_{}_loss: {:.4f} (low: {:.4f}, high: {:.4f})'.format(
                task_idx, aux_loss, aux_loss_low_avg[task_idx], aux_loss_high_avg[task_idx]))

    if writer:
        writer.add_scalar('train/total_loss', train_loss, epoch)
        writer.add_scalar('train/main_task_loss', main_task_loss_avg, epoch)
        writer.add_scalar('train/main_task_loss_low', main_task_loss_low_avg, epoch)
        writer.add_scalar('train/main_task_loss_high', main_task_loss_high_avg, epoch)
        if contrastive_loss_avg > 0:
            writer.add_scalar('train/contrastive_loss', contrastive_loss_avg, epoch)
        if consistency_loss_avg > 0:
            writer.add_scalar('train/consistency_loss', consistency_loss_avg, epoch)
        for task_idx, aux_loss in enumerate(aux_loss_avg):
            if aux_loss > 0:
                writer.add_scalar(f'train/aux_task_{task_idx}_loss', aux_loss, epoch)
                writer.add_scalar(f'train/aux_task_{task_idx}_loss_low', aux_loss_low_avg[task_idx], epoch)
                writer.add_scalar(f'train/aux_task_{task_idx}_loss_high', aux_loss_high_avg[task_idx], epoch)

    # Record detailed loss information if enabled
    if hasattr(args, 'record_loss') and args.record_loss:
        loss_record = {
            'epoch': epoch,
            'phase': 'train',
            'total_loss': train_loss,
            'main_task_loss': main_task_loss_avg,
            'main_task_loss_low': main_task_loss_low_avg,
            'main_task_loss_high': main_task_loss_high_avg,
            'contrastive_loss': contrastive_loss_avg,
            'consistency_loss': consistency_loss_avg
        }
        for task_idx, aux_loss in enumerate(aux_loss_avg):
            loss_record[f'aux_task_{task_idx}_loss'] = aux_loss
            loss_record[f'aux_task_{task_idx}_loss_low'] = aux_loss_low_avg[task_idx]
            loss_record[f'aux_task_{task_idx}_loss_high'] = aux_loss_high_avg[task_idx]
        
        # Return loss record for further processing
        return loss_record
    
    return None


def validate_mtl(cur, epoch, model, loader, args, early_stopping=None, writer=None, results_dir=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss = 0.
    main_task_loss_sum = 0.
    main_task_loss_low_sum = 0.
    main_task_loss_high_sum = 0.
    aux_loss_sum = [0.] * args.num_aux_tasks
    aux_loss_low_sum = [0.] * args.num_aux_tasks
    aux_loss_high_sum = [0.] * args.num_aux_tasks

    # Metrics for early stopping
    mdn_metrics = {}
    cox_metrics = {}
    # 分类主任务: 收集标签/预测/概率, 用于逐 epoch 记录 AUC/BACC
    cls_labels, cls_preds, cls_probs = [], [], []
    if args.main_task == 'mdn':
        mdn_metrics["concordance_time_dependent"] = CIndexMeter()
        mdn_metrics["ipw_concordance_time_dependent"] = IPWCIndexMeter()
        mdn_metrics["brier_score"] = BrierScoreMeter()
    elif args.main_task == 'cox':
        # For Cox model, we collect risk scores, durations and events for C-index calculation
        cox_metrics["risk_scores"] = []
        cox_metrics["durations"] = []
        cox_metrics["events"] = []

    with torch.no_grad():
        for batch_idx, (features_s_list, coords_s_list, features_l_list, coords_l_list,
                        cox_durations, cox_events, main_task_labels, aux_labels_list) in enumerate(loader):
            
            # Forward pass with the entire batch
            outputs = model(features_s_list, coords_s_list, features_l_list, coords_l_list,
                           cox_durations=cox_durations,
                           cox_events=cox_events,
                           main_task_labels=main_task_labels,
                           aux_labels=aux_labels_list)
            
            # Get loss
            batch_loss = outputs['total_loss']
            val_loss += batch_loss.item()
            
            if 'main_task' in outputs and 'loss' in outputs['main_task']:
                main_task_loss_sum += outputs['main_task']['loss'].item()
                # Collect low and high resolution losses
                if 'loss_low' in outputs['main_task']:
                    main_task_loss_low_sum += outputs['main_task']['loss_low'].item()
                if 'loss_high' in outputs['main_task']:
                    main_task_loss_high_sum += outputs['main_task']['loss_high'].item()
            
            for task_idx, task_output in enumerate(outputs['aux_tasks']):
                if 'loss' in task_output:
                    aux_loss_sum[task_idx] += task_output['loss'].item()
                    # Collect low and high resolution losses
                    if 'loss_low' in task_output:
                        aux_loss_low_sum[task_idx] += task_output['loss_low'].item()
                    if 'loss_high' in task_output:
                        aux_loss_high_sum[task_idx] += task_output['loss_high'].item()

            # 分类主任务: 逐 batch 收集标签/预测/概率 (验证集无梯度)
            if args.main_task == 'binary' and 'main_task' in outputs:
                _mt = outputs['main_task']
                if 'labels' in _mt:
                    cls_labels.append(_mt['labels'].detach().cpu().numpy().reshape(-1))
                if 'pred' in _mt:
                    cls_preds.append(_mt['pred'].detach().cpu().numpy().reshape(-1))
                if 'prob' in _mt:
                    _p = _mt['prob'].detach().cpu().numpy()
                    cls_probs.append(_p if _p.ndim == 2 else _p[None, :])

            # Calculate MDN metrics for early stopping
            if args.main_task == 'mdn' and 'main_task' in outputs and 'eval_outputs' in outputs['main_task']:
                eval_outputs = outputs['main_task']['eval_outputs']
                
                # Get event indicator and time info - support batch processing
                if cox_events is not None:
                    # Handle both scalar and batch inputs
                    if cox_events.dim() == 0:
                        event_indicator = cox_events.item()
                    else:
                        event_indicator = cox_events.cpu().numpy()
                else:
                    event_indicator = 0
                
                if cox_durations is not None:
                    if cox_durations.dim() == 0:
                        t = cox_durations.item()
                    else:
                        t = cox_durations.cpu().numpy()
                else:
                    t = 0
                
                # Prepare data for survival metrics calculation
                metric_outputs = {
                    't': eval_outputs['t'],
                    'eval_t': eval_outputs['eval_t'],
                    'cum_hazard_seqs': eval_outputs['cum_hazard_seqs'],
                    'survival_seqs': eval_outputs['survival_seqs']
                }
                
                # Add survival sequences for different eps values
                for eps in [0.1, 0.2, 0.3, 0.4, 0.5]:
                    if f'survival_seqs_{eps}' in eval_outputs:
                        metric_outputs[f'survival_seqs_{eps}'] = eval_outputs[f'survival_seqs_{eps}']
                
                # Labels for metrics: cox_events is already the event indicator (1=event, 0=censored)
                # Support both scalar and batch inputs
                if isinstance(event_indicator, (int, float)):
                    labels = np.array([event_indicator])
                else:
                    labels = np.array(event_indicator)
                
                # Add data to all MDN metrics
                for metric_name in mdn_metrics:
                    mdn_metrics[metric_name].add(metric_outputs, labels)
            
            # Calculate Cox metrics for early stopping
            if args.main_task == 'cox' and 'main_task' in outputs and 'risk_scores' in outputs['main_task']:
                risk_scores = outputs['main_task']['risk_scores']
                
                # Get durations and events
                if cox_durations is not None and cox_events is not None:
                    # Handle both scalar and batch inputs
                    if risk_scores.dim() == 0:
                        risk_scores = risk_scores.unsqueeze(0)
                    
                    # Convert to numpy and collect
                    cox_metrics["risk_scores"].extend(risk_scores.cpu().numpy().flatten().tolist())
                    
                    if cox_durations.dim() == 0:
                        cox_metrics["durations"].append(cox_durations.item())
                    else:
                        cox_metrics["durations"].extend(cox_durations.cpu().numpy().flatten().tolist())
                    
                    if cox_events.dim() == 0:
                        cox_metrics["events"].append(cox_events.item())
                    else:
                        cox_metrics["events"].extend(cox_events.cpu().numpy().flatten().tolist())

    val_loss /= len(loader)
    main_task_loss_avg = main_task_loss_sum / len(loader) if main_task_loss_sum > 0 else 0.
    main_task_loss_low_avg = main_task_loss_low_sum / len(loader) if main_task_loss_low_sum > 0 else 0.
    main_task_loss_high_avg = main_task_loss_high_sum / len(loader) if main_task_loss_high_sum > 0 else 0.
    aux_loss_avg = [loss / len(loader) if loss > 0 else 0. for loss in aux_loss_sum]
    aux_loss_low_avg = [loss / len(loader) if loss > 0 else 0. for loss in aux_loss_low_sum]
    aux_loss_high_avg = [loss / len(loader) if loss > 0 else 0. for loss in aux_loss_high_sum]

    # Calculate early stopping metric
    early_stopping_value = val_loss
    if args.early_stopping_metric != 'loss':
        if args.main_task == 'mdn' and args.early_stopping_metric in mdn_metrics:
            early_stopping_value = mdn_metrics[args.early_stopping_metric].value()
            print(f'\nEarly stopping metric ({args.early_stopping_metric}): {early_stopping_value:.4f}')
        elif args.main_task == 'cox' and args.early_stopping_metric == 'c_index':
            # Calculate C-index for Cox model
            if len(cox_metrics["risk_scores"]) > 0:
                from lifelines.utils import concordance_index
                # Note: higher risk score = higher risk = shorter survival
                # So we use negative risk scores for concordance_index
                c_index = concordance_index(
                    event_times=cox_metrics["durations"],
                    predicted_scores=[-x for x in cox_metrics["risk_scores"]],
                    event_observed=cox_metrics["events"]
                )
                early_stopping_value = c_index
                print(f'\nEarly stopping metric (c_index): {early_stopping_value:.4f}')

    print('\nVal Set, val_loss: {:.4f}, main_task_loss: {:.4f} (low: {:.4f}, high: {:.4f})'.format(
        val_loss, main_task_loss_avg, main_task_loss_low_avg, main_task_loss_high_avg))
    for task_idx, aux_loss in enumerate(aux_loss_avg):
        if aux_loss > 0:
            print('  aux_task_{}_loss: {:.4f} (low: {:.4f}, high: {:.4f})'.format(
                task_idx, aux_loss, aux_loss_low_avg[task_idx], aux_loss_high_avg[task_idx]))

    # MDN specific metrics output
    if args.main_task == 'mdn':
        ctd = mdn_metrics.get("concordance_time_dependent", None)
        ipw_ctd = mdn_metrics.get("ipw_concordance_time_dependent", None)
        brier = mdn_metrics.get("brier_score", None)
        
        if ctd and ipw_ctd and brier:
            ctd_value = ctd.value()
            ipw_ctd_value = ipw_ctd.value()
            brier_value = brier.value()
            print(f'Val_Metrics: CTD:{ctd_value:.4f}, IPWCTD:{ipw_ctd_value:.4f}, Brier:{brier_value:.4f}')
    
    # Cox specific metrics output
    if args.main_task == 'cox':
        if len(cox_metrics["risk_scores"]) > 0:
            from lifelines.utils import concordance_index
            c_index = concordance_index(
                event_times=cox_metrics["durations"],
                predicted_scores=[-x for x in cox_metrics["risk_scores"]],
                event_observed=cox_metrics["events"]
            )
            print(f'Val_Metrics: C-index:{c_index:.4f}')

    if writer:
        writer.add_scalar('val/total_loss', val_loss, epoch)
        writer.add_scalar('val/main_task_loss', main_task_loss_avg, epoch)
        writer.add_scalar('val/main_task_loss_low', main_task_loss_low_avg, epoch)
        writer.add_scalar('val/main_task_loss_high', main_task_loss_high_avg, epoch)
        for task_idx, aux_loss in enumerate(aux_loss_avg):
            if aux_loss > 0:
                writer.add_scalar(f'val/aux_task_{task_idx}_loss', aux_loss, epoch)
                writer.add_scalar(f'val/aux_task_{task_idx}_loss_low', aux_loss_low_avg[task_idx], epoch)
                writer.add_scalar(f'val/aux_task_{task_idx}_loss_high', aux_loss_high_avg[task_idx], epoch)
        
        # Log early stopping metric
        if args.early_stopping_metric != 'loss':
            if args.main_task == 'mdn':
                writer.add_scalar(f'val/{args.early_stopping_metric}', early_stopping_value, epoch)
            elif args.main_task == 'cox':
                writer.add_scalar(f'val/{args.early_stopping_metric}', early_stopping_value, epoch)

    # Record detailed loss information if enabled
    loss_record = None
    if hasattr(args, 'record_loss') and args.record_loss:
        loss_record = {
            'epoch': epoch,
            'phase': 'val',
            'total_loss': val_loss,
            'main_task_loss': main_task_loss_avg,
            'main_task_loss_low': main_task_loss_low_avg,
            'main_task_loss_high': main_task_loss_high_avg
        }
        for task_idx, aux_loss in enumerate(aux_loss_avg):
            loss_record[f'aux_task_{task_idx}_loss'] = aux_loss
            loss_record[f'aux_task_{task_idx}_loss_low'] = aux_loss_low_avg[task_idx]
            loss_record[f'aux_task_{task_idx}_loss_high'] = aux_loss_high_avg[task_idx]

        # 分类主任务: 逐 epoch 记录 AUC / BACC (写入 loss_records CSV, 供收敛性分析)
        if args.main_task == 'binary' and cls_labels:
            y_true = np.concatenate(cls_labels)
            if len(np.unique(y_true)) > 1:
                if cls_probs:
                    y_probs = np.concatenate(cls_probs, axis=0)
                    try:
                        if y_probs.ndim == 2 and y_probs.shape[1] == 2:
                            loss_record['main_task_auc'] = float(roc_auc_score(y_true, y_probs[:, 1]))
                        elif y_probs.ndim == 2:
                            loss_record['main_task_auc'] = float(roc_auc_score(y_true, y_probs, multi_class='ovr'))
                    except Exception as e:
                        print(f'Warning: AUC compute failed at epoch {epoch}: {e}')
                if cls_preds:
                    y_pred = np.concatenate(cls_preds)
                    if len(y_pred) == len(y_true):
                        try:
                            loss_record['main_task_bacc'] = float(balanced_accuracy_score(y_true, y_pred))
                        except Exception as e:
                            print(f'Warning: BACC compute failed at epoch {epoch}: {e}')

        # Add evaluation metrics to loss record
        # For Cox model, always calculate and record c-index if we have risk scores
        if args.main_task == 'cox':
            if len(cox_metrics["risk_scores"]) > 0:
                from lifelines.utils import concordance_index
                c_index = concordance_index(
                    event_times=cox_metrics["durations"],
                    predicted_scores=[-x for x in cox_metrics["risk_scores"]],
                    event_observed=cox_metrics["events"]
                )
                loss_record['c_index'] = c_index
                # Also update early_stopping_value if needed for consistency
                if args.early_stopping_metric == 'c_index':
                    early_stopping_value = c_index
        
        # Add early stopping metric to loss record if it's different from loss
        if args.early_stopping_metric != 'loss':
            if args.main_task == 'mdn':
                loss_record[args.early_stopping_metric] = early_stopping_value
            elif args.main_task == 'cox':
                loss_record[args.early_stopping_metric] = early_stopping_value

    if early_stopping:
        assert results_dir
        early_stopping(epoch, early_stopping_value, model, ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True, loss_record

    return False, loss_record


def summary_mtl(mode, model, loader, args, dataset_type='test', fold_idx=None):
    """
    Calculate metrics for multi-task learning
    
    Args:
        mode: Model mode (e.g., 'transformer')
        model: The trained model
        loader: Data loader for the dataset
        args: Command line arguments
        dataset_type: Type of dataset ('train', 'val', or 'test')
        fold_idx: Fold index for saving Gaussian weights
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    # Main task variables
    all_risk_scores = []
    all_cox_durations = []
    all_cox_events = []
    all_main_task_predictions = []
    all_main_task_labels = []
    all_main_task_probs = []
    
    # MDN survival metrics
    mdn_metrics = {}
    if args.main_task == 'mdn':
        mdn_metrics["concordance_time_dependent"] = CIndexMeter()
        mdn_metrics["ipw_concordance_time_dependent"] = IPWCIndexMeter()
        mdn_metrics["ipw_2_concordance_time_dependent"] = IPWCIndexMeter(eps=0.2)
        mdn_metrics["ipw_4_concordance_time_dependent"] = IPWCIndexMeter(eps=0.4)
        mdn_metrics["brier_score"] = BrierScoreMeter()
        mdn_metrics["brier_score_2"] = BrierScoreMeter(eps=0.2)
        mdn_metrics["brier_score_4"] = BrierScoreMeter(eps=0.4)
    
    # Auxiliary task variables
    all_aux_predictions = [[] for _ in range(args.num_aux_tasks)]
    all_aux_labels = [[] for _ in range(args.num_aux_tasks)]
    all_aux_probs = [[] for _ in range(args.num_aux_tasks)]
    # Zero-shot specific variables
    all_aux_predictions_zero_shot = [[] for _ in range(args.num_aux_tasks)]
    all_aux_labels_zero_shot = [[] for _ in range(args.num_aux_tasks)]
    all_aux_probs_zero_shot = [[] for _ in range(args.num_aux_tasks)]
    all_aux_predictions_non_zero_shot = [[] for _ in range(args.num_aux_tasks)]
    all_aux_labels_non_zero_shot = [[] for _ in range(args.num_aux_tasks)]
    all_aux_probs_non_zero_shot = [[] for _ in range(args.num_aux_tasks)]
    
    patient_results = {}

    # Get slide IDs for each sample
    slide_ids = loader.dataset.slide_data['slide_id'].tolist()
    
    # MDN survival prediction storage
    mdn_survival_data = {}
    
    # MDN Gaussian weights storage (for each sample)
    mdn_sample_weights = {}
    if args.main_task == 'mdn':
        # Create evaluation time points for survival prediction
        eval_times = torch.arange(0.1, 100, 0.1)  # Same as SCMIL's default
    
    with torch.no_grad():
        processed_count = 0
        for batch_idx, (features_s_list, coords_s_list, features_l_list, coords_l_list,
                        cox_durations, cox_events, main_task_labels, aux_labels_list) in enumerate(loader):
            
            # Get slide IDs for current batch (support batch processing)
            batch_size = len(features_s_list) if isinstance(features_s_list, list) else features_s_list.shape[0]
            start_idx = processed_count
            end_idx = min(start_idx + batch_size, len(slide_ids))
            batch_slide_ids = slide_ids[start_idx:end_idx]
            
            processed_count += batch_size
            
            # Forward pass with the entire batch
            outputs = model(features_s_list, coords_s_list, features_l_list, coords_l_list,
                           cox_durations=cox_durations,
                           cox_events=cox_events,
                           main_task_labels=main_task_labels,
                           aux_labels=aux_labels_list)
            
            # For MDN, compute survival probability curves
            if args.main_task == 'mdn' and 'main_task' in outputs and 'eval_outputs' in outputs['main_task']:
                eval_outputs = outputs['main_task']['eval_outputs']
                
                # Collect Gaussian weights for each sample
                if 'params' in outputs['main_task']:
                    params = outputs['main_task']['params']
                    if 'w' in params:
                        weights = params['w'].cpu().numpy()
                        # Handle single sample and batch cases
                        if weights.ndim == 1:
                            # Single sample
                            mdn_sample_weights[batch_slide_ids[0]] = weights
                        else:
                            # Batch case
                            batch_size = weights.shape[0]
                            for i in range(batch_size):
                                mdn_sample_weights[batch_slide_ids[i]] = weights[i]
                
                # Get event indicator and time info - support batch processing
                if cox_events is not None:
                    if cox_events.dim() == 0:
                        event_indicator = cox_events.item()
                    else:
                        event_indicator = cox_events.cpu().numpy()
                else:
                    event_indicator = 0
                
                if cox_durations is not None:
                    if cox_durations.dim() == 0:
                        t = cox_durations.item()
                    else:
                        t = cox_durations.cpu().numpy()
                else:
                    t = 0
                
                # Handle batch processing for survival curves
                if isinstance(event_indicator, (int, float)):
                    # Single sample case
                    # survival_seqs shape is [eval_len, 1] from MDN model
                    survival_func = eval_outputs['survival_seqs'].squeeze(1)  # [eval_len]
                    eval_times_np = eval_outputs['eval_t'].cpu().numpy()
                    survival_func_np = survival_func.cpu().numpy()
                    
                    mdn_survival_data[batch_slide_ids[0]] = {
                        'censorship': int(1 - event_indicator),  # Convert to censorship for storage
                        't': float(t),
                        'x': eval_times_np.tolist(),
                        'y': survival_func_np.tolist()
                    }
                else:
                    # Batch case - store each sample's data
                    batch_size = len(event_indicator)
                    # survival_seqs shape is [eval_len, batch_size] from MDN model
                    survival_func = eval_outputs['survival_seqs']  # [eval_len, batch_size]
                    eval_t = eval_outputs['eval_t']  # [eval_len]
                    
                    for i in range(batch_size):
                        # Use actual slide ID from batch_slide_ids
                        sample_slide_id = batch_slide_ids[i]
                        # Get survival curve for sample i: transpose to get [batch_size, eval_len]
                        sample_survival = survival_func[:, i].cpu().numpy()
                        eval_times_np = eval_t.cpu().numpy()
                        
                        mdn_survival_data[sample_slide_id] = {
                            'censorship': int(1 - event_indicator[i]),  # Convert to censorship for storage
                            't': float(t[i]),
                            'x': eval_times_np.tolist(),
                            'y': sample_survival.tolist()
                        }
                
                # Prepare data for survival metrics calculation
                # Use eval_outputs directly for metrics
                # Keep as torch tensors for to_numpy conversion in metrics
                metric_outputs = {
                    't': eval_outputs['t'],
                    'eval_t': eval_outputs['eval_t'],
                    'cum_hazard_seqs': eval_outputs['cum_hazard_seqs'],
                    'survival_seqs': eval_outputs['survival_seqs']
                }
                
                # Add survival sequences for different eps values
                for eps in [0.1, 0.2, 0.3, 0.4, 0.5]:
                    if f'survival_seqs_{eps}' in eval_outputs:
                        metric_outputs[f'survival_seqs_{eps}'] = eval_outputs[f'survival_seqs_{eps}']
                
                # Labels for metrics: cox_events is already the event indicator (1=event, 0=censored)
                # Handle both scalar and array cases
                if isinstance(event_indicator, (int, float)):
                    labels = np.array([event_indicator])
                else:
                    labels = np.array(event_indicator)  # Ensure 1D array for batch case
                
                # Add data to all MDN metrics
                for metric_name in mdn_metrics:
                    mdn_metrics[metric_name].add(metric_outputs, labels)
            
            # Collect main task results
            main_task_info = {}
            if 'main_task' in outputs:
                if args.main_task == 'cox' and 'risk_scores' in outputs['main_task']:
                    risk_scores = outputs['main_task']['risk_scores'].cpu().numpy()
                    # Handle different types of risk_scores
                    try:
                        # First try to extend (if it's iterable)
                        all_risk_scores.extend(risk_scores.tolist())
                        main_task_info['risk_score'] = risk_scores.tolist()
                    except (AttributeError, TypeError):
                        # If it's not iterable or doesn't have tolist(), append as a scalar
                        all_risk_scores.append(risk_scores)
                        main_task_info['risk_score'] = float(risk_scores)
                    
                    if cox_durations is not None:
                        try:
                            all_cox_durations.extend(cox_durations.cpu().numpy().tolist())
                            main_task_info['duration'] = cox_durations.cpu().numpy().tolist()
                        except (AttributeError, TypeError):
                            all_cox_durations.append(cox_durations)
                            main_task_info['duration'] = float(cox_durations)
                    if cox_events is not None:
                        try:
                            all_cox_events.extend(cox_events.cpu().numpy().tolist())
                            main_task_info['event'] = cox_events.cpu().numpy().tolist()
                        except (AttributeError, TypeError):
                            all_cox_events.append(cox_events)
                            main_task_info['event'] = float(cox_events)
                elif args.main_task == 'binary' and 'pred' in outputs['main_task'] and 'labels' in outputs['main_task']:
                    predictions = outputs['main_task']['pred'].cpu().numpy()
                    labels = outputs['main_task']['labels'].cpu().numpy()
                    
                    # Handle predictions
                    try:
                        all_main_task_predictions.extend(predictions.tolist())
                        main_task_info['pred'] = predictions.tolist()
                    except (AttributeError, TypeError):
                        all_main_task_predictions.append(predictions)
                        main_task_info['pred'] = int(predictions)
                    
                    # Handle labels
                    try:
                        all_main_task_labels.extend(labels.tolist())
                        main_task_info['label'] = labels.tolist()
                    except (AttributeError, TypeError):
                        all_main_task_labels.append(labels)
                        main_task_info['label'] = int(labels)
                    
                    if 'prob' in outputs['main_task']:
                        probs = outputs['main_task']['prob'].cpu().numpy()
                        # Handle probs
                        try:
                            all_main_task_probs.extend(probs.tolist())
                            main_task_info['prob'] = probs.tolist()
                        except (AttributeError, TypeError):
                            all_main_task_probs.append(probs)
                            main_task_info['prob'] = probs.tolist()
            
            # Collect auxiliary task results
            aux_tasks_info = []
            for task_idx, task_output in enumerate(outputs['aux_tasks']):
                task_info = {}
                if 'pred' in task_output and 'labels' in task_output:
                    predictions = task_output['pred'].cpu().numpy()
                    labels = task_output['labels'].cpu().numpy()
                    
                    # Get zero-shot mask if available
                    zero_shot_mask = task_output.get('zero_shot_mask', None)
                    if zero_shot_mask is not None:
                        zero_shot_mask_np = zero_shot_mask.cpu().numpy()
                    else:
                        zero_shot_mask_np = np.zeros_like(labels, dtype=bool)
                    
                    # Handle predictions
                    try:
                        all_aux_predictions[task_idx].extend(predictions.tolist())
                        task_info['pred'] = predictions.tolist()
                    except (AttributeError, TypeError):
                        all_aux_predictions[task_idx].append(predictions)
                        task_info['pred'] = int(predictions)
                    
                    # Handle labels
                    try:
                        all_aux_labels[task_idx].extend(labels.tolist())
                        task_info['label'] = labels.tolist()
                    except (AttributeError, TypeError):
                        all_aux_labels[task_idx].append(labels)
                        task_info['label'] = int(labels)
                    
                    if 'prob' in task_output:
                        probs = task_output['prob'].cpu().numpy()
                        # Handle probs
                        try:
                            all_aux_probs[task_idx].extend(probs.tolist())
                            task_info['prob'] = probs.tolist()
                        except (AttributeError, TypeError):
                            all_aux_probs[task_idx].append(probs)
                            task_info['prob'] = probs.tolist()
                    
                    # Add zero-shot mask
                    task_info['zero_shot'] = bool(zero_shot_mask_np) if zero_shot_mask_np.ndim == 0 else zero_shot_mask_np.tolist()
                    
                    # Separate zero-shot and non-zero-shot samples
                    if zero_shot_mask is not None:
                        # Zero-shot samples
                        zero_shot_indices = np.where(zero_shot_mask_np)[0]
                        if len(zero_shot_indices) > 0:
                            if predictions.ndim == 0:
                                all_aux_predictions_zero_shot[task_idx].append(predictions.item())
                                all_aux_labels_zero_shot[task_idx].append(labels.item())
                            else:
                                all_aux_predictions_zero_shot[task_idx].extend(predictions[zero_shot_indices].tolist())
                                all_aux_labels_zero_shot[task_idx].extend(labels[zero_shot_indices].tolist())
                            
                            if 'prob' in task_output:
                                if probs.ndim == 1:
                                    all_aux_probs_zero_shot[task_idx].append(probs.item())
                                else:
                                    all_aux_probs_zero_shot[task_idx].extend(probs[zero_shot_indices].tolist())
                        
                        # Non-zero-shot samples
                        non_zero_shot_indices = np.where(~zero_shot_mask_np)[0]
                        if len(non_zero_shot_indices) > 0:
                            if predictions.ndim == 0:
                                all_aux_predictions_non_zero_shot[task_idx].append(predictions.item())
                                all_aux_labels_non_zero_shot[task_idx].append(labels.item())
                            else:
                                all_aux_predictions_non_zero_shot[task_idx].extend(predictions[non_zero_shot_indices].tolist())
                                all_aux_labels_non_zero_shot[task_idx].extend(labels[non_zero_shot_indices].tolist())
                            
                            if 'prob' in task_output:
                                if probs.ndim == 1:
                                    all_aux_probs_non_zero_shot[task_idx].append(probs.item())
                                else:
                                    all_aux_probs_non_zero_shot[task_idx].extend(probs[non_zero_shot_indices].tolist())
                aux_tasks_info.append(task_info)
            
            # Update patient_results with current sample information
            # Handle both single sample and batch cases
            if len(batch_slide_ids) == 1:
                # Single sample case
                patient_results[batch_slide_ids[0]] = {
                    'slide_id': batch_slide_ids[0],
                    'main_task': main_task_info,
                    'aux_tasks': aux_tasks_info
                }
            else:
                # Batch case - handle each sample individually
                batch_size = len(batch_slide_ids)
                for sample_idx in range(batch_size):
                    sample_slide_id = batch_slide_ids[sample_idx]
                    
                    # Extract sample-specific main task info
                    sample_main_task_info = {}
                    for key, value in main_task_info.items():
                        if isinstance(value, list) and len(value) == batch_size:
                            sample_main_task_info[key] = value[sample_idx]
                        else:
                            sample_main_task_info[key] = value
                    
                    # Extract sample-specific aux tasks info
                    sample_aux_tasks_info = []
                    for task_info in aux_tasks_info:
                        sample_task_info = {}
                        for key, value in task_info.items():
                            if isinstance(value, list) and len(value) == batch_size:
                                sample_task_info[key] = value[sample_idx]
                            else:
                                sample_task_info[key] = value
                        sample_aux_tasks_info.append(sample_task_info)
                    
                    patient_results[sample_slide_id] = {
                        'slide_id': sample_slide_id,
                        'main_task': sample_main_task_info,
                        'aux_tasks': sample_aux_tasks_info
                    }
    
    # Calculate metrics
    metrics = {}
    
    # Main task metrics
    if args.main_task == 'cox' and len(all_risk_scores) > 0 and len(all_cox_events) > 0:
        try:
            from lifelines.utils import concordance_index
            
            # Use lifelines library for consistent C-index calculation
            # Same implementation as in validate_mtl function
            c_index = concordance_index(
                event_times=np.array(all_cox_durations),
                predicted_scores=[-x for x in all_risk_scores],  # Negative risk scores for lifelines convention
                event_observed=np.array(all_cox_events)
            )
            metrics['main_task_c_index'] = c_index
        except Exception as e:
            print(f"Error calculating C-index: {e}")
            metrics['main_task_c_index'] = 0.0
    elif args.main_task == 'binary' and len(all_main_task_predictions) > 0:
        from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, balanced_accuracy_score
        y_true = np.array(all_main_task_labels)
        y_pred = np.array(all_main_task_predictions)
        
        acc = accuracy_score(y_true, y_pred)
        metrics['main_task_acc'] = acc
        
        bacc = balanced_accuracy_score(y_true, y_pred)
        metrics['main_task_bacc'] = bacc
        
        if len(all_main_task_probs) > 0:
            y_probs = np.array(all_main_task_probs)
            if len(np.unique(y_true)) > 1:  # Check if binary
                if y_probs.shape[1] == 2:  # Binary
                    auc = roc_auc_score(y_true, y_probs[:, 1])
                else:  # Multi-class
                    auc = roc_auc_score(y_true, y_probs, multi_class='ovr')
                metrics['main_task_auc'] = auc
            
            f1 = f1_score(y_true, y_pred, average='macro')
            metrics['main_task_f1'] = f1
    elif args.main_task == 'mdn':
        # Calculate MDN survival metrics
        try:
            for metric_name in mdn_metrics:
                metric_value = mdn_metrics[metric_name].value()
                metrics[f'main_task_{metric_name}'] = metric_value
                print(f'  MDN {metric_name}: {metric_value:.4f}')
            
            # Format metrics output
            ctd = mdn_metrics.get("concordance_time_dependent", None)
            ipw_ctd = mdn_metrics.get("ipw_concordance_time_dependent", None)
            brier = mdn_metrics.get("brier_score", None)
            
            if ctd and ipw_ctd and brier:
                ctd_value = ctd.value()
                ipw_ctd_value = ipw_ctd.value()
                brier_value = brier.value()
                print(f'{dataset_type.capitalize()}_Metrics: CTD:{ctd_value:.4f}, IPWCTD:{ipw_ctd_value:.4f}, Brier:{brier_value:.4f}')
        except Exception as e:
            print(f"Error calculating MDN survival metrics: {e}")
            # Set default values if calculation fails
            metrics['main_task_concordance_time_dependent'] = 0.0
            metrics['main_task_ipw_concordance_time_dependent'] = 0.0
            metrics['main_task_ipw_2_concordance_time_dependent'] = 0.0
            metrics['main_task_ipw_4_concordance_time_dependent'] = 0.0
            metrics['main_task_brier_score'] = 1.0
            metrics['main_task_brier_score_2'] = 1.0
            metrics['main_task_brier_score_4'] = 1.0
    # else: main_task == 'none', no main task metrics to calculate
    
    # Auxiliary task metrics
    for task_idx in range(args.num_aux_tasks):
        # Check if this task has zero-shot classes
        zero_shot_classes = args.zero_shot_config.get(task_idx, [])
        
        # Overall metrics (including both zero-shot and non-zero-shot)
        if len(all_aux_predictions[task_idx]) > 0:
            from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
            y_true = np.array(all_aux_labels[task_idx])
            y_pred = np.array(all_aux_predictions[task_idx])
            
            acc = accuracy_score(y_true, y_pred)
            metrics[f'aux_task_{task_idx}_acc'] = acc
            
            if len(all_aux_probs[task_idx]) > 0:
                y_probs = np.array(all_aux_probs[task_idx])
                unique_classes = np.unique(y_true)
                num_classes = len(unique_classes)

                # if dataset_type in ['train', 'val'] and num_classes != y_probs.shape[1]:
                if dataset_type in ['train', 'val', 'test'] and num_classes != y_probs.shape[1]:
                    print(f"警告: 类别数不匹配! y_true有{num_classes}类, y_probs有{y_probs.shape[1]}列")
                    # 处理zero-shot任务的情况：y_true的类别数少于y_probs的列数
                    # 只选择y_true中存在的类别的概率
                    unique_classes = np.unique(y_true)
                    if len(unique_classes) == num_classes and num_classes < y_probs.shape[1]:
                        # 只保留y_true中存在的类别的概率
                        y_probs = y_probs[:, unique_classes]
                        # 重新归一化概率，确保总和为1
                        y_probs = y_probs / y_probs.sum(axis=1, keepdims=True)
                        print(f"已调整y_probs: 只保留y_true中存在的{num_classes}个类别的概率并重新归一化")
                if len(np.unique(y_true)) > 1:  # Check if binary or multi-class
                    if y_probs.shape[1] == 2:  # Binary
                        auc = roc_auc_score(y_true, y_probs[:, 1])
                    else:  # Multi-class
                        # 确保y_true中的类别索引在y_probs的列范围内
                        unique_y_true = np.unique(y_true)
                        if np.all(unique_y_true < y_probs.shape[1]):
                            auc = roc_auc_score(y_true, y_probs, multi_class='ovr')
                        else:
                            print(f"警告: y_true中的类别索引超出y_probs的列范围，跳过AUC计算")
                            auc = 0.0
                    metrics[f'aux_task_{task_idx}_auc'] = auc
                
                f1 = f1_score(y_true, y_pred, average='macro')
                metrics[f'aux_task_{task_idx}_f1'] = f1
        
        # Zero-shot specific metrics (only for zero-shot classes)
        if zero_shot_classes and len(all_aux_predictions_zero_shot[task_idx]) > 0:
            from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
            y_true_zs = np.array(all_aux_labels_zero_shot[task_idx])
            y_pred_zs = np.array(all_aux_predictions_zero_shot[task_idx])
            
            acc_zs = accuracy_score(y_true_zs, y_pred_zs)
            metrics[f'aux_task_{task_idx}_zero_shot_acc'] = acc_zs
            
            if len(all_aux_probs_zero_shot[task_idx]) > 0:
                y_probs_zs = np.array(all_aux_probs_zero_shot[task_idx])
                unique_classes_zs = np.unique(y_true_zs)
                num_classes_zs = len(unique_classes_zs)

                if len(np.unique(y_true_zs)) > 1:  # Check if binary or multi-class
                    # 处理类别数不匹配的情况
                    if dataset_type in ['train', 'val', 'test'] and y_probs_zs.shape[1] != num_classes_zs:
                    # if dataset_type in ['train', 'val'] and y_probs_zs.shape[1] != num_classes_zs:
                        print(f"警告: Zero-shot类别数不匹配! y_true有{num_classes_zs}类, y_probs有{y_probs_zs.shape[1]}列")
                        # 只选择y_true中存在的类别的概率
                        if num_classes_zs < y_probs_zs.shape[1]:
                            y_probs_zs = y_probs_zs[:, unique_classes_zs]
                            # 重新归一化概率，确保总和为1
                            y_probs_zs = y_probs_zs / y_probs_zs.sum(axis=1, keepdims=True)
                            print(f"已调整zero-shot y_probs: 只保留y_true中存在的{num_classes_zs}个类别的概率并重新归一化")
                    
                    if y_probs_zs.shape[1] == 2:  # Binary
                        auc_zs = roc_auc_score(y_true_zs, y_probs_zs[:, 1])
                    else:  # Multi-class
                        # 确保y_true中的类别索引在y_probs的列范围内
                        unique_y_true_zs = np.unique(y_true_zs)
                        if np.all(unique_y_true_zs < y_probs_zs.shape[1]):
                            auc_zs = roc_auc_score(y_true_zs, y_probs_zs, multi_class='ovr')
                        else:
                            print(f"警告: Zero-shot y_true中的类别索引超出y_probs的列范围，跳过AUC计算")
                            auc_zs = 0.0
                    metrics[f'aux_task_{task_idx}_zero_shot_auc'] = auc_zs
                
                f1_zs = f1_score(y_true_zs, y_pred_zs, average='macro')
                metrics[f'aux_task_{task_idx}_zero_shot_f1'] = f1_zs
            
            print(f'  Task {task_idx} Zero-shot metrics: acc={acc_zs:.4f}, f1={f1_zs:.4f}')
        
        # Non-zero-shot specific metrics (only for non-zero-shot classes)
        if zero_shot_classes and len(all_aux_predictions_non_zero_shot[task_idx]) > 0:
            from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
            y_true_nzs = np.array(all_aux_labels_non_zero_shot[task_idx])
            y_pred_nzs = np.array(all_aux_predictions_non_zero_shot[task_idx])
            
            acc_nzs = accuracy_score(y_true_nzs, y_pred_nzs)
            metrics[f'aux_task_{task_idx}_non_zero_shot_acc'] = acc_nzs
            
            if len(all_aux_probs_non_zero_shot[task_idx]) > 0:
                y_probs_nzs = np.array(all_aux_probs_non_zero_shot[task_idx])
                unique_classes_nzs = np.unique(y_true_nzs)
                num_classes_nzs = len(unique_classes_nzs)

                if len(np.unique(y_true_nzs)) > 1:  # Check if binary or multi-class
                    # 处理类别数不匹配的情况 - 对于non-zero-shot样本，无论数据集类型如何都需要处理
                    if y_probs_nzs.shape[1] != num_classes_nzs:
                        print(f"警告: Non-zero-shot类别数不匹配! y_true有{num_classes_nzs}类, y_probs有{y_probs_nzs.shape[1]}列")
                        # 只选择y_true中存在的类别的概率
                        if num_classes_nzs < y_probs_nzs.shape[1]:
                            y_probs_nzs = y_probs_nzs[:, unique_classes_nzs]
                            # 重新归一化概率，确保总和为1
                            y_probs_nzs = y_probs_nzs / y_probs_nzs.sum(axis=1, keepdims=True)
                            print(f"已调整non-zero-shot y_probs: 只保留y_true中存在的{num_classes_nzs}个类别的概率并重新归一化")
                    
                    if y_probs_nzs.shape[1] == 2:  # Binary
                        auc_nzs = roc_auc_score(y_true_nzs, y_probs_nzs[:, 1])
                    else:  # Multi-class
                        # 确保y_true中的类别索引在y_probs的列范围内
                        unique_y_true_nzs = np.unique(y_true_nzs)
                        if np.all(unique_y_true_nzs < y_probs_nzs.shape[1]):
                            auc_nzs = roc_auc_score(y_true_nzs, y_probs_nzs, multi_class='ovr')
                        else:
                            print(f"警告: Non-zero-shot y_true中的类别索引超出y_probs的列范围，跳过AUC计算")
                            auc_nzs = 0.0
                    metrics[f'aux_task_{task_idx}_non_zero_shot_auc'] = auc_nzs
                
                f1_nzs = f1_score(y_true_nzs, y_pred_nzs, average='macro')
                metrics[f'aux_task_{task_idx}_non_zero_shot_f1'] = f1_nzs
            
            print(f'  Task {task_idx} Non-zero-shot metrics: acc={acc_nzs:.4f}, f1={f1_nzs:.4f}')
    
    # Save MDN sample weights to the same CSV file as Gaussian parameters
    if args.main_task == 'mdn' and dataset_type == 'test' and fold_idx is not None and len(mdn_sample_weights) > 0:
        import pandas as pd
        gaussian_csv_path = os.path.join(args.results_dir, f'gaussian_fold_{fold_idx}.csv')
        
        if os.path.exists(gaussian_csv_path):
            # Read existing Gaussian parameters
            gaussian_df = pd.read_csv(gaussian_csv_path)
            
            # Add sample weights as new columns
            for slide_id, weights in mdn_sample_weights.items():
                gaussian_df[slide_id] = weights
            
            # Save updated CSV
            gaussian_df.to_csv(gaussian_csv_path, index=False)
            print(f"Gaussian parameters with sample weights saved to: {gaussian_csv_path}")
    
    return patient_results, metrics, mdn_survival_data if args.main_task == 'mdn' else None

