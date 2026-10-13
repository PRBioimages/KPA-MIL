"""
ViLa-MIL Attention Visualization Script (No model modification needed)

Uses PyTorch hooks to extract attention weights without modifying model code.

Usage:
    python vis_attention.py \
        --ckpt_path tj0.0002,0.0001/tj_conch_aux0_aux_label_1_aux_label_2_p2_crosca0.8_step_gpt_nopre_s0/s_0_checkpoint.pt \
        --h5_dir /home/hzxu/ViLa-MIL-main/feature_dir/tj/10feattj/h5_files/ \
        --h5_dir_l /home/hzxu/ViLa-MIL-main/feature_dir/tj/20feattj/h5_files/ \
        --wsi_dir /home/hzxu/MIL-Liver/TJ_datasets/mrxs/ \
        --output_dir attention_results_fold0 \
        --slide_ids A2024004

Output files:
    {slide_name}_wsi.png          - WSI thumbnail (no attention)
    {slide_name}_heatmap_low.png  - Low-res attention heatmap (transparent background)
    {slide_name}_heatmap_high.png - High-res attention heatmap (transparent background)
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import h5py
from PIL import Image
from types import SimpleNamespace
from functools import partial

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


class AttentionExtractor:
    """Extract attention weights using hooks."""
    
    def __init__(self, model):
        self.model = model
        self.attention_weights = {}
        self.hooks = []
        self.call_count = {}
        
    def register_hooks(self):
        mil_aggregator = getattr(self.model, 'mil_aggregator', None)
        if mil_aggregator is not None:
            cross_attn_1 = getattr(mil_aggregator, 'cross_attention_1', None)
            if cross_attn_1 is not None:
                self.call_count['cross_attn_1'] = 0
                hook = cross_attn_1.register_forward_hook(
                    partial(self._hook_fn, name='cross_attn_1')
                )
                self.hooks.append(hook)
        
        cross_attn_text = getattr(self.model, 'cross_attention_text', None)
        if cross_attn_text is not None and hasattr(cross_attn_text, 'register_forward_hook'):
            self.call_count['cross_attn_2'] = 0
            hook = cross_attn_text.register_forward_hook(
                partial(self._hook_fn, name='cross_attn_2')
            )
            self.hooks.append(hook)
    
    def _hook_fn(self, module, input, output, name):
        self.call_count[name] = self.call_count.get(name, 0) + 1
        call_idx = self.call_count[name]
        scale_name = 'low' if call_idx == 1 else 'high'
        key = f'{name}_{scale_name}'
        
        if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
            attn_weights = output[1]
            self.attention_weights[key] = attn_weights.detach()
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_attention(self, name, scale='low'):
        key = f'{name}_{scale}'
        return self.attention_weights.get(key, None)
    
    def clear(self):
        self.attention_weights.clear()
        self.call_count.clear()


def load_model(ckpt_path, model_type='ViLa_MTL_MIL'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 先加载 checkpoint 以推断 prototype_number
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    state_dict_clean = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # 从 state_dict 中推断 prototype_number
    inferred_prototype_number = 2  # 默认值
    for key, value in state_dict_clean.items():
        if 'learnable_image_center' in key and isinstance(value, torch.Tensor):
            inferred_prototype_number = value.shape[0]
            print(f"Inferred prototype_number={inferred_prototype_number} from {key}")
            break
    
    if model_type == 'ViLa_MTL_MIL':
        from models.model_ViLa_MIL_MTL import ViLa_MTL_MIL_Model
        
        config = SimpleNamespace(
            input_size=512,
            hidden_size=192,
            prototype_number=inferred_prototype_number,
            num_heads=1,
            mil_aggregator='prototype',
            main_task_type='binary',
            main_task='binary',
            main_task_weight=1.0,
            num_aux_tasks=3,
            aux_task_num_classes=[4, 4, 4],
            aux_task_weights=[1.0, 1.0, 1.0],
            aux_task_losses=['ce', 'ce', 'ce'],
            use_cross_scale_contrastive=True,
            cross_scale_weight=0.8,
            contrast_temperature=0.5,
            contrast_intra_weight=1.0,
            contrast_inter_weight=0.5,
            consistency_weight=0.1,
            use_consistency=False,
            consistency_noise_std=0.1,
            text_extractor='conch',
            conch_checkpoint='/home/hzxu/CONCH/checkpoints/CONCH/',
            freeze_conch=True,
            conch_unfreeze_layers=1,
            use_text_prompts=True,
            n_ctx=16,
            class_token_position='end',
            ctx_init_std=0.02,
            prototype_init_std=0.02,
        )
        
        model = ViLa_MTL_MIL_Model(config)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    
    model.load_state_dict(state_dict_clean, strict=False)
    model = model.to(device)
    model.eval()
    
    return model


def load_h5_features(h5_path):
    with h5py.File(h5_path, 'r') as f:
        features = torch.tensor(f['features'][:], dtype=torch.float32)
        coords = torch.tensor(f['coords'][:], dtype=torch.float32)
        coords_dset = f['coords']
        patch_size = coords_dset.attrs.get('patch_size', 256)
        patch_level = coords_dset.attrs.get('patch_level', 0)
    return features, coords, patch_size, patch_level


def get_attention_with_hook(model, features_s, coords_s, features_l, coords_l):
    device = next(model.parameters()).device
    
    features_s = features_s.to(device)
    coords_s = coords_s.to(device)
    features_l = features_l.to(device)
    coords_l = coords_l.to(device)
    
    extractor = AttentionExtractor(model)
    extractor.register_hooks()
    
    with torch.no_grad():
        try:
            model([features_s.squeeze(0)], [coords_s.squeeze(0)],
                  [features_l.squeeze(0)], [coords_l.squeeze(0)],
                  return_attention=True)
        except TypeError:
            model([features_s.squeeze(0)], [coords_s.squeeze(0)],
                  [features_l.squeeze(0)], [coords_l.squeeze(0)])
    
    attn_low = extractor.get_attention('cross_attn_1', scale='low')
    attn_high = extractor.get_attention('cross_attn_1', scale='high')
    
    extractor.remove_hooks()
    
    return attn_low, attn_high, coords_s, coords_l


def save_wsi_thumbnail(wsi_path, output_path, vis_level=2, max_size=2048):
    """Save WSI thumbnail without attention overlay."""
    from openslide import OpenSlide
    
    wsi = OpenSlide(wsi_path)
    
    # 确保 vis_level 在有效范围内
    if vis_level < 0:
        vis_level = wsi.level_count - 1
    vis_level = min(vis_level, wsi.level_count - 1)
    
    downsample = wsi.level_downsamples[vis_level]
    wsi_width, wsi_height = wsi.level_dimensions[vis_level]
    
    scale = min(max_size / max(wsi_width, wsi_height), 1.0)
    out_width = int(wsi_width * scale)
    out_height = int(wsi_height * scale)
    
    print(f"  Saving WSI thumbnail: level={vis_level}, downsample={downsample:.1f}x, output={out_width}x{out_height}")
    
    wsi_img = wsi.read_region((0, 0), vis_level, (wsi_width, wsi_height)).convert('RGB')
    wsi_img = wsi_img.resize((out_width, out_height), Image.NEAREST)
    wsi_img.save(output_path, quality=85)
    
    print(f"  Saved: {output_path}")
    wsi.close()


def draw_heatmap_only(scores, coords, wsi_path, output_path, 
                      vis_level=2, alpha=0.4, cmap='jet', 
                      patch_size=256, patch_level=0, max_size=2048):
    """Draw attention heatmap only (transparent background, no WSI overlay)."""
    import matplotlib.pyplot as plt
    from openslide import OpenSlide
    
    scores_flat = scores.flatten()
    if scores_flat.max() > scores_flat.min():
        scores_norm = (scores_flat - scores_flat.min()) / (scores_flat.max() - scores_flat.min())
    else:
        scores_norm = np.ones_like(scores_flat) * 0.5
    
    wsi = OpenSlide(wsi_path)
    
    # 确保 vis_level 在有效范围内
    if vis_level < 0:
        vis_level = wsi.level_count - 1
    vis_level = min(vis_level, wsi.level_count - 1)
    
    downsample = wsi.level_downsamples[vis_level]
    wsi_width, wsi_height = wsi.level_dimensions[vis_level]
    
    scale = min(max_size / max(wsi_width, wsi_height), 1.0)
    out_width = int(wsi_width * scale)
    out_height = int(wsi_height * scale)
    
    patch_level = min(patch_level, wsi.level_count - 1)
    patch_downsample = wsi.level_downsamples[patch_level]
    
    coords_scaled_float = coords.astype(np.float32) / downsample * scale
    coords_scaled = coords_scaled_float.astype(np.int32)
    
    if len(coords) > 1:
        x_coords_unique = np.sort(np.unique(coords[:, 0]))
        y_coords_unique = np.sort(np.unique(coords[:, 1]))
        
        if len(x_coords_unique) > 1:
            x_step_original = np.median(np.diff(x_coords_unique))
        else:
            x_step_original = patch_size
        
        if len(y_coords_unique) > 1:
            y_step_original = np.median(np.diff(y_coords_unique))
        else:
            y_step_original = patch_size
        
        step_size_original = max(x_step_original, y_step_original)
        vis_patch_size = max(1, int(step_size_original / downsample * scale))
    else:
        vis_patch_size = max(1, int(patch_size * patch_downsample / downsample * scale))
    
    print(f"  Debug: patch_size={patch_size}, patch_level={patch_level}, patch_downsample={patch_downsample:.2f}")
    print(f"  Debug: vis_level={vis_level}, downsample={downsample:.2f}, scale={scale:.4f}")
    print(f"  Debug: vis_patch_size={vis_patch_size}, output_size={out_width}x{out_height}")
    print(f"  Debug: coords range - min={coords.min(axis=0)}, max={coords.max(axis=0)}")
    print(f"  Debug: coords_scaled range - min={coords_scaled.min(axis=0)}, max={coords_scaled.max(axis=0)}")
    if len(coords) > 1:
        print(f"  Debug: estimated step size (original): x={x_step_original:.1f}, y={y_step_original:.1f}, using={step_size_original:.1f}")
    
    colormap = plt.get_cmap(cmap)
    lut = np.array([colormap(i / 255.0)[:3] for i in range(256)], dtype=np.float32)
    color_indices = (scores_norm * 255).astype(np.uint8)
    colors = lut[color_indices]
    
    heatmap = np.zeros((out_height, out_width, 4), dtype=np.float32)
    
    drawn_count = 0
    for i in range(len(coords_scaled)):
        x, y = coords_scaled[i]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(out_width, x + vis_patch_size), min(out_height, y + vis_patch_size)
        if x1 < x2 and y1 < y2:
            heatmap[y1:y2, x1:x2, :3] = colors[i]
            heatmap[y1:y2, x1:x2, 3] = 1.0
            drawn_count += 1
    
    Image.fromarray((heatmap * 255).astype(np.uint8)).save(output_path)
    
    print(f"  Heatmap: {drawn_count}/{len(coords_scaled)} patches drawn, saved to {output_path}")
    wsi.close()


def process_single_wsi(model, slide_name, h5_dir, h5_dir_l, wsi_dir, output_dir, wsi_ext='mrxs', vis_level=2, max_size=2048, 
                       save_wsi=True, save_low=True, save_high=True):
    """Process a single WSI."""
    h5_path_s = os.path.join(h5_dir, f'{slide_name}.h5')
    h5_path_l = os.path.join(h5_dir_l, f'{slide_name}.h5')
    wsi_path = os.path.join(wsi_dir, f'{slide_name}.{wsi_ext}')
    
    for path, name in [(h5_path_s, 'Low-res H5'), (h5_path_l, 'High-res H5'), (wsi_path, 'WSI')]:
        if not os.path.exists(path):
            print(f"Warning: {name} not found: {path}")
            return False
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing: {slide_name}")
    
    if save_wsi:
        wsi_output = os.path.join(output_dir, f'{slide_name}_wsi.png')
        save_wsi_thumbnail(wsi_path, wsi_output, vis_level=vis_level, max_size=max_size)
    
    features_s, coords_s, patch_size_s, patch_level_s = load_h5_features(h5_path_s)
    features_l, coords_l, patch_size_l, patch_level_l = load_h5_features(h5_path_l)
    print(f"  Patches: low-res={features_s.shape[0]}, high-res={features_l.shape[0]}")
    print(f"  Patch sizes: low-res={patch_size_s} (level={patch_level_s}), high-res={patch_size_l} (level={patch_level_l})")
    
    attn_low, attn_high, coords_s, coords_l = get_attention_with_hook(model, features_s, coords_s, features_l, coords_l)
    
    if save_low and attn_low is not None:
        attn = attn_low.squeeze(0).squeeze(0)
        attn_scores = attn.mean(dim=0).cpu().numpy()
        
        output_path = os.path.join(output_dir, f'{slide_name}_heatmap_low.png')
        draw_heatmap_only(attn_scores.reshape(-1, 1), coords_s.cpu().numpy(), wsi_path, output_path,
                          vis_level=vis_level, max_size=max_size,
                          patch_size=patch_size_s, patch_level=patch_level_s)
    elif save_low:
        print("Warning: No low-resolution attention weights captured")
    
    if save_high and attn_high is not None:
        attn = attn_high.squeeze(0).squeeze(0)
        attn_scores = attn.mean(dim=0).cpu().numpy()
        
        output_path = os.path.join(output_dir, f'{slide_name}_heatmap_high.png')
        draw_heatmap_only(attn_scores.reshape(-1, 1), coords_l.cpu().numpy(), wsi_path, output_path,
                          vis_level=vis_level, max_size=max_size,
                          patch_size=patch_size_l, patch_level=patch_level_l)
    elif save_high:
        print("Warning: No high-resolution attention weights captured")
    
    return True


def find_checkpoints_and_splits(batch_path):
    """Find all checkpoints and their corresponding splits files in batch_path.
    
    Returns:
        list of tuples: [(ckpt_path, splits_path, fold_id), ...]
    """
    results = []
    
    # 查找所有 checkpoint 文件
    ckpt_files = []
    for f in os.listdir(batch_path):
        if f.startswith('s_') and f.endswith('_checkpoint.pt'):
            ckpt_files.append(f)
    
    # 按 fold id 排序
    ckpt_files.sort()
    
    for ckpt_file in ckpt_files:
        # 提取 fold id (如 s_0_checkpoint.pt -> fold_id=0)
        fold_id = ckpt_file.split('_')[1]
        
        ckpt_path = os.path.join(batch_path, ckpt_file)
        
        # 查找对应的 splits 文件
        splits_file = f'splits_{fold_id}.csv'
        splits_path = os.path.join(batch_path, splits_file)
        
        if os.path.exists(splits_path):
            results.append((ckpt_path, splits_path, fold_id))
        else:
            print(f"Warning: splits file not found for fold {fold_id}: {splits_path}")
    
    return results


def read_test_slides_from_splits(splits_path):
    """Read test slide IDs from splits CSV file.
    
    Returns:
        list of slide IDs
    """
    import csv
    
    slide_ids = []
    
    print(f"  Reading splits from: {splits_path}")
    
    with open(splits_path, 'r') as f:
        reader = csv.DictReader(f)
        print(f"  CSV columns: {reader.fieldnames}")
        
        for row in reader:
            slide_id = None
            
            # 检查是否有 'test' 列或类似的列
            if 'test' in row:
                slide_id = row['test']
            elif 'slide_id' in row:
                slide_id = row['slide_id']
            elif 'filename' in row:
                slide_id = row['filename']
            else:
                # 尝试获取第一列的值
                if reader.fieldnames:
                    first_col = reader.fieldnames[0]
                    slide_id = row[first_col]
            
            if slide_id:
                slide_id = slide_id.strip()
                
                # 去除常见的 WSI 文件后缀
                wsi_extensions = ['.mrxs', '.svs', '.tif', '.tiff', '.czi', '.ndpi', '.vms', '.vmu', '.bif', '.jpeg', '.jpg']
                for ext in wsi_extensions:
                    if slide_id.lower().endswith(ext):
                        slide_id = slide_id[:-len(ext)]
                        break
                
                slide_ids.append(slide_id)
    
    print(f"  Read {len(slide_ids)} test slides: {slide_ids[:5]}..." if len(slide_ids) > 5 else f"  Read {len(slide_ids)} test slides: {slide_ids}")
    
    return slide_ids


def main():
    parser = argparse.ArgumentParser(description='ViLa-MIL Attention Visualization')
    parser.add_argument('--ckpt_path', type=str, default=None, help='Single checkpoint path')
    parser.add_argument('--batch_path', type=str, default=None, help='Batch path containing multiple checkpoints and splits')
    parser.add_argument('--model_type', type=str, default='ViLa_MTL_MIL')
    parser.add_argument('--h5_dir', type=str, required=True, help='Low-res H5 dir')
    parser.add_argument('--h5_dir_l', type=str, default=None, help='High-res H5 dir')
    parser.add_argument('--wsi_dir', type=str, required=True)
    parser.add_argument('--wsi_ext', type=str, default='mrxs')
    parser.add_argument('--output_dir', type=str, default='attention_results')
    parser.add_argument('--vis_level', type=int, default=2)
    parser.add_argument('--max_size', type=int, default=2048)
    parser.add_argument('--slide_ids', type=str, nargs='+', default=None)
    parser.add_argument('--max_slides', type=int, default=None)
    
    parser.add_argument('--save_wsi', action='store_true', default=True, help='Save WSI thumbnail')
    parser.add_argument('--no_save_wsi', action='store_true', help='Do not save WSI thumbnail')
    parser.add_argument('--save_low', action='store_true', default=True, help='Save low-resolution heatmap')
    parser.add_argument('--no_save_low', action='store_true', help='Do not save low-resolution heatmap')
    parser.add_argument('--save_high', action='store_true', default=True, help='Save high-resolution heatmap')
    parser.add_argument('--no_save_high', action='store_true', help='Do not save high-resolution heatmap')
    
    args = parser.parse_args()
    
    # 检查参数
    if args.ckpt_path is None and args.batch_path is None:
        parser.error("Either --ckpt_path or --batch_path must be provided")
    
    if args.h5_dir_l is None:
        args.h5_dir_l = args.h5_dir
    
    # 处理输出控制参数
    save_wsi = args.save_wsi and not args.no_save_wsi
    save_low = args.save_low and not args.no_save_low
    save_high = args.save_high and not args.no_save_high
    
    # 批量处理模式
    if args.batch_path:
        print(f"\nBatch processing mode")
        print(f"Batch path: {args.batch_path}")
        
        # 查找所有 checkpoint 和 splits
        checkpoints = find_checkpoints_and_splits(args.batch_path)
        
        if not checkpoints:
            print("No checkpoints found!")
            return
        
        print(f"Found {len(checkpoints)} folds")
        print(f"Output options: WSI={save_wsi}, Low-res={save_low}, High-res={save_high}")
        print("=" * 50)
        
        total_success = 0
        total_slides = 0
        interrupted = False
        
        try:
            for ckpt_path, splits_path, fold_id in checkpoints:
                print(f"\n{'=' * 50}")
                print(f"Processing fold {fold_id}")
                print(f"Checkpoint: {ckpt_path}")
                print(f"Splits: {splits_path}")
                
                # 加载模型
                model = load_model(ckpt_path, args.model_type)
                
                # 读取测试样本
                test_slides = read_test_slides_from_splits(splits_path)
                
                if args.slide_ids:
                    # 如果用户指定了 slide_ids，只处理这些样本
                    test_slides = [s for s in test_slides if s in args.slide_ids]
                
                if args.max_slides:
                    test_slides = test_slides[:args.max_slides]
                
                print(f"Test slides: {len(test_slides)}")
                print("Press Ctrl+C to stop...")
                
                # 为每折创建子目录
                fold_output_dir = os.path.join(args.output_dir, f'fold_{fold_id}')
                
                success = 0
                for i, name in enumerate(test_slides):
                    print(f"\n[{i+1}/{len(test_slides)}]")
                    if process_single_wsi(model, name, args.h5_dir, args.h5_dir_l, args.wsi_dir, fold_output_dir, args.wsi_ext,
                                         vis_level=args.vis_level, max_size=args.max_size,
                                         save_wsi=save_wsi, save_low=save_low, save_high=save_high):
                        success += 1
                
                total_success += success
                total_slides += len(test_slides)
                print(f"\nFold {fold_id} completed: {success}/{len(test_slides)} slides")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user (Ctrl+C)")
            interrupted = True
        
        print(f"\n{'=' * 50}")
        if interrupted:
            print(f"Stopped! {total_success}/{total_slides} slides processed before interruption")
        else:
            print(f"All folds completed! {total_success}/{total_slides} slides processed")
    
    # 单 checkpoint 模式
    else:
        model = load_model(args.ckpt_path, args.model_type)
        
        if args.slide_ids:
            slide_names = args.slide_ids
        else:
            h5_files = [f for f in os.listdir(args.h5_dir) if f.endswith('.h5')]
            slide_names = [f.replace('.h5', '') for f in h5_files]
        
        if args.max_slides:
            slide_names = slide_names[:args.max_slides]
        
        print(f"\nProcessing {len(slide_names)} slides")
        print(f"Output options: WSI={save_wsi}, Low-res={save_low}, High-res={save_high}")
        print("=" * 50)
        print("Press Ctrl+C to stop...")
        
        success = 0
        interrupted = False
        try:
            for i, name in enumerate(slide_names):
                print(f"\n[{i+1}/{len(slide_names)}]")
                if process_single_wsi(model, name, args.h5_dir, args.h5_dir_l, args.wsi_dir, args.output_dir, args.wsi_ext,
                                     vis_level=args.vis_level, max_size=args.max_size,
                                     save_wsi=save_wsi, save_low=save_low, save_high=save_high):
                    success += 1
        except KeyboardInterrupt:
            print("\n\nInterrupted by user (Ctrl+C)")
            interrupted = True
        
        print(f"\n{'=' * 50}")
        if interrupted:
            print(f"Stopped! {success}/{len(slide_names)} slides processed before interruption")
        else:
            print(f"Done! {success}/{len(slide_names)} slides processed")


if __name__ == '__main__':
    main()