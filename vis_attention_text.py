"""
ViLa-MIL Text-to-Patch Attention Visualization

Visualizes which patches are most relevant to each text class prompt.

Usage:
    python vis_attention_text.py \
        --ckpt_path tj0.0002,0.0001/tj_conch_aux0_aux_label_1_aux_label_2_p2_crosca0.8_step_gpt_nopre_s0/s_0_checkpoint.pt \
        --h5_dir /home/hzxu/ViLa-MIL-main/feature_dir/tj/10feattj/h5_files/ \
        --h5_dir_l /home/hzxu/ViLa-MIL-main/feature_dir/tj/20feattj/h5_files/ \
        --wsi_dir /home/hzxu/MIL-Liver/TJ_datasets/mrxs/ \
        --output_dir text_attention_results \
        --slide_ids A2024004
"""

import os
import sys
import argparse
import csv
import numpy as np
import torch
import h5py
from PIL import Image
from types import SimpleNamespace
from functools import partial

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


class TextAttentionExtractor:
    """Extract text-to-patch attention weights using hooks."""

    def __init__(self, model):
        self.model = model
        self.attention_weights = {}
        self.hooks = []
        self.call_count = {}

    def register_hooks(self):
        """Register hooks on cross_attention_text layers."""
        cross_attn_text = getattr(self.model, 'cross_attention_text', None)

        if cross_attn_text is None:
            return False

        for idx, attn_module in enumerate(cross_attn_text):
            if attn_module is not None:
                self.call_count[idx] = 0
                hook = attn_module.register_forward_hook(
                    partial(self._hook_fn, name=idx)
                )
                self.hooks.append(hook)
        return True

    def _hook_fn(self, module, input, output, name):
        """Hook function to capture attention weights.
        
        cross_attention_text is called twice per task:
        - 1st call: low-res (prototypes + low-res patches)
        - 2nd call: high-res (prototypes + high-res patches)
        """
        self.call_count[name] = self.call_count.get(name, 0) + 1
        call_idx = self.call_count[name]
        scale_name = 'low' if call_idx == 1 else 'high'

        if isinstance(output, tuple) and len(output) == 2:
            attn_output, attn_weights = output
            if attn_weights is not None:
                key = f'text_attn_{name}_{scale_name}'
                self.attention_weights[key] = attn_weights.detach()

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_attention(self, task_idx, scale='low'):
        key = f'text_attn_{task_idx}_{scale}'
        return self.attention_weights.get(key, None)

    def clear(self):
        self.attention_weights.clear()
        self.call_count.clear()


DATASET_TEXT_PROMPT_FILES = {
    'tj': 'text_prompt/tj_text_prompt_gpt_nopre.csv',
    'sr386': 'text_prompt/sr386_text_prompt_gemini_nopre.csv',
    'gmu': 'text_prompt/gmu_text_prompt5total_gptnew_nopre.csv',
    'crc': 'text_prompt/crc_text_prompt_gpt_nopre.csv',
}


def parse_experiment_txt(txt_path):
    """从 experiment_xxx.txt 的 Command line 行中解析 --text_prompt_file 和 --aux_label_cols。"""
    text_prompt_file = None
    aux_label_cols = None
    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if 'Command line:' in line:
                tokens = line.split('Command line:', 1)[1].strip().split()
                for i, tok in enumerate(tokens):
                    if tok == '--text_prompt_file' and i + 1 < len(tokens):
                        text_prompt_file = tokens[i + 1]
                    elif tok == '--aux_label_cols' and i + 1 < len(tokens):
                        aux_label_cols = tokens[i + 1]
                break
    return text_prompt_file, aux_label_cols


def resolve_text_prompt_args(args):
    """解析文本提示词路径与 aux_label_cols：显式参数 > experiment_txt > dataset 默认值。"""
    text_prompt_file = args.text_prompt_file
    aux_label_cols = args.aux_label_cols
    experiment_txt = args.experiment_txt
    if experiment_txt is None and getattr(args, 'batch_path', None):
        import glob
        found = sorted(glob.glob(os.path.join(args.batch_path, 'experiment_*.txt')))
        if len(found) == 1:
            experiment_txt = found[0]
            print(f"Auto-detected experiment txt: {experiment_txt}")
        elif len(found) > 1:
            print(f"Warning: multiple experiment txt in batch_path, set --experiment_txt explicitly: {found}")
    if experiment_txt:
        exp_tpf, exp_cols = parse_experiment_txt(experiment_txt)
        if text_prompt_file is None:
            text_prompt_file = exp_tpf
        if aux_label_cols is None:
            aux_label_cols = exp_cols
    if text_prompt_file is None and args.dataset:
        rel = DATASET_TEXT_PROMPT_FILES.get(args.dataset.lower())
        if rel:
            text_prompt_file = os.path.join(project_root, rel)
    if text_prompt_file and not os.path.exists(text_prompt_file):
        candidate = os.path.join(project_root, 'text_prompt', os.path.basename(text_prompt_file))
        if os.path.exists(candidate):
            text_prompt_file = candidate
        else:
            print(f"Warning: text prompt file not found: {text_prompt_file}")
    if text_prompt_file:
        print(f"Using text prompt file: {text_prompt_file}")
        print(f"Using aux_label_cols: {aux_label_cols}")
    return text_prompt_file, aux_label_cols


def build_task_configs_from_prompts(text_prompt_file, aux_label_cols, aux_task_num_classes):
    """从训练用的 text prompt CSV 构建 task_configs。

    CSV 每列(taskN)的提示词按 [class0_low..classN_low, class0_high..classN_high] 阻塞顺序排列，
    与模型内 text_features[:num_classes]=low / [num_classes:]=high 的假设一致，因此直接按原顺序使用。
    """
    import pandas as pd
    df = pd.read_csv(text_prompt_file, header=0)

    if aux_label_cols:
        task_indices = []
        for col in [c.strip() for c in str(aux_label_cols).split(',') if c.strip()]:
            try:
                task_indices.append(int(col.split('_')[-1]))
            except ValueError:
                task_indices.append(len(task_indices))
    else:
        task_indices = list(range(len(aux_task_num_classes)))

    task_configs = []
    for order_idx, num_classes in enumerate(aux_task_num_classes):
        csv_task_idx = task_indices[order_idx] if order_idx < len(task_indices) else order_idx
        col_name = f'task{csv_task_idx + 1}'
        if col_name in df.columns:
            col = df[col_name]
        elif csv_task_idx < len(df.columns):
            col = df.iloc[:, csv_task_idx]
        else:
            col = None

        prompts = [str(p) for p in col.dropna().tolist()] if col is not None else []
        used_real = True
        if len(prompts) != 2 * num_classes:
            print(f"  Warning: task {order_idx} (col {col_name}) has {len(prompts)} prompts, "
                  f"expected {2 * num_classes}; falling back to placeholders.")
            prompts = [f'class_{i}_low' for i in range(num_classes)] + \
                      [f'class_{i}_high' for i in range(num_classes)]
            used_real = False
        task_configs.append({'text_prompts': prompts, 'num_classes': num_classes})
        _print_task_config_check(order_idx, col_name, num_classes, prompts, used_real)
    return task_configs


def load_model(ckpt_path, model_type='ViLa_MTL_MIL',
               text_prompt_file=None, aux_label_cols=None):
    """Load ViLa-MIL model from checkpoint."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    state_dict_clean = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    inferred_prototype_number = 2
    for key, value in state_dict_clean.items():
        if 'learnable_image_center' in key and isinstance(value, torch.Tensor):
            inferred_prototype_number = value.shape[0]
            break
    
    aux_task_num_classes = []
    task_idx = 0
    while True:
        weight_key = f'task_heads.{task_idx}.low.weight'
        if weight_key in state_dict_clean:
            num_classes = state_dict_clean[weight_key].shape[0]
            aux_task_num_classes.append(num_classes)
            task_idx += 1
        else:
            break
    
    num_aux_tasks = len(aux_task_num_classes)
    
    main_task_type = 'binary'
    if any('cox_regressor' in key for key in state_dict_clean.keys()):
        main_task_type = 'cox'
    elif any('mdn_regressor' in key for key in state_dict_clean.keys()):
        main_task_type = 'mdn'
    
    if text_prompt_file is not None:
        task_configs = build_task_configs_from_prompts(
            text_prompt_file, aux_label_cols, aux_task_num_classes)
    else:
        print("\n[Check] No --text_prompt_file/--experiment_txt/--dataset given; "
              "using placeholder prompts (class_i_low/class_i_high).")
        task_configs = []
        for task_idx in range(num_aux_tasks):
            num_classes = aux_task_num_classes[task_idx]
            text_prompts = [f'class_{i}_low' for i in range(num_classes)] + \
                           [f'class_{i}_high' for i in range(num_classes)]
            task_configs.append({
                'text_prompts': text_prompts,
                'num_classes': num_classes
            })
            _print_task_config_check(task_idx, 'placeholder', num_classes, text_prompts, False)
    
    if model_type == 'ViLa_MTL_MIL':
        from models.model_ViLa_MIL_MTL import ViLa_MTL_MIL_Model
        
        config = SimpleNamespace(
            input_size=512,
            hidden_size=192,
            prototype_number=inferred_prototype_number,
            num_heads=1,
            mil_aggregator='prototype',
            main_task_type=main_task_type,
            main_task=main_task_type,
            main_task_weight=1.0,
            num_aux_tasks=num_aux_tasks,
            aux_task_num_classes=aux_task_num_classes,
            aux_task_weights=[1.0] * num_aux_tasks,
            aux_task_losses=['ce'] * num_aux_tasks,
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
            task_configs=task_configs,
            zero_shot_config={},
        )
        
        model = ViLa_MTL_MIL_Model(config)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    
    model.load_state_dict(state_dict_clean, strict=False)
    model = model.to(device)
    model.eval()
    
    return model, inferred_prototype_number


def _print_task_config_check(order_idx, col_name, num_classes, prompts, used_real):
    """打印校验信息：aux 任务序号 -> CSV 列名 + 类别数 + 提示词条数，并逐条列出实际文本提示。"""
    print(f"\n[Check] aux task {order_idx} -> CSV column '{col_name}' | "
          f"num_classes={num_classes} | prompts={len(prompts)} (expect {2 * num_classes})")
    if not used_real:
        print("  (placeholders, not real text prompts)")
    for i in range(num_classes):
        print(f"  class_{i}_low : {prompts[i] if i < len(prompts) else '<missing>'}")
    for i in range(num_classes):
        j = num_classes + i
        print(f"  class_{i}_high: {prompts[j] if j < len(prompts) else '<missing>'}")


def load_h5_features(h5_path):
    with h5py.File(h5_path, 'r') as f:
        features = torch.tensor(f['features'][:], dtype=torch.float32)
        coords = torch.tensor(f['coords'][:], dtype=torch.float32)
        coords_dset = f['coords']
        patch_size = coords_dset.attrs.get('patch_size', 256)
        patch_level = coords_dset.attrs.get('patch_level', 0)
    return features, coords, patch_size, patch_level


def get_text_attention(model, features_s, coords_s, features_l, coords_l, task_idx=0):
    """Get text-to-patch attention weights using hooks.
    
    Returns both low-res and high-res attention (cross_attention_text is called twice).
    """
    device = next(model.parameters()).device

    features_s = features_s.to(device)
    coords_s = coords_s.to(device)
    features_l = features_l.to(device)
    coords_l = coords_l.to(device)

    extractor = TextAttentionExtractor(model)
    hooks_registered = extractor.register_hooks()

    if not hooks_registered:
        return None, None, coords_s, coords_l

    with torch.no_grad():
        outputs = model([features_s.squeeze(0)], [coords_s.squeeze(0)],
                       [features_l.squeeze(0)], [coords_l.squeeze(0)])

    attn_low = extractor.get_attention(task_idx, scale='low')
    attn_high = extractor.get_attention(task_idx, scale='high')

    extractor.remove_hooks()

    return attn_low, attn_high, coords_s, coords_l


def draw_heatmap_per_class(scores_list, coords, wsi_path, output_dir,
                            class_names=None, task_idx=0, vis_level=3, alpha=0.4,
                            cmap='jet', patch_size=256, patch_level=0, max_size=1024,
                            save_wsi=True, save_heatmap=True, heatmap_only=False,
                            suffix='', shared_scale=False):
    """Draw separate heatmap for each text class (same coordinate logic as vis_attention.py).

    支持 OpenSlide (mrxs/svs/tif) 和 CZI (via aicspylibczi) 两种 WSI 格式。
    对于 CZI mosaic 文件，直接基于 base resolution 做坐标映射，不走 level 系统。
    """
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    # 根据扩展名选择 WSI 读取方式
    is_czi = wsi_path.lower().endswith('.czi')

    if is_czi:
        # --- CZI 路径：使用 viz_attention_czi.CZIReader ---
        from viz_attention_czi import CZIReader
        wsi = CZIReader(wsi_path)
        base_w, base_h = wsi.dimensions
        is_mosaic = wsi._is_mosaic

        if is_mosaic:
            # mosaic: 直接基于 base resolution 映射
            scale = min(max_size / max(base_w, base_h), 1.0)
            out_width = int(base_w * scale)
            out_height = int(base_h * scale)
            downsample = 1.0  # base resolution，无 downsample
            patch_downsample = 1.0
        else:
            if vis_level < 0:
                vis_level = wsi.level_count - 1
            vis_level = min(vis_level, wsi.level_count - 1)
            level_dims = wsi.level_dimensions
            if vis_level >= len(level_dims):
                vis_level = len(level_dims) - 1
            level_downsamples = wsi.level_downsamples
            downsample = level_downsamples[vis_level] if vis_level < len(level_downsamples) else level_downsamples[-1]
            wsi_width, wsi_height = level_dims[vis_level]
            scale = min(max_size / max(wsi_width, wsi_height), 1.0)
            out_width = int(wsi_width * scale)
            out_height = int(wsi_height * scale)
            patch_downsample = 1.0
    else:
        # --- OpenSlide 路径 ---
        from openslide import OpenSlide
        wsi = OpenSlide(wsi_path)

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
    if len(coords_scaled) > 1:
        print(f"  Debug: estimated step size (original): x={x_step_original:.1f}, y={y_step_original:.1f}, using={step_size_original:.1f}")

    colormap = plt.get_cmap(cmap)
    lut = np.array([colormap(i / 255.0)[:3] for i in range(256)], dtype=np.float32)

    if save_wsi:
        if is_czi and is_mosaic:
            # mosaic: 直接按 scale 读取
            img = wsi._czi.read_mosaic(C=0, scale_factor=scale)
            img = CZIReader._normalize_to_hwc_uint8(img)
            wsi_thumb = Image.fromarray(img).convert('RGB')
            if wsi_thumb.size != (out_width, out_height):
                wsi_thumb = wsi_thumb.resize((out_width, out_height), Image.LANCZOS)
        else:
            wsi_thumb = wsi.read_region((0, 0), vis_level, (wsi_width, wsi_height)).convert('RGB')
            wsi_thumb = wsi_thumb.resize((out_width, out_height), Image.NEAREST)
        wsi_thumb_path = os.path.join(output_dir, f'task{task_idx}{suffix}_wsi.png')
        wsi_thumb.save(wsi_thumb_path, quality=85)

    if heatmap_only:
        wsi_img_np = None
    else:
        if is_czi and is_mosaic:
            img = wsi._czi.read_mosaic(C=0, scale_factor=scale)
            img = CZIReader._normalize_to_hwc_uint8(img)
            wsi_img = Image.fromarray(img).convert('RGBA')
            if wsi_img.size != (out_width, out_height):
                wsi_img = wsi_img.resize((out_width, out_height), Image.LANCZOS)
        else:
            wsi_img = wsi.read_region((0, 0), vis_level, (wsi_width, wsi_height)).convert('RGBA')
            wsi_img = wsi_img.resize((out_width, out_height), Image.NEAREST)
        wsi_img_np = np.array(wsi_img, dtype=np.float32) / 255.0

    wsi.close()

    if not save_heatmap:
        return

    shared_min = shared_max = None
    if shared_scale and len(scores_list) > 0:
        all_scores = np.concatenate([s.flatten() for s in scores_list])
        shared_min = float(all_scores.min())
        shared_max = float(all_scores.max())
        print(f"  [shared_scale] global range across {len(scores_list)} classes: "
              f"[{shared_min:.6f}, {shared_max:.6f}]")

    for class_idx, scores in enumerate(scores_list):
        if class_names and class_idx < len(class_names):
            class_name = class_names[class_idx]
        else:
            class_name = f'class_{class_idx}'

        scores_flat = scores.flatten()
        if shared_scale and shared_max is not None and shared_max > shared_min:
            scores_norm = np.clip((scores_flat - shared_min) / (shared_max - shared_min), 0.0, 1.0)
        elif scores_flat.max() > scores_flat.min():
            scores_norm = (scores_flat - scores_flat.min()) / (scores_flat.max() - scores_flat.min())
        else:
            scores_norm = np.ones_like(scores_flat) * 0.5

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

        if heatmap_only:
            output_path = os.path.join(output_dir, f'task{task_idx}{suffix}_{class_name}.png')
            Image.fromarray((heatmap * 255).astype(np.uint8)).save(output_path)
        else:
            mask = np.any(heatmap > 0, axis=2)
            overlay = wsi_img_np.copy()
            overlay[mask, :3] = wsi_img_np[mask, :3] * (1 - alpha) + heatmap[mask, :3] * alpha
            output_path = os.path.join(output_dir, f'task{task_idx}{suffix}_{class_name}.png')
            Image.fromarray((overlay * 255).astype(np.uint8)).save(output_path, quality=85)

        print(f"  {class_name}: {drawn_count}/{len(coords_scaled)} patches drawn")



def process_single_wsi(model, slide_name, h5_dir, h5_dir_l, wsi_dir, output_dir, 
                       wsi_ext='mrxs', task_idx=0, class_names=None, 
                       num_protos=2, vis_level=3, max_size=1024,
                       save_wsi=True, save_heatmap=True, heatmap_only=False,
                       save_low=True, save_high=True, shared_scale=False):
    """Process a single WSI."""
    h5_path_s = os.path.join(h5_dir, f'{slide_name}.h5')
    h5_path_l = os.path.join(h5_dir_l, f'{slide_name}.h5')
    wsi_path = os.path.join(wsi_dir, f'{slide_name}.{wsi_ext}')
    
    for path, name in [(h5_path_s, 'Low-res H5'), (h5_path_l, 'High-res H5'), (wsi_path, 'WSI')]:
        if not os.path.exists(path):
            print(f"Warning: {name} not found: {path}")
            return False
    
    print(f"Processing: {slide_name}", flush=True)
    features_s, coords_s, patch_size_s, patch_level_s = load_h5_features(h5_path_s)
    features_l, coords_l, patch_size_l, patch_level_l = load_h5_features(h5_path_l)
    print(f"  Patches: low-res={features_s.shape[0]}, high-res={features_l.shape[0]}", flush=True)
    print(f"  Patch sizes: low-res={patch_size_s} (level={patch_level_s}), high-res={patch_size_l} (level={patch_level_l})", flush=True)
    
    attn_low, attn_high, coords_s, coords_l = get_text_attention(
        model, features_s, coords_s, features_l, coords_l, task_idx)
    
    slide_output_dir = os.path.join(output_dir, slide_name)
    os.makedirs(slide_output_dir, exist_ok=True)
    
    processed = False
    
    if save_low and attn_low is not None:
        print(f"  Low-res text attention shape: {attn_low.shape}")
        attn = attn_low
        while attn.dim() > 2:
            attn = attn.squeeze(0)
        
        text_len, image_len = attn.shape
        num_patches = features_s.shape[0]
        
        if image_len >= num_protos + num_patches:
            attn_patches = attn[:, num_protos:num_protos + num_patches]
        elif image_len == num_patches:
            attn_patches = attn
        else:
            print(f"  Warning: Low-res shape mismatch. image_len={image_len}, expected={num_protos + num_patches}")
            attn_patches = None
        
        if attn_patches is not None:
            attn_patches = attn_patches.cpu().numpy()
            num_classes = attn_patches.shape[0]
            names = _get_class_names(class_names, num_classes)
            scores = [attn_patches[i] for i in range(num_classes)]
            print(f"  Low-res: {num_classes} classes")
            for i, s in enumerate(scores):
                print(f"    {names[i]}: range [{s.min():.4f}, {s.max():.4f}]")
            draw_heatmap_per_class(scores, coords_s.cpu().numpy(), wsi_path,
                                   slide_output_dir, class_names=names, task_idx=task_idx,
                                   vis_level=vis_level, max_size=max_size,
                                   save_wsi=save_wsi, save_heatmap=save_heatmap,
                                   heatmap_only=heatmap_only, suffix='_low',
                                   patch_size=patch_size_s, patch_level=patch_level_s,
                                   shared_scale=shared_scale)
            processed = True
    elif save_low:
        print("  Warning: No low-res text attention weights captured")
    
    if save_high and attn_high is not None:
        print(f"  High-res text attention shape: {attn_high.shape}")
        attn = attn_high
        while attn.dim() > 2:
            attn = attn.squeeze(0)
        
        text_len, image_len = attn.shape
        num_patches = features_l.shape[0]
        
        if image_len >= num_protos + num_patches:
            attn_patches = attn[:, num_protos:num_protos + num_patches]
        elif image_len == num_patches:
            attn_patches = attn
        else:
            print(f"  Warning: High-res shape mismatch. image_len={image_len}, expected={num_protos + num_patches}")
            attn_patches = None
        
        if attn_patches is not None:
            attn_patches = attn_patches.cpu().numpy()
            num_classes = attn_patches.shape[0]
            names = _get_class_names(class_names, num_classes)
            scores = [attn_patches[i] for i in range(num_classes)]
            print(f"  High-res: {num_classes} classes")
            for i, s in enumerate(scores):
                print(f"    {names[i]}: range [{s.min():.4f}, {s.max():.4f}]")
            draw_heatmap_per_class(scores, coords_l.cpu().numpy(), wsi_path,
                                   slide_output_dir, class_names=names, task_idx=task_idx,
                                   vis_level=vis_level, max_size=max_size,
                                   save_wsi=False, save_heatmap=save_heatmap,
                                   heatmap_only=heatmap_only, suffix='_high',
                                   patch_size=patch_size_l, patch_level=patch_level_l,
                                   shared_scale=shared_scale)
            processed = True
    elif save_high:
        print("  Warning: No high-res text attention weights captured")
    
    return processed


def _get_class_names(class_names, num_classes):
    if class_names is None:
        return [f'class_{i}' for i in range(num_classes)]
    if len(class_names) < num_classes:
        return class_names + [f'class_{i}' for i in range(len(class_names), num_classes)]
    return class_names[:num_classes]


def find_checkpoints_and_splits(batch_path):
    """Find all checkpoint and splits files in batch path."""
    import glob
    
    checkpoint_pattern = os.path.join(batch_path, 's_*_checkpoint.pt')
    splits_pattern = os.path.join(batch_path, 'splits_*.csv')
    
    checkpoints = sorted(glob.glob(checkpoint_pattern))
    splits_files = sorted(glob.glob(splits_pattern))
    
    pairs = []
    for ckpt in checkpoints:
        fold_id = None
        basename = os.path.basename(ckpt)
        if 's_' in basename and '_checkpoint' in basename:
            parts = basename.split('s_')[1].split('_checkpoint')[0]
            try:
                fold_id = int(parts)
            except ValueError:
                pass
        
        if fold_id is not None:
            splits_path = os.path.join(batch_path, f'splits_{fold_id}.csv')
            if os.path.exists(splits_path):
                pairs.append((fold_id, ckpt, splits_path))
    
    if not pairs:
        for ckpt in checkpoints:
            pairs.append((0, ckpt, None))
    
    return pairs


def read_test_slides_from_splits(splits_path):
    """Read test slide IDs from splits CSV file."""
    slide_ids = []
    
    with open(splits_path, 'r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            slide_id = None
            
            if 'test' in row:
                slide_id = row['test']
            elif 'slide_id' in row:
                slide_id = row['slide_id']
            elif 'filename' in row:
                slide_id = row['filename']
            else:
                if reader.fieldnames:
                    first_col = reader.fieldnames[0]
                    slide_id = row[first_col]
            
            if slide_id:
                slide_id = slide_id.strip()
                
                wsi_extensions = ['.mrxs', '.svs', '.tif', '.tiff', '.czi', '.ndpi', '.vms', '.vmu', '.bif', '.jpeg', '.jpg']
                for ext in wsi_extensions:
                    if slide_id.lower().endswith(ext):
                        slide_id = slide_id[:-len(ext)]
                        break
                
                slide_ids.append(slide_id)
    
    return slide_ids


def main():
    parser = argparse.ArgumentParser(description='ViLa-MIL Text-to-Patch Attention Visualization')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='Single-fold checkpoint .pt (for single-fold/debug only)')
    parser.add_argument('--batch_path', type=str, default=None,
                        help='Experiment folder containing s_*_checkpoint.pt and splits_*.csv; '
                             'runs all folds (use this for 5-fold CV)')
    parser.add_argument('--model_type', type=str, default='ViLa_MTL_MIL')
    parser.add_argument('--h5_dir', type=str, required=True, help='Low-res H5 dir')
    parser.add_argument('--h5_dir_l', type=str, default=None, help='High-res H5 dir')
    parser.add_argument('--wsi_dir', type=str, required=True)
    parser.add_argument('--wsi_ext', type=str, default='mrxs')
    parser.add_argument('--output_dir', type=str, default='text_attention_results')
    parser.add_argument('--vis_level', type=int, default=3)
    parser.add_argument('--max_size', type=int, default=1024)
    parser.add_argument('--task_idx', type=int, default=0, help='Task index for text attention')
    parser.add_argument('--class_names', type=str, nargs='+', default=None, 
                        help='Class names for each text prompt')
    parser.add_argument('--slide_ids', type=str, nargs='+', default=None)
    parser.add_argument('--max_slides', type=int, default=None)
    parser.add_argument('--save_wsi', action='store_true', default=True)
    parser.add_argument('--no_save_wsi', action='store_true', default=False)
    parser.add_argument('--save_heatmap', action='store_true', default=True)
    parser.add_argument('--no_save_heatmap', action='store_true', default=False)
    parser.add_argument('--heatmap_only', action='store_true', default=False,
                        help='Save heatmap only, no WSI overlay (still needs WSI for dimensions)')
    parser.add_argument('--save_low', action='store_true', default=True,
                        help='Save low-resolution text attention heatmap')
    parser.add_argument('--no_save_low', action='store_true', default=False)
    parser.add_argument('--save_high', action='store_true', default=True,
                        help='Save high-resolution text attention heatmap')
    parser.add_argument('--no_save_high', action='store_true', default=False)
    parser.add_argument('--text_prompt_file', type=str, default=None,
                        help='CSV with real text prompts (same as training --text_prompt_file)')
    parser.add_argument('--aux_label_cols', type=str, default=None,
                        help='Comma-separated aux label cols, e.g. aux_label_1 (same as training)')
    parser.add_argument('--experiment_txt', type=str, default=None,
                        help='Path to experiment_xxx.txt; auto-parse text_prompt_file/aux_label_cols')
    parser.add_argument('--dataset', type=str, default=None,
                        choices=['tj', 'sr386', 'gmu', 'crc'],
                        help='Dataset name for default text prompt file lookup')
    parser.add_argument('--shared_scale', action='store_true', default=False,
                        help='Normalize all classes of a slide (same scale) with a shared vmin/vmax '
                             'so cross-class heatmaps are comparable')
    
    args = parser.parse_args()
    
    text_prompt_file, aux_label_cols = resolve_text_prompt_args(args)
    
    if args.h5_dir_l is None:
        args.h5_dir_l = args.h5_dir
    
    if args.no_save_wsi:
        args.save_wsi = False
    if args.no_save_heatmap:
        args.save_heatmap = False
    if args.heatmap_only:
        args.save_wsi = False
    if args.no_save_low:
        args.save_low = False
    if args.no_save_high:
        args.save_high = False
    
    if not args.ckpt_path and not args.batch_path:
        parser.error("Either --ckpt_path or --batch_path must be provided")
    
    if args.batch_path:
        checkpoint_pairs = find_checkpoints_and_splits(args.batch_path)
        
        total_success = 0
        total_slides = 0
        interrupted = False
        
        try:
            for fold_id, ckpt_path, splits_path in checkpoint_pairs:
                print(f"\n{'=' * 50}")
                print(f"Processing Fold {fold_id}")
                print(f"{'=' * 50}")
                
                model, num_protos = load_model(ckpt_path, args.model_type,
                                               text_prompt_file=text_prompt_file,
                                               aux_label_cols=aux_label_cols)
                
                fold_output_dir = os.path.join(args.output_dir, f'fold_{fold_id}')
                
                if splits_path:
                    slide_names = read_test_slides_from_splits(splits_path)
                elif args.slide_ids:
                    slide_names = args.slide_ids
                else:
                    h5_files = [f for f in os.listdir(args.h5_dir) if f.endswith('.h5')]
                    slide_names = [f.replace('.h5', '') for f in h5_files]
                
                if args.max_slides:
                    slide_names = slide_names[:args.max_slides]
                
                print(f"\nProcessing {len(slide_names)} slides")
                
                success = 0
                for i, name in enumerate(slide_names):
                    print(f"\n[{i+1}/{len(slide_names)}]")
                    if process_single_wsi(model, name, args.h5_dir, args.h5_dir_l, args.wsi_dir, 
                                         fold_output_dir, args.wsi_ext, args.task_idx, 
                                         args.class_names, num_protos=num_protos,
                                         vis_level=args.vis_level, max_size=args.max_size,
                                         save_wsi=args.save_wsi, save_heatmap=args.save_heatmap,
                                         heatmap_only=args.heatmap_only,
                                         save_low=args.save_low, save_high=args.save_high,
                                         shared_scale=args.shared_scale):
                        success += 1
                
                total_success += success
                total_slides += len(slide_names)
                print(f"\nFold {fold_id} done! {success}/{len(slide_names)} slides processed")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user (Ctrl+C)")
            interrupted = True
        
        print(f"\n{'=' * 50}")
        if interrupted:
            print(f"Stopped! {total_success}/{total_slides} slides processed before interruption")
        else:
            print(f"Done! {total_success}/{total_slides} slides processed")
    
    else:
        model, num_protos = load_model(args.ckpt_path, args.model_type,
                                       text_prompt_file=text_prompt_file,
                                       aux_label_cols=aux_label_cols)
        
        if args.slide_ids:
            slide_names = args.slide_ids
        else:
            h5_files = [f for f in os.listdir(args.h5_dir) if f.endswith('.h5')]
            slide_names = [f.replace('.h5', '') for f in h5_files]
        
        if args.max_slides:
            slide_names = slide_names[:args.max_slides]
        
        print(f"\nProcessing {len(slide_names)} slides")
        print("=" * 50)
        
        success = 0
        interrupted = False
        
        try:
            for i, name in enumerate(slide_names):
                print(f"\n[{i+1}/{len(slide_names)}]")
                if process_single_wsi(model, name, args.h5_dir, args.h5_dir_l, args.wsi_dir, 
                                     args.output_dir, args.wsi_ext, args.task_idx, 
                                     args.class_names, num_protos=num_protos,
                                     vis_level=args.vis_level, max_size=args.max_size,
                                     save_wsi=args.save_wsi, save_heatmap=args.save_heatmap,
                                     heatmap_only=args.heatmap_only,
                                     save_low=args.save_low, save_high=args.save_high,
                                     shared_scale=args.shared_scale):
                    success += 1
        except KeyboardInterrupt:
            print("\n\nInterrupted by user (Ctrl+C)")
            interrupted = False
        
        print(f"\n{'=' * 50}")
        if interrupted:
            print(f"Stopped! {success}/{len(slide_names)} slides processed before interruption")
        else:
            print(f"Done! {success}/{len(slide_names)} slides processed")


if __name__ == '__main__':
    main()
