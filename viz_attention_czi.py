"""
ViLa-MIL Attention Visualization Script for CZI files (SR386 dataset)

Uses PyTorch hooks to extract attention weights without modifying model code.

Usage:
    python viz_attention_czi.py \
        --ckpt_path sr0.0002,0.0001_cox/sr386_conch_aux0_aux_label_1_aux_label_2_p16_step_s0/s_0_checkpoint.pt \
        --h5_dir /home/hzxu/remote_data/sr386_feat/10featsr/h5_files/ \
        --h5_dir_l /home/hzxu/remote_data/sr386_feat/20featsr/h5_files/ \
        --wsi_dir /home/hzxu/remote_data/sr386/ \
        --output_dir attention_results_sr386 \
        --slide_ids SR386_40X_HE_T003_01 \
        --wsi_ext czi

Output files:
    {slide_name}_wsi.png          - WSI thumbnail (no attention)
    {slide_name}_heatmap_low.png  - Low-res attention heatmap (transparent background)
    {slide_name}_heatmap_high.png - High-res attention heatmap (transparent background)

Requirements:
    pip install aicsimageio czifile
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


class CZIReader:
    """CZI file reader using aicspylibczi (supports scale_factor reading).

    Key point: aicspylibczi 的包名和导入名都是 'aicspylibczi'（不是 pylibczi）。
    对于 mosaic WSI 文件，使用 read_mosaic(scale_factor=...) 只读取下采样后的
    拼接图像，不会把全分辨率图读进内存。
    """

    def __init__(self, czi_path):
        self.czi_path = czi_path
        self._czi = None
        self._base_w = 0
        self._base_h = 0
        self._is_mosaic = False
        self._dims = ''
        self._init_reader()

    def _init_reader(self):
        import sys

        # --- 使用 aicspylibczi（pip install aicspylibczi，导入名同为 aicspylibczi）---
        try:
            from aicspylibczi import CziFile
        except ImportError:
            raise ImportError(
                "Cannot read CZI: aicspylibczi not installed. "
                "Please run: pip install aicspylibczi"
            )

        print(f"    [aicspylibczi] opening CZI: {os.path.basename(self.czi_path)}")
        sys.stdout.flush()
        self._czi = CziFile(self.czi_path)

        # 判断是否为 mosaic 文件
        try:
            self._is_mosaic = self._czi.is_mosaic()
            print(f"    [aicspylibczi] is_mosaic={self._is_mosaic}")
        except Exception:
            self._is_mosaic = False
        sys.stdout.flush()

        # 获取维度信息
        try:
            self._dims = str(self._czi.dims)
            print(f"    [aicspylibczi] dims={self._dims}")
        except Exception:
            self._dims = ''
        sys.stdout.flush()

        # 获取尺寸（base resolution）
        # 对于 mosaic 文件，get_dims_shape() 返回的是单个 tile 的尺寸，
        # 需要用 get_mosaic_bounding_box() 获取全图尺寸
        if self._is_mosaic:
            self._base_w, self._base_h = self._get_mosaic_size()
        else:
            self._base_w, self._base_h = self._get_xy_size_from_shape()
        if self._base_w == 0 or self._base_h == 0:
            raise RuntimeError("Cannot determine CZI X/Y dimensions")

        print(f"    [aicspylibczi] base resolution: {self._base_w}x{self._base_h}")
        sys.stdout.flush()

    def _get_mosaic_size(self):
        """获取 mosaic 文件的全图尺寸（可能远大于单个 tile）。"""
        # 优先用 get_mosaic_bounding_box()（aicspylibczi API）
        try:
            bbox = self._czi.get_mosaic_bounding_box()
            # bbox 可能是 BBox 对象或 tuple
            if hasattr(bbox, 'w'):
                return int(bbox.w), int(bbox.h)
            elif hasattr(bbox, 'x') and hasattr(bbox, 'y'):
                # bbox = BBox(x, y, w, h)
                return int(bbox.w), int(bbox.h)
            elif isinstance(bbox, (tuple, list)) and len(bbox) >= 4:
                return int(bbox[2]), int(bbox[3])
            elif isinstance(bbox, dict):
                return int(bbox.get('w', 0)), int(bbox.get('h', 0))
        except AttributeError:
            pass
        except Exception as e:
            print(f"    [aicspylibczi] get_mosaic_bounding_box failed: {e}")

        # 备选：read_mosaic(C=0, scale_factor=1) 拿全图后看 shape
        # 但这很慢，尽量不用
        # 再备选：从 dims_shape 中 M 维度推算
        try:
            dims_shape = self._czi.get_dims_shape()
            if isinstance(dims_shape, list) and len(dims_shape) > 0:
                d = dims_shape[0]
                if isinstance(d, dict):
                    x_info = d.get('X', (0, 0))
                    y_info = d.get('Y', (0, 0))
                    m_info = d.get('M', (0, 1))
                    n_tiles = m_info[1] if m_info[1] > 1 else 1
                    # 粗估：单 tile 大小 * sqrt(n_tiles)，不精确但作为 fallback
                    tile_w, tile_h = x_info[1], y_info[1]
                    import math
                    scale_approx = math.sqrt(n_tiles)
                    return int(tile_w * scale_approx), int(tile_h * scale_approx)
        except Exception:
            pass

        return 0, 0

    def _get_xy_size_from_shape(self):
        """从 get_dims_shape() 获取 X, Y 的尺寸."""
        try:
            dims_shape = self._czi.get_dims_shape()
            # dims_shape 是 list of dict，形如 [{'X': (0, 1900), 'Y': (0, 1300), ...}]
            if isinstance(dims_shape, list) and len(dims_shape) > 0:
                d = dims_shape[0]
                if isinstance(d, dict):
                    x_info = d.get('X', (0, 0))
                    y_info = d.get('Y', (0, 0))
                    return int(x_info[1]), int(y_info[1])
        except Exception as e:
            print(f"    [aicspylibczi] get_dims_shape failed: {e}")

        # 备选：使用 size 属性 + dims
        try:
            size = self._czi.size  # tuple，对应 self._czi.dims 的各维度大小
            dims = self._czi.dims
            w = h = 0
            for i, dim_char in enumerate(dims):
                if dim_char == 'X':
                    w = int(size[i])
                elif dim_char == 'Y':
                    h = int(size[i])
            if w > 0 and h > 0:
                return w, h
        except Exception:
            pass

        return 0, 0

    # ---- 金字塔层级（CZI 无原生金字塔，按尺寸自动构造虚拟层级）----

    @property
    def level_count(self):
        import math
        max_dim = max(self._base_w, self._base_h)
        n = max(1, int(math.log2(max(2, max_dim / 512))) + 1)
        return min(n, 10)

    @property
    def level_downsamples(self):
        return tuple(2.0 ** i for i in range(self.level_count))

    @property
    def level_dimensions(self):
        return tuple(
            (max(1, self._base_w // (2 ** i)), max(1, self._base_h // (2 ** i)))
            for i in range(self.level_count)
        )

    @property
    def dimensions(self):
        return (self._base_w, self._base_h)

    # ---- 区域读取 ----

    def read_region(self, location, level, size):
        """Read a region from the CZI at the specified zoom level.

        对于 mosaic WSI 文件，使用 read_mosaic(scale_factor=...) 直接读取
        下采样后的整图，然后裁剪到请求的区域。
        """
        x, y = location
        w, h = size
        level = max(0, min(level, self.level_count - 1))

        print(f"    [aicspylibczi] read_region: loc=({x},{y}) level={level} size={w}x{h}")
        import sys
        sys.stdout.flush()

        try:
            # 计算 scale_factor（CZI mosaic 用 scale_factor 而非 level）
            scale_factor = 1.0 / (2 ** level)

            if self._is_mosaic:
                # mosaic 文件：read_mosaic 需要 C 参数指定通道
                print(f"    [aicspylibczi] read_mosaic(C=0, scale_factor={scale_factor:.4f})...")
                sys.stdout.flush()
                img = self._czi.read_mosaic(C=0, scale_factor=scale_factor)
                # img shape: (T, Z, C, Y, X) 或类似，squeeze 掉前导 1
                img = self._normalize_to_hwc_uint8(img)
                print(f"    [aicspylibczi] mosaic image shape: {img.shape}")
                sys.stdout.flush()
            else:
                # 非 mosaic：直接 read_image，需指定 C=0
                print(f"    [aicspylibczi] read_image(C=0)...")
                sys.stdout.flush()
                try:
                    img, _ = self._czi.read_image(C=0)
                except (TypeError, Exception):
                    img, _ = self._czi.read_image()
                img = self._normalize_to_hwc_uint8(img)

            # 在下采样后的图上裁剪请求区域
            # location 是 level 0 坐标，需转换为当前 level 坐标
            x_ds = int(x * scale_factor)
            y_ds = int(y * scale_factor)
            w_ds = int(w * scale_factor) if scale_factor < 1 else w
            h_ds = int(h * scale_factor) if scale_factor < 1 else h

            img_h, img_w = img.shape[:2]
            x1, y1 = max(0, x_ds), max(0, y_ds)
            x2, y2 = min(img_w, x1 + w_ds), min(img_h, y1 + h_ds)

            if x1 >= x2 or y1 >= y2:
                # 越界，返回灰色
                return Image.new('RGB', (w, h), (128, 128, 128))

            region = img[y1:y2, x1:x2]

            # 缩放到请求的 size（read_mosaic 返回的实际尺寸可能与请求不完全一致）
            if region.shape[1] != w or region.shape[0] != h:
                region = np.array(
                    Image.fromarray(region).resize((w, h), Image.LANCZOS)
                )

            return Image.fromarray(region)

        except Exception as e:
            print(f"    [aicspylibczi] read_region failed: {e}")
            import traceback
            traceback.print_exc()
            return Image.new('RGB', (w, h), (128, 128, 128))

    @staticmethod
    def _normalize_to_hwc_uint8(arr):
        """将任意维度的 numpy 数组转为 (H, W, 3) uint8."""
        arr = np.squeeze(arr)

        # 去掉 T, Z, S, B 等多余维度
        while arr.ndim > 3:
            arr = arr[0]

        if arr.ndim == 2:  # 灰度
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3:
            # CHW -> HWC（aicspylibczi 返回的通常是 ...C,Y,X）
            if arr.shape[0] <= 4 and arr.shape[0] < arr.shape[-1]:
                arr = np.transpose(arr, (1, 2, 0))
            # 去掉 alpha
            if arr.shape[-1] == 4:
                arr = arr[..., :3]
            elif arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)

        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = (arr * 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)

        return arr

    def close(self):
        # aicspylibczi 的 CziFile 没有 close 方法，文件句柄由 GC 管理
        pass


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
    """Save WSI thumbnail without attention overlay using CZI reader.

    对于 mosaic CZI 文件，直接用 read_mosaic(scale_factor=...) 按 max_size 读取，
    不走 level 系统（避免读取过大的中间图像导致 OOM）。
    """
    wsi = CZIReader(wsi_path)
    base_w, base_h = wsi.dimensions
    is_mosaic = wsi._is_mosaic

    if is_mosaic:
        # mosaic: 直接按 max_size 计算 scale_factor，一次性读取
        scale = min(max_size / max(base_w, base_h), 1.0)
        out_width = int(base_w * scale)
        out_height = int(base_h * scale)
        print(f"  Saving WSI thumbnail: mosaic, base={base_w}x{base_h}, "
              f"scale_factor={scale:.6f}, output={out_width}x{out_height}")
        import sys
        sys.stdout.flush()
        img = wsi._czi.read_mosaic(C=0, scale_factor=scale)
        img = CZIReader._normalize_to_hwc_uint8(img)
        wsi_img = Image.fromarray(img).convert('RGB')
        # 确保尺寸精确
        if wsi_img.size != (out_width, out_height):
            wsi_img = wsi_img.resize((out_width, out_height), Image.LANCZOS)
    else:
        # 非 mosaic: 走 level 系统
        if vis_level < 0:
            vis_level = wsi.level_count - 1
        vis_level = min(vis_level, wsi.level_count - 1)
        level_dims = wsi.level_dimensions
        if vis_level >= len(level_dims):
            vis_level = len(level_dims) - 1
        wsi_width, wsi_height = level_dims[vis_level]
        scale = min(max_size / max(wsi_width, wsi_height), 1.0)
        out_width = int(wsi_width * scale)
        out_height = int(wsi_height * scale)
        print(f"  Saving WSI thumbnail: level={vis_level}/{wsi.level_count-1}, "
              f"region={wsi_width}x{wsi_height}, output={out_width}x{out_height}")
        wsi_img = wsi.read_region((0, 0), vis_level, (wsi_width, wsi_height))
        wsi_img = wsi_img.convert('RGB')
        wsi_img = wsi_img.resize((out_width, out_height), Image.NEAREST)

    # 将接近黑色的背景像素设为透明（CZI mosaic 空白区域为 0,0,0）
    # 保存为 RGBA PNG
    wsi_rgba = wsi_img.convert('RGBA')
    arr = np.array(wsi_rgba)
    # 判定背景：R+G+B 都很低的像素（阈值 30，避免误伤深色组织）
    is_bg = (arr[:, :, 0].astype(np.int32) + arr[:, :, 1].astype(np.int32) + arr[:, :, 2].astype(np.int32)) < 30
    arr[is_bg, 3] = 0  # alpha=0 透明
    wsi_final = Image.fromarray(arr, 'RGBA')
    wsi_final.save(output_path)
    print(f"  Saved (transparent bg, {is_bg.sum()} px): {output_path}")
    wsi.close()


def draw_heatmap_only(scores, coords, wsi_path, output_path,
                      vis_level=2, alpha=0.4, cmap='jet',
                      patch_size=256, max_size=2048):
    """Draw attention heatmap only (transparent background, no WSI overlay) using CZI reader.

    对于 mosaic CZI 文件，直接基于 base resolution 做坐标映射，
    不走 level 系统。coords 假设为 base resolution 坐标。
    """
    import matplotlib.pyplot as plt

    scores_flat = scores.flatten()
    if scores_flat.max() > scores_flat.min():
        scores_norm = (scores_flat - scores_flat.min()) / (scores_flat.max() - scores_flat.min())
    else:
        scores_norm = np.ones_like(scores_flat) * 0.5

    wsi = CZIReader(wsi_path)
    base_w, base_h = wsi.dimensions
    is_mosaic = wsi._is_mosaic

    if is_mosaic:
        # mosaic: 直接基于 base resolution 映射
        scale = min(max_size / max(base_w, base_h), 1.0)
        out_width = int(base_w * scale)
        out_height = int(base_h * scale)
        downsample = 1.0  # base resolution，无 downsample
    else:
        # 非 mosaic: 走 level 系统
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

    # coords -> output 坐标映射
    # coords 在 base resolution，需映射到 output 分辨率
    # output_coord = coord / downsample * scale
    # 对于 mosaic: downsample=1, output_coord = coord * scale = coord * max_size / base_w
    coords_f = coords.astype(np.float32)
    coords_scaled = (coords_f / downsample * scale).astype(np.int32)

    # vis_patch_size: 参考 vis_attention.py，用坐标间的实际步长而不是固定 patch_size
    # 这样可以避免 patches 之间出现间隔（当步长 > patch_size 时）
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
        print(f"  Coords step (original): x={x_step_original:.1f}, y={y_step_original:.1f}, using={step_size_original:.1f}")
    else:
        vis_patch_size = max(1, int(patch_size / downsample * scale))

    # 调试: 打印坐标范围
    if len(coords_scaled) > 0:
        x_min, x_max = coords_scaled[:, 0].min(), coords_scaled[:, 0].max()
        y_min, y_max = coords_scaled[:, 1].min(), coords_scaled[:, 1].max()
        print(f"  Coords range: x=[{x_min},{x_max}], y=[{y_min},{y_max}], "
              f"output=[{out_width},{out_height}], vis_patch_size={vis_patch_size}")

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
            heatmap[y1:y2, x1:x2, 3] = alpha
            drawn_count += 1

    Image.fromarray((heatmap * 255).astype(np.uint8)).save(output_path)

    print(f"  Heatmap: {drawn_count}/{len(coords_scaled)} patches drawn, saved to {output_path}")
    wsi.close()


def process_single_wsi(model, slide_name, h5_dir, h5_dir_l, wsi_dir, output_dir, wsi_ext='czi', vis_level=2, max_size=2048,
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
    print(f"  Patch sizes: low-res={patch_size_s} (level={patch_level_s}), high-res={patch_size_l} (level={patch_level_l})")
    print(f"  Patches: low-res={features_s.shape[0]}, high-res={features_l.shape[0]}")

    attn_low, attn_high, coords_s, coords_l = get_attention_with_hook(model, features_s, coords_s, features_l, coords_l)

    if save_low and attn_low is not None:
        attn = attn_low.squeeze(0).squeeze(0)
        attn_scores = attn.mean(dim=0).cpu().numpy()

        output_path = os.path.join(output_dir, f'{slide_name}_heatmap_low.png')
        draw_heatmap_only(attn_scores.reshape(-1, 1), coords_s.cpu().numpy(), wsi_path, output_path,
                          vis_level=vis_level, max_size=max_size,
                          patch_size=patch_size_s)
    elif save_low:
        print("Warning: No low-resolution attention weights captured")

    if save_high and attn_high is not None:
        attn = attn_high.squeeze(0).squeeze(0)
        attn_scores = attn.mean(dim=0).cpu().numpy()

        output_path = os.path.join(output_dir, f'{slide_name}_heatmap_high.png')
        draw_heatmap_only(attn_scores.reshape(-1, 1), coords_l.cpu().numpy(), wsi_path, output_path,
                          vis_level=vis_level, max_size=max_size,
                          patch_size=patch_size_l)
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
    parser = argparse.ArgumentParser(description='ViLa-MIL Attention Visualization for CZI files (SR386)')
    parser.add_argument('--ckpt_path', type=str, default=None, help='Single checkpoint path')
    parser.add_argument('--batch_path', type=str, default=None, help='Batch path containing multiple checkpoints and splits')
    parser.add_argument('--model_type', type=str, default='ViLa_MTL_MIL')
    parser.add_argument('--h5_dir', type=str, required=True, help='Low-res H5 dir')
    parser.add_argument('--h5_dir_l', type=str, default=None, help='High-res H5 dir')
    parser.add_argument('--wsi_dir', type=str, required=True)
    parser.add_argument('--wsi_ext', type=str, default='czi')
    parser.add_argument('--output_dir', type=str, default='attention_results_sr386')
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
