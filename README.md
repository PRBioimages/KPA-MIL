# KPA-MIL

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

### KPA-MIL: Knowledge-Prompted Auxiliary-task Multiple Instance Learning for Whole Slide Image Analysis

KPA-MIL is a multi-task multiple instance learning (MIL) framework for whole slide image (WSI) analysis. The main task never consumes text directly; instead, clinical knowledge is injected **indirectly** through text-prompt-guided auxiliary tasks, where each auxiliary label is paired with a natural-language prompt encoded by a pathology vision-language model (CONCH). A cross-scale contrastive objective further aligns global features aggregated from the low- and high-resolution branches. KPA-MIL was validated on 5 real-world pathology datasets (TJ, CRC-OS, CRC-PFS, GMU, SR386), supporting both classification and survival analysis paradigms.

🔬 Key Features:

- Text-prompt-guided auxiliary tasks: clinical knowledge enters the model only through auxiliary heads, keeping the main task architecture text-free
- Dual-scale MIL aggregation with prototype-based attention pooling (built on ViLa-MIL)
- Cross-scale contrastive learning (NT-Xent) between low- and high-resolution branch global features
- Three main-task heads: binary classification, Cox proportional hazards, and Mixture Density Network (MDN) survival distribution prediction
- Flexible auxiliary-task configuration: arbitrary label columns with per-task loss types (CE / Focal / Cox)
- Attention visualization: text-image attention, prototype attention, and baseline attention heatmaps

📦 Applications: cancer survival prediction (overall / progression-free survival), liver-transplant rejection classification, and interpretable attention analysis on WSIs.

## Project Structure

```
KPA-MIL/
├── main_MTL.py                  # Main training entry point (multi-task)
├── run_experiments.sh           # All 9 experiment configurations used in the paper
├── create_patches_fp.py         # WSI patch extraction (CLAM pipeline)
├── create_splits_seq_MTL.py     # 5-fold cross-validation split generation
├── vis_attention.py             # Visualization: baseline (no text) attention
├── vis_attention_text.py        # Visualization: text-image cross attention
├── viz_attention_czi.py         # Visualization: prototype attention (supports CZI)
│
├── models/
│   ├── model_ViLa_MIL_MTL.py    # Main multi-task model
│   ├── model_ViLa_MIL.py        # Single-task baseline
│   ├── mil_aggregators.py       # MIL aggregators (prototype / attention pooling)
│   └── mdn.py                   # Mixture Density Network head
│
├── utils/
│   ├── core_utils.py            # Training loop, validation, early stopping
│   ├── survival_metrics.py      # C-index, time-dependent concordance, Brier score
│   └── loss_utils.py            # Losses (CE, Focal, Cox NLL, MDN NLL)
│
├── datasets/
│   └── dataset_generic.py       # Generic MIL / MTL dataset loaders
│
├── conch/                       # CONCH text encoder (official code)
│   └── open_clip_custom/        # Text encoder + tokenizer
│
├── dataset_csv/                 # Labels for the 5 datasets
├── text_prompt/                 # Text prompts per dataset (tj/, crc_outcome/, gmu/, surgen/)
├── splits/                      # Pre-generated 5-fold splits (e.g., MTLtj_100/, crcos_100/)
├── wsi_core/                    # WSI I/O and patching utilities (CLAM official code)
└── feature_extraction/          # Feature extraction utilities (ResNet / ViT encoders)
```

### Third-party Resources

The following external resources are used in this project:

- **CONCH** (`conch/`): pathology vision-language model from [mahmoodlab/CONCH](https://github.com/mahmoodlab/CONCH)
- **ViLa-MIL** (`models/`): dual-scale MIL backbone adapted from [JiangZhengran/ViLa-MIL](https://github.com/JiangZhengran/ViLa-MIL)
- **CLAM** (`wsi_core/`, `create_patches_fp.py`): WSI processing pipeline from [mahmoodlab/CLAM](https://github.com/mahmoodlab/CLAM)

**Key dependencies:**
- Python 3.10+
- PyTorch (CUDA)
- openslide-python, pandas, numpy, scikit-learn, lifelines, tensorboardX
- timm, ml_collections, h5py, opencv-python, pillow, matplotlib, seaborn

**Hardware:** NVIDIA GPU recommended (tested on a single GPU).

## Data

Five real-world pathology cohorts are used in this study:

| File | Dataset | Main task | Auxiliary labels |
|------|---------|-----------|------------------|
| `dataset_csv/tj.csv` | TJ (liver transplant rejection, classification) | binary | 3 |
| `dataset_csv/crc_outcomesOS.csv` | CRC-OS (colorectal cancer, overall survival) | cox / mdn | 2 |
| `dataset_csv/crc_outcomesPFS.csv` | CRC-PFS (colorectal cancer, progression-free survival) | cox / mdn | 2 |
| `dataset_csv/MTL182.csv` | GMU (colorectal cancer) | cox / mdn | 1 |
| `dataset_csv/sr386OS.csv` | SR386 (colorectal cancer, overall survival) | cox / mdn | 3 |

Text prompts for the auxiliary labels are stored under `text_prompt/`, one CSV per dataset. Each row corresponds to one slide (case), and each column provides the prompt of one auxiliary task.

## Training KPA-MIL

### 1. Environment

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install openslide-python pandas numpy scikit-learn lifelines tensorboardX timm ml_collections h5py opencv-python pillow matplotlib seaborn
```

### 2. Data Preparation

**Step 1: extract WSI patches (CLAM pipeline)**

```bash
python create_patches_fp.py \
    --source /path/to/wsi \
    --save_dir /path/to/patches \
    --patch_size 256 \
    --seg --patch --stitch
```

**Step 2: extract CONCH features for both resolution branches**

Extract features separately for the low-resolution branch (e.g., 10x/20x) and the high-resolution branch (e.g., 20x/40x) following the official CONCH feature extraction pipeline. Features are stored as one `.h5` file per slide.

**Step 3: generate 5-fold cross-validation splits**

```bash
python create_splits_seq_MTL.py \
    --csv_path dataset_csv/tj.csv \
    --output_dir splits/MTLtj_100 \
    --k 5
```

### 3. Training

Run the full set of paper configurations in `run_experiments.sh` (edit the path variables at the top first), or launch a single experiment:

```bash
# Example: TJ dataset (binary classification)
python main_MTL.py \
    --data_root_dir /path/to/tj_features/ \
    --data_folder_s 10feattj/h5_files/ \
    --data_folder_l 20feattj/h5_files/ \
    --model_type ViLa_MTL_MIL \
    --text_extractor conch \
    --task MTLtj \
    --text_prompt_file text_prompt/tj/tj_text_prompt_gpt_nopre.csv \
    --main_task binary \
    --aux_label_cols aux_label_0,aux_label_1,aux_label_2 \
    --aux_task_losses ce,ce,ce \
    --mil_aggregator prototype \
    --prototype_number 2 \
    --use_cross_scale_contrastive 1 \
    --cross_scale_weight 0.2 \
    --lr 0.0002 --reg 0.0001 \
    --scheduler_type step \
    --batch_size 4 \
    --record_loss \
    --results_dir ./results_tj \
    --exp_code tj_conch_aux3_p2_cross0.2_step
```

### 4. Experiment Configurations (Paper Results)

| # | Dataset | Main task | # Aux | Cross-scale weight | Prompt file |
|---|---------|-----------|-------|--------------------|-------------|
| 1 | CRC-OS | cox | 2 | — | `crc_text_prompt_gpt_nopre.csv` |
| 2 | CRC-OS | mdn | 2 | 0.7 | `crc_text_prompt_gpt_nopre.csv` |
| 3 | CRC-PFS | cox | 2 | — | `crc_text_prompt_gpt_nopre.csv` |
| 4 | CRC-PFS | mdn | 2 | 0.7 | `crc_text_prompt_gpt_nopre.csv` |
| 5 | GMU | cox | 1 | 0.5 | `gmu_text_prompt_gpt_nopre.csv` |
| 6 | GMU | mdn | 1 | — | `gmu_text_prompt_gpt_nopre.csv` |
| 7 | SR386 | cox | 3 | 0.2 | `sr386_text_prompt_gemini_nopre.csv` |
| 8 | SR386 | mdn | 3 | 0.7 | `sr386_text_prompt_gemini_nopre.csv` |
| 9 | TJ | binary | 3 | 0.2 | `tj_text_prompt_gpt_nopre.csv` |

> **Note:** the raw GMU CSV contains 5 auxiliary columns, but the paper only uses `aux_label_1` (LVI). The GMU prompt files have been reduced to the single retained column (`task0`).

After training, per-fold outputs are saved under `results_dir/exp_code_s0/`:

- `results_{fold}.csv` — test-set metrics
- `prediction_results_{fold}.csv` — per-sample predictions
- `loss_records_fold_{fold}.csv` — training loss records

### Key Parameters

| Parameter | Description | Values used in the paper |
|-----------|-------------|--------------------------|
| `--main_task` | Main task type | `binary` / `cox` / `mdn` |
| `--aux_label_cols` | Auxiliary label columns (comma-separated) | `aux_label_0,aux_label_1,...` |
| `--aux_task_losses` | Loss type per auxiliary task | `ce` / `focal` / `cox` |
| `--text_prompt_file` | Text prompt CSV path | per dataset |
| `--mil_aggregator` | MIL aggregator | `prototype` |
| `--prototype_number` | Number of prototypes | 2 / 12 / 16 / 24 |
| `--use_cross_scale_contrastive` | Cross-scale contrastive loss | 0 / 1 |
| `--cross_scale_weight` | Contrastive loss weight | 0.2 / 0.5 / 0.7 |
| `--early_stopping_metric` | Early stopping metric | `c_index` / `concordance_time_dependent` |
| `--mdn_mode` | MDN mode | `unified` |
| `--mdn_sigma_min` | Lower bound of MDN sigma | 0.01 |

## Visualization

Attention heatmaps can be generated after training:

| Script | Description |
|--------|-------------|
| `vis_attention_text.py` | Text-image cross attention: where the prompt attends on the slide |
| `viz_attention_czi.py` | Prototype attention heatmaps (supports CZI slides via `--wsi_ext czi`) |
| `vis_attention.py` | Baseline attention without text guidance |

```bash
python vis_attention_text.py \
    --batch_path results_tj/exp_code_s0/ \
    --h5_dir /path/to/features_10x/ \
    --h5_dir_l /path/to/features_20x/ \
    --wsi_dir /path/to/wsi/ \
    --output_dir attention_results_tj \
    --task_idx 0 \
    --shared_scale
```

Refer to each script's `argparse` definitions for available options.

## Citation

If you use this code, please cite our paper (currently under review):

<!-- TODO: update author list, journal, and bibliographic details once published -->

```bibtex
@article{kpamil2025,
  title={KPA-MIL: Knowledge-Prompted Auxiliary-task Multiple Instance Learning for Whole Slide Image Analysis},
  author={To be updated},
  journal={To be updated},
  year={2025},
  note={Under review}
}
```

This project builds upon the following works. Please also cite them:

```bibtex
@article{lu2024vision,
  title={A vision-language foundation model for computational pathology},
  author={Lu, Ming Y. and Chen, Bowen and Williamson, Drew F. K. and Chen, Richard J. and Zhao, Melissa and Chow, Aaron K. and Ikemura, Kenji and Kim, Ahrong and Pouli, Dimitra and Patel, Ankush and others},
  journal={Nature Medicine},
  volume={30},
  pages={863--874},
  year={2024}
}
```

```bibtex
@inproceedings{shi2024vila,
 title={{ViLa-MIL}: Dual-scale Vision-Language Multiple Instance Learning for Whole Slide Image Classification},
 author={Shi, Jiangbo and Li, Chen and Gong, Tieliang and Zheng, Yefeng and Fu, Huazhu},
 booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
 pages={11248--11258},
 year={2024}
}
```

## Contact

**Correspondence:** yyxu@smu.edu.cn

## License

MIT
