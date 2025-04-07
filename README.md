# CycleDiffusion

**CycleDiffusion** is a novel *cycle-consistent diffusion model* designed for **non-parallel, many-to-many voice conversion (VC)**. By simultaneously learning both **reconstruction and conversion paths**, CycleDiffusion outperforms conventional VAE- and GAN-based VC models in **speaker similarity**, **linguistic preservation**, and **speech naturalness**.

---

## 📝 Overview

> **CycleDiffusion: Voice Conversion Using Cycle-Consistent Diffusion Models**  
> Dongsuk Yook, Geonhee Han, Hyung-Pil Chang, and In-Chul Yoo  
> [*Applied Sciences, 2024*](https://doi.org/10.3390/app14209595)

Voice conversion modifies a source speaker's voice to sound like that of a target speaker while preserving linguistic content. Diffusion Models (DMs) are promising due to their training stability and high-quality output. However, traditional DMs primarily learn reconstruction paths, limiting conversion quality. CycleDiffusion addresses this by:

- Learning both **reconstruction and conversion paths**
- Enforcing **cycle consistency loss** to preserve linguistic content
- Operating without **parallel training data**

---

## 🧠 Architecture

<p align="center">
  <img width="552" alt="Image" src="https://github.com/user-attachments/assets/d71929d9-e3d4-4e11-b205-f49208fa680b" />
</p>

---

## 📁 Directory Structure and Usage
- `VCTK_2F2M/`: Subset of the VCTK dataset with 2 female and 2 male speakers.
  - Contains subfolders like `wavs/`, `mels/`, `embeds/`, `textgrids/`, and `txt/`.
- `converted_all/`: All voice conversion output files.
  - `CycleDiffusion_Inter_gender/`, `CycleDiffusion_Intra_gender/`: Final outputs for CycleDiffusion.
  - `DiffVC_Inter_gender/`, `DiffVC_Intra_gender/`: Baseline outputs from DiffVC.
- `model/`: Main architecture files including diffusion and cycle-consistent modules.
- `make_converted_wav.py`: Inference script to generate converted waveforms from trained checkpoints.
- `real_last_cycle_train_dec_4speakers_*.py`: Training scripts for CycleDiffusion with different cycle consistency configurations.
- `logs_enc_2speakers/`: Speaker encoder trained on 2 speakers.
- `get_mels_embeds_HEE.py`: Script to extract mel-spectrograms and speaker embeddings.
- `get_textgrids.py`: Generates forced-alignment textgrid files.
- `calculate/`: Contains MCD evaluation scripts and output.
  - `cal_pymcd.py`: Computes MCD using DTW.
  - `make_json.py`: Generates a default JSON template for storing evaluation results.
- `tree_files.py`: Utility for generating markdown descriptions of directory structures.

## ⚙️ Training & Evaluation Setup
- **Dataset**: VCTK (p236, p239, p259, p263) — 471 training utterances and 10 test utterances per speaker.
  - Test Sentence Numbers: 2, 3, 4, 5, 6, 7, 9, 10, 11, 12
  - Speaker Information
  - p239: 502 files (female) → VCTK_F1: 471 files
  - p236: 492 files (female) → VCTK_F2: 471 files
  - p259: 481 files (male) → VCTK_M1: 471 files
  - p263: 471 files (male) → VCTK_M2: 471 files
  
- **Epochs**: Training was conducted for up to 300 epochs. Evaluation was performed every 10 epochs, and the model checkpoint with the best MCD score was selected.
- **Diffusion Steps**:
  - Default: 5 (Used during cycle inference for most experiments)
  - Final Experiment: 6
- **Cycle Batch Sample Count (iii)**:
  - Default: 2 (During cycle consistency loss calculation, only 2 samples out of each batch of 4 were used due to VRAM limitations.)
  - Final Experiment: **iii = 3** and **diffusion step = 6** were applied using a high-memory setup with **24GB VRAM**.
- **Best Model Script**:  
  `real_last_cycle_train_dec_4speakers_iii3_cycle6_from_50.py`

## 🚩 Highlights

- 🌀 **Cycle-Consistent Diffusion Architecture**  
  Diffusion model trained to reconstruct and convert speech.

- 🔁 **Cycle Consistency Loss**  
  Minimizes the difference between the original and cyclically reconstructed speech, ensuring linguistic preservation.

- 🔗 **Non-parallel Training**  
  No need for aligned utterances across speakers.

- 📊 **Robust Performance**  
  Outperforms baseline DiffVC in cosine similarity, ASR accuracy, Mel-Cepstral Distance (MCD), and MOS.

---

## 📊 Experimental Results (vs. DiffVC)

| Metric | DiffVC | CycleDiffusion | Improvement |
|--------|--------|----------------|-------------|
| **Cosine Similarity (↑)** | 0.6880 | **0.7223** | +5.0% |
| **ASR Accuracy (↑)** | 71.3% | **74.4%** | +4.4% |
| **Mel-Cepstral Distance (↓)** | 5.90 | **5.09** | -15.9% |
| **MOS Score** | 3.50 | **3.70** | +5.7% |

---

### 🧪 Ablation Study Summary

| Condition                                           | MCD Score |
|----------------------------------------------------|-----------|
| Cycle from epoch 0 (a→b→a, full backward)           | 5.59      |
| Cycle from epoch 50 (a→b→a, full backward)          | 5.72      |
| Cycle from epoch 100 (a→b→a, full backward)         | 5.58      |
| Cycle from epoch 50 (only b→a backprop, iii=3)      | **5.09** ✅ |
| DiffVC with speaker encoder                         | 5.90      |
| DiffVC with one-hot speaker vector                  | 6.34      |

---

## 📖 Citation

If you find this project helpful, please cite our paper:

```bibtex
@Article{app14209595,
AUTHOR = {Yook, Dongsuk and Han, Geonhee and Chang, Hyung-Pil and Yoo, In-Chul},
TITLE = {CycleDiffusion: Voice Conversion Using Cycle-Consistent Diffusion Models},
JOURNAL = {Applied Sciences},
VOLUME = {14},
YEAR = {2024},
NUMBER = {20},
ARTICLE-NUMBER = {9595},
URL = {https://www.mdpi.com/2076-3417/14/20/9595},
ISSN = {2076-3417},
DOI = {10.3390/app14209595}
}
```

## 🙏 Acknowledgements

This work builds upon and was inspired by:

- **[DiffVC](https://arxiv.org/abs/2109.13821)** – A diffusion-based voice conversion baseline that provided strong foundations for this work.
- **[MaskCycleGAN-VC](https://arxiv.org/abs/2102.12841)** – Prior works that introduced the concept of cycle consistency in voice conversion.
- **[HiFi-GAN](https://arxiv.org/abs/2010.05646)** – Used as the vocoder for waveform reconstruction in our experiments.

We sincerely thank the authors and contributors of these projects for making their work openly available. Their contributions were invaluable in the development and evaluation of CycleDiffusion.
