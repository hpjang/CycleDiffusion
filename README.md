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

## 📁 Directory Structure
The following is an overview of the project directory and major files:

VCTK_2F2M/: Dataset folder containing 2 male and 2 female speakers.

converted_all/: All generated voice conversion outputs.

CycleDiffusion_Inter_gender/, CycleDiffusion_Intra_gender/: Final results used in the paper (converted samples).

DiffVC_Inter_gender/, DiffVC_Intra_gender/: Baseline DiffVC outputs for comparison.

model/: Main model architecture code.

logs_enc_2speakers/: Encoder trained with 2 speakers.

make_converted_wav.py: Inference script to generate converted waveforms.

real_last_cycle_train_dec_4speakers_*.py: Training scripts with various cycle consistency configurations.

tree_files.py: Generates a markdown structure of the codebase.

calculate/: Scripts and result files for MCD evaluation.

cal_pymcd.py: MCD measurement script.

make_json.py: Creates default result JSON structure.

get_mels_embeds_HEE.py: Generates mel-spectrogram and speaker embeddings.

get_textgrids.py: Generates textgrid files for alignment.

---

##⚙️ Training & Evaluation Setup
Dataset: Subset of the VCTK corpus with 4 speakers (2F, 2M), each providing 471 training and 10 test utterances.

Training Epochs: Up to 300, with evaluation every 10 epochs.

Diffusion Steps: 5 (for cycle inference), 6 in final experiments.

Cycle Consistency Setting: iii = 2 for most experiments (batch subset used due to VRAM limits), iii = 3 in final version.

Best Model: real_last_cycle_train_dec_4speakers_iii3_cycle6_from_50.py

---

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
