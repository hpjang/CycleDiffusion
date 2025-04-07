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
  <img src="image.png" alt="CycleDiffusion Architecture" width="700"/>
</p>

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
