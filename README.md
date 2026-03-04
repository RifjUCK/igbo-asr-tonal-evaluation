# Igbo Blind Spot Dataset for omniASR-CTC-1B

[![Dataset](https://img.shields.io/badge/🤗%20Dataset-omniASR--igbo--blindspots-blue)](https://huggingface.co/datasets/chiz/omniASR-igbo-blindspots)
[![License](https://img.shields.io/badge/License-CC--BY--4.0-green.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)

Systematic evaluation of tonal fidelity in facebook/omniASR-CTC-1B when processing Igbo, a tonal Niger-Congo language with ~45 million speakers.

## 🔍 Overview

This project reveals systematic tonal diacritic loss in a state-of-the-art multilingual ASR model:
- **75.5% diacritic loss** on tonal markers (bootstrap 95% CI: [57.1%, 89.7%])
- **Minimal pair collapse**: Model cannot distinguish phonemically contrastive tones
- **Orthographic bias**: Model hallucinates tone marks on monotone speech

**Key Insight:** The model appears to generate diacritics probabilistically based on lexical priors rather than acoustic conditioning.

## 📊 Dataset

**21 audio samples** across 4 error categories:
1. Cross-lingual Orthographic Interference (5 samples)
2. Phonemic Tone Sensitivity (6 samples)
3. Language Boundary Effects (5 samples)
4. Domain-Specific Lexical Coverage (5 samples)

🔗 **[View Dataset on HuggingFace](https://huggingface.co/datasets/chiz/omniASR-igbo-blindspots)**

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/[YOUR_USERNAME]/omniASR-igbo-blindspots.git
cd omniASR-igbo-blindspots
pip install -r requirements.txt
```

### Run Analysis

```bash
jupyter notebook analysis.ipynb
```

Or open in Google Colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/[YOUR_USERNAME]/omniASR-igbo-blindspots/blob/main/analysis.ipynb)

## 📁 Repository Structure

```
omniASR-igbo-blindspots/
├── README.md                    # This file
├── analysis.ipynb               # Full analysis notebook
├── requirements.txt             # Python dependencies
├── data/
│   ├── audio/                   # 21 WAV files (16kHz mono)
│   ├── metadata.csv             # Ground truth, model outputs, metrics
│   └── visualizations/          # Generated figures
├── src/
│   ├── evaluate.py              # Evaluation metrics (DER, bootstrap CIs)
│   ├── visualize.py             # Plotting functions
│   └── utils.py                 # Helper functions
└── docs/
    └── METHODOLOGY.md           # Detailed methodology
```

## 📈 Key Results

### Quantitative Summary

| Category | Samples | Diacritic Loss | Avg CER |
|----------|---------|----------------|---------|
| **Phonemic Tone Sensitivity** | 6 | **75.5%** | 50.6% |
| Cross-lingual Interference | 5 | -38.9% (hallucination) | 28.8% |
| Domain-Specific Coverage | 5 | 6.3% | 30.1% |
| Language Boundary Effects | 5 | 14.3% | 20.0% |
| **Overall** | **21** | **26.8%** | **32.5%** |

### Bootstrap Confidence Intervals

- **Tonal category:** 75.5% (95% CI: [57.1%, 89.7%])
- **Overall:** 52.6% (95% CI: [30.3%, 69.7%])

Even the worst-case lower bound (57.1%) indicates severe tonal degradation.

## 🎯 Example: Tonal Minimal Pairs

**Input:** "akwa, akwa, akwa. Akwà, akwà, akwà. Àkwà, àkwà, àkwà. Ákwá, ákwá, ákwá."  
(4 distinct Igbo words with different meanings)

**Model Output:** "akua akua akua akua akwa akwa akwa akua akwa ọkua ọkua ọkua"  
(Random variations, semantic distinctions lost)

**Impact:** 
- akwà (cloth) → akwa (could mean "crying")
- àkwà (egg) → akwa (meaning lost)
- ákwá (bridge) → akua (wrong word)

## 🔬 Methodology

### Model Evaluated
- **Model:** [facebook/omniASR-CTC-1B](https://huggingface.co/facebook/omniASR-CTC-1B)
- **Parameters:** 975M
- **Architecture:** CTC-based ASR (wav2vec2-style)
- **Languages:** 1,600+ (including Igbo)

### Recording Details
- **Speaker:** Native Igbo speaker (Afikpo dialect, Ebonyi State)
- **Device:** iPhone SE 2nd Generation
- **Format:** 16kHz mono WAV (converted from m4a)
- **Duration:** 4-15 seconds per sample

### Metrics
- **DER (Diacritic Error Rate):** Captures dropped + hallucinated tone marks
- **Bootstrap CIs:** 10,000 iterations at utterance level
- **CER (Character Error Rate):** Standard transcription accuracy

## 📚 Citation

If you use this dataset or code, please cite:

```bibtex
@misc{obasi2026igbo,
  title={Igbo Blind Spot Dataset for omniASR-CTC-1B: Systematic Evaluation of Tonal Diacritic Loss},
  author={Obasi, Chizoba},
  year={2026},
  publisher={HuggingFace},
  howpublished={\url{https://huggingface.co/datasets/chiz/omniASR-igbo-blindspots}},
  note={Model evaluated: facebook/omniASR-CTC-1B (975M parameters)}
}
```

## 🔗 Related Work

- **Dataset:** [HuggingFace Hub](https://huggingface.co/datasets/chiz/omniASR-igbo-blindspots)
- **Model:** [omniASR-CTC-1B](https://huggingface.co/facebook/omniASR-CTC-1B)
- **Paper:** [Meta AI - Omnilingual ASR (arXiv:2511.09690)](https://arxiv.org/abs/2511.09690)

## 🛠️ Future Work

1. **Scale to multi-speaker evaluation** (10+ speakers across dialects)
2. **Comparative model audit** (Whisper, MMS, USM, Azure Speech)
3. **Fine-tuning intervention** with tone-annotated data
4. **Downstream impact studies** in voice assistants

## 📄 License

- **Audio recordings:** CC-BY-4.0 (attribution required)
- **Metadata/annotations:** CC0 (public domain)
- **Code:** MIT License

## 👤 Author

**Chizoba Obasi**  
🔗 [HuggingFace](https://huggingface.co/chiz) | 🌐 [GitHub](https://github.com/[YOUR_USERNAME])

---

*This project demonstrates systematic evaluation of ML model blind spots using native speaker expertise.*
