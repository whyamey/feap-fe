# Fuzzy Extractors are Practical - Feature Extractor

This repository contains code for iris feature extraction and embedding generation for [FEAP](https://eprint.iacr.org/2024/100) using a multi-stage training pipeline. The system is designed to work with the IITD Iris Database.

## Pre-trained Models
Pre-trained model checkpoints are packaged in this GitHub repository's releases.

## Dataset Requirements
This project uses the IITD Iris Dataset, which you need to obtain separately due to licensing requirements. The dataset can be requested from [IIT Delhi Iris Database](http://www4.comp.polyu.edu.hk/~csajaykr/IITD/Database_Iris.htm).

### Dataset Segmentation
The iris images need to be segmented using [Hofbauer et al.](https://ieeexplore.ieee.org/document/6976811). The masks can be directly downloaded from [WaveLab's website](https://www.wavelab.at/sources/Hofbauer14b/). We provide a utility script in `scripts/` to help with segmentation.

### Dataset Structure
The dataset should be organized as follows:

#### Training & Inference Data
```
IITD_Segmented/
├── 001/
│   ├── 01_L.bmp
│   ├── 02_L.bmp
│   └── ...
├── 002/
├── 003/
└── ...
```

## Installation

### Requirements
Install the required packages:

```bash
pip install -r requirements.txt
```

## Installation

### Requirements
```
torch>=1.10.0
torchvision>=0.11.0
numpy>=1.19.2
opencv-python>=4.5.0
matplotlib>=3.3.0
tqdm>=4.50.0
```

Install the required packages:
```bash
pip install -r requirements.txt
```

## Training Pipeline

### Training from Scratch
The training process consists of three sequential stages:

1. **Stage 1: Base Model**
   ```bash
   python src/stage_1.py
   ```
   This trains the initial embedding model using the marginal unified cross-entropy loss.

2. **Stage 2: Refinement**
   ```bash
   python src/stage_2.py
   ```
   Note: Before running, you may need to edit the file to point to the Stage 1 checkpoint:
   ```python
   PRIMARY_CHECKPOINT_TO_LOAD = "/path/to/stage_1.pt"
   ```

### Inference
If you have the pre-trained models, you can run inference directly:

```bash
python src/infer.py
```

Make sure the inference script is pointing to the correct model checkpoints and dataset directories.

## Citation
If you use our work, please cite:
```
@misc{cryptoeprint:2024/100,
      author = {Sohaib Ahmad and Sixia Chen and Luke Demarest and Benjamin Fuller and Caleb Manicke and Alexander Russell and Amey Shukla},
      title = {Fuzzy Extractors are Practical: Cryptographic Strength Key Derivation from the Iris},
      howpublished = {Cryptology {ePrint} Archive, Paper 2024/100},
      year = {2024},
      url = {https://eprint.iacr.org/2024/100}
}
```

## License
This project is provided under GPL. The IITD dataset and the segmentation masks has its own licensing terms which must be respected.
