# Fuzzy Extractors are Practical - Feature Extractor

This repository contains code for iris feature extraction and embedding generation for [FEAP](https://eprint.iacr.org/2024/100) using a multi-stage training pipeline. The system is designed to work with the IITD Iris Database.

## Pre-trained Models
Pre-trained model checkpoints are available at [this link](https://uconn-my.sharepoint.com/:f:/g/personal/benjamin_fuller_uconn_edu/Em5J7jAZdgtBqEcJwuKdOioBJtR9FHeS3vujnCj215Td-Q?e=deWjy0). The password is `IDONTKNOW`.

## Dataset Requirements
This project uses the IITD Iris Dataset, which you need to obtain separately due to licensing requirements. The dataset can be requested from [IIT Delhi Iris Database](http://www4.comp.polyu.edu.hk/~csajaykr/IITD/Database_Iris.htm).

### Dataset Segmentation
The iris images need to be segmented using [Ahmad and Fuller](https://arxiv.org/pdf/1812.08245). 

### Dataset Structure
The dataset should be organized as follows:

#### Training Data
```
IITD_folders/IITD_folders_train_only/
├── 001/
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── 002/
├── 003/
└── ...
```

Where each numbered folder represents a class (person).

#### Inference Data

```
IITD_folders/IITD_folders_inference_only/
├── 014d5R.png
├── 023d8R.png
└── ...
```
Note that the inference data is flattened (not organized in folders by class) to prevent accidentally training on inference data.

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
scikit-learn>=0.24.0
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

2. **Stage 2: Circle Loss**
   ```bash
   python src/stage_2.py
   ```
   Note: Before running, you may need to edit the file to point to the Stage 1 checkpoint:
   ```python
   train_iris_model(config, resume_training=True, checkpoint_path='path/to/stage_1.pt')
   ```

3. **Stage 3: Refinement**
   ```bash
   python src/stage_3.py
   ```
   Note: Before running, ensure the file correctly references the Stage 2 checkpoint.

### Inference
If you have the pre-trained models, you can run inference directly:

```bash
python src/infer_stage_3.py
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
This project is provided under GPL. The IITD dataset has its own licensing terms which must be respected.
