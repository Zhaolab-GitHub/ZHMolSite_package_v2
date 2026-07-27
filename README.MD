# ZHMolSite

## Introduction

RNAs have increasingly been recognized as key players in biological functions through their interactions with proteins. These RNA-protein interactions regulate diverse cellular processes, including gene transcription, post-transcriptional gene regulation, mRNA splicing and translation, non-coding RNA processing, and RNA virus replication. Notably, RNA-protein interactions are closely associated with various diseases. ZHMolSite is a deep learning approach for predicting RNA binding nucleotides hierarchically with explicit consideration of protein receptors.

## Installation

### System Requirements

ZHMolSite currently supports Ubuntu Linux with either:

- An NVIDIA GPU and an NVIDIA driver compatible with CUDA 11.8 (recommended), or
- A CPU-only environment with sufficient system memory.

Anaconda or Miniconda is recommended for environment management. ZHMolSite was tested on Ubuntu 20.04 with a 64GB-cpu or an NVIDIA GeForce RTX 4090 GPU.

> **Note:** CPU-only inference is supported but is considerably slower than GPU-accelerated inference. A GPU is strongly recommended, especially for sequence-only prediction.


### GPU Installation with NIVDIA GPU (CUDA 11.8 was used in the authors’ environment)

Download or clone this repository, navigate to the project root directory, and create a dedicated Conda environment:

```bash
git clone https://github.com/Zhaolab-GitHub/ZHMolSite_package_v2.git
conda create -n zhmolsite python=3.9 -y
conda activate zhmolsite
```

Install ZHMolSite and its CUDA 11.8-compatible dependencies:

```bash
python -m pip install ".[notebook, cu118]" \
  --extra-index-url https://download.pytorch.org/whl/cu118 \
  --find-links https://data.pyg.org/whl/torch-2.0.0+cu118.html \
  --find-links ./wheels \
  --prefer-binary
```


### CPU-Only Installation

Create and activate a separate CPU-only environment:

```bash
conda create -n zhmolsite_cpu python=3.9 -y
conda activate zhmolsite_cpu
```

Install the CPU builds of PyTorch 2.0.0 and its companion packages:

```bash
python -m pip install ".[notebook,cpu]" \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --find-links https://data.pyg.org/whl/torch-2.0.0+cpu.html \
  --find-links ./wheels \
  --prefer-binary
```

### Verify the Installation

Check the installed PyTorch version, CUDA, and selected device:

```bash
python - <<'PY'
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Selected device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

For the CUDA 11.8 installation, the expected output includes:

```text
CUDA version: 11.8
CUDA available: True
Selected device: cuda:0
```

For the CPU-only installation, the expected output includes:

```text
CUDA version: None
CUDA available: False
Selected device: cpu
```

## Code and Data

### Data Files

The following directories contain the datasets, examples, and trained model required to evaluate and run ZHMolSite:

- `Data/`: Data required to reproduce the evaluation results on the TE and CASP_PZ test sets.
- `example/`: Input and configuration files for example predictions.
- `trained_model/`: Trained ZHMolSite model weights.

### Evaluation Notebooks

The following Jupyter notebooks are provided for evaluating ZHMolSite on the benchmark datasets:

- `1_TE_evaluate.ipynb`: Evaluates ZHMolSite on the TE test set.
- `2_CASP_PZ_evaluate.ipynb`: Evaluates ZHMolSite on the CASP_PZ test set.
- `3_FoldBench21_evaluate.ipynb`: Evaluates ZHMolSite on the FoldBench21 test set.

> **Note:** To evaluate the test sets, users should first download the “Data” archive from the corresponding Zenodo repository: https://zenodo.org/records/21616001.

## Predicting Protein-Binding Nucleotides

ZHMolSite predicts protein-binding nucleotides from RNA and protein sequences, with optional RNA and protein structures.

### Prediction with Input Sequences and Structures

Run the prediction script from the root directory of the ZHMolSite project and specify a configuration file using `--config`:

```bash
# Example using an experimentally determined RNA-protein complex
python ZHMolSite_predict.py --config example/8PTZ_X_C/config.json

# Example using an aminoacyl-tRNA synthetase (aaRS)-tRNA complex
python ZHMolSite_predict.py --config example/aaRS_tRNA/config.json
```

> **Note:** The configuration parameter `RNA_thresh` specifies the probability threshold for classifying RNA nucleotides as protein-binding sites. The default value was determined using the validation dataset during model development and was used for all evaluations reported in the manuscript.

### Pretrained Language-Model Weights

The required pretrained language-model weights are downloaded automatically during the first inference run. An internet connection is therefore required when ZHMolSite is executed for the first time.

Downloaded weights are stored in the `pretrained/` directory and reused in subsequent runs.

### Sequence-Only Prediction

ZHMolSite can also run when RNA and protein structures are unavailable. Ensure that the corresponding dependencies are installed before running sequence-only inference.

#### 1. Install RhoFold and Its Dependencies

From the root directory of the ZHMolSite project, clone the official RhoFold repository and enter the new directory:

```bash
git clone https://github.com/ml4bio/RhoFold.git
```

Install RhoFold in editable mode together with its required dependencies:

```bash
cd RhoFold
python -m pip install -e . \
  "einops==0.6.1" \
  "ml-collections==0.1.1" \
  "dm-tree==0.1.8" \
  "openmm" \
  "omegaconf==2.3.0" \
  "huggingface_hub" \
  --only-binary=dm-tree,openmm
cd ..
```

> **Note:** Install these packages in the same Python or Conda environment used for ZHMolSite.


#### 2. Download the RhoFold Pretrained Model

Return to the root directory of the ZHMolSite project and download the RhoFold model into `pretrained/`:

```bash
wget \
  "https://proj.cse.cuhk.edu.hk/aihlab/RhoFold/api/download?filename=RhoFold_pretrained.pt" \
  -O RhoFold/pretrained/RhoFold_pretrained.pt
```


#### 3. Run a Sequence-Only Example

From the root directory of the ZHMolSite project, run:

```bash
python ZHMolSite_predict.py --config example/8PTZ_X_C_seq/config.json
```

## Contact

For questions, comments, or suggestions regarding ZHMolSite, please contact:

- Yunjie Zhao: [yjzhaowh@ccnu.edu.cn](mailto:yjzhaowh@ccnu.edu.cn)
- Haoquan Liu: [liuhaoquan@mails.ccnu.edu.cn](mailto:liuhaoquan@mails.ccnu.edu.cn)