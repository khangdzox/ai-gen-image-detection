# AI-Generated Image Detection Pipeline

## Installation

This pipeline requires the following Python packages:
- `accelerate>=1.8.1`
- `boto3>=1.39.13`
- `diffusers>=0.33.1`
- `hf-xet>=1.1.4`
- `kagglehub>=0.3.12`
- `matplotlib>=3.10.3`
- `mlflow>=3.1.1`
- `numpy>=2.3.0`
- `pandas>=2.3.0`
- `safetensors>=0.5.3`
- `scikit-learn>=1.7.0`
- `scipy>=1.15.3`
- `torch==2.6.0`
- `torchvision>=0.21.0`
- `transformers>=4.52.4`

You can install these packages using [uv](https://docs.astral.sh/uv/):

```bash
uv pip install -r pyproject.toml
```

Or you can install missing packages manually using pip:

```bash
pip install accelerate>=1.8.1 boto3>=1.39.13 diffusers>=0.33.1 hf-xet>=1.1.4 kagglehub>=0.3.12 matplotlib>=3.10.3 mlflow>=3.1.1 numpy>=2.3.0 pandas>=2.3.0 safetensors>=0.5.3 scikit-learn>=1.7.0 scipy>=1.15.3 transformers>=4.52.4
pip install torch==2.6.0 torchvision>=0.21.0 --index-url https://download.pytorch.org/whl/cu126
```

## Usage

Run the pipeline using the following command:

```bash
python main.py exps/others/features_10_20_exp.yaml
python main.py exps/others/features_10_30_exp.yaml
python main.py exps/others/features_20_30_exp.yaml
python main.py exps/others/features_20_40_exp.yaml
python main.py exps/others/full_dataset_exp.yaml
python main.py exps/generalisation/generalisation_exp_sd14.yaml
python main.py exps/generalisation/generalisation_exp_sd15.yaml
python main.py exps/generalisation/generalisation_exp_sd21.yaml
```

A batch size of 8 used nearly all 15GB GPU memory on the Google Colab T4 GPU.