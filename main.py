from argparse import ArgumentParser
import copy
import gc
import json
import logging
import os
import random
from pathlib import Path
from pprint import pformat
from datetime import datetime
import shutil

import kagglehub
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import scipy
import torch
import torchvision
import yaml
from diffusers import DDIMScheduler, StableDiffusionPipeline  # type: ignore
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from torchvision import transforms
from tqdm.auto import tqdm

CONFIG_ALLOWED_TYPE = {"generalisation", "generic"}
CONFIG_ALLOWED_DATA_DIR = {"data", "data_x", "data_aio", "data_aio_x"}
CONFIG_ALLOWED_CLASSIFIER_TYPE = {"lstm", "gru", "transformer", "first", "last"}
CONFIG_ALLOWED_DEVICE = {"cuda", "cpu"}
CONFIG_ALLOWED_FEATURES = {"noise_stats", "noise_fft", "residual_stats", "cos_sim"}

CONFIG_REQUIRED_KEYS = {
    "experiment_name",
    "type",
    "data_dir",
    "output_dir",
    "x_train_samples",
    "x_test_samples",
    "base",
    "runs",
}
CONFIG_RUN_REQUIRED_KEYS = {
    "run_name",
    "seed",
    "hf_repo",
    "timesteps",
    "diffusion_steps",
    "device",
    "classifier_type",
    "hidden_size",
    "num_layers",
    "num_classes",
    "batch_size",
    "epochs",
    "lr",
    "weight_decay",
    "included_features",
}

CONFIG_KEYS_REQUIRE_DENOISE = {
    "hf_repo",
    "timesteps",
    "diffusion_steps",
}
CONFIG_DEFAULT = {
    "experiment_name": "default_experiment",
    "data_dir": "data_aio_x",
    "output_dir": "output",
    "x_train_samples": 400,
    "x_test_samples": 100,
    "base": {
        "run_name": "this_will_be_overridden",
        "seed": 42,
        "hf_repo": "CompVis/stable-diffusion-v1-4",
        "timesteps": 30,
        "diffusion_steps": 10,
        "device": "cuda",
        "classifier_type": "lstm",
        "hidden_size": 64,
        "num_layers": 1,
        "num_classes": 2,
        "batch_size": 8,
        "epochs": 50,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "included_features": ["noise_stats", "noise_fft", "residual_stats", "cos_sim"],
    },
    "runs": [{"run_name": "default_run"}],
}

# Set up logger
# Create a custom logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set the desired logging level

# Prevent propagation to the root logger
logger.propagate = False

def add_handlers(filename):
    # Check if handlers already exist to avoid duplicate logs in Colab
    if not logger.handlers:
        # Create a file handler
        file_handler = logging.FileHandler(filename, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.INFO)

        # Create a stream handler for stdout
        stdout = logging.StreamHandler()
        stdout.setLevel(logging.INFO)

        # Create a formatter
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
        )
        file_handler.setFormatter(formatter)
        stdout.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(file_handler)
        logger.addHandler(stdout)


mlflow.set_tracking_uri("http://khangdzox.duckdns.org:25000")
os.environ["AWS_ACCESS_KEY_ID"] = "khangvo3103"
os.environ["AWS_SECRET_ACCESS_KEY"] = "vk3103@minio"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://khangdzox.duckdns.org:25001"


def plot_embedding(X_embedded, labels, title="Embedding"):
    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(
        X_embedded[:, 0], X_embedded[:, 1], c=labels, cmap="coolwarm", alpha=0.6
    )
    plt.legend(*scatter.legend_elements(), title="Class")
    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(True)
    plt.show()


def compute_energy(x_fft):
    """
    Compute the energy in low, mid, and high frequency bands of the FFT of an image.
    Args:
        x_fft (torch.Tensor): FFT of the image tensor, shape (B, T, C, H, W).
    Returns:
        low_energy (torch.Tensor): Energy in the low frequency band.
        mid_energy (torch.Tensor): Energy in the mid frequency band.
        high_energy (torch.Tensor): Energy in the high frequency band.
    """
    mag = torch.abs(x_fft)
    power = mag**2

    # shift zero-frequency to center
    power = torch.fft.fftshift(power, dim=(-2, -1))

    # define low/mid/high fred masks
    B, S, C, H, W = power.shape
    yy, xx = torch.meshgrid(
        torch.arange(H) - H // 2, torch.arange(W) - W // 2, indexing="ij"
    )

    rr = torch.sqrt(xx**2 + yy**2).to(power.device)  # radial frequency
    r_norm = rr / rr.max()

    low_mask = r_norm < 0.2
    mid_mask = (r_norm >= 0.2) & (r_norm < 0.5)
    high_mask = r_norm >= 0.5

    low_mask = low_mask[None, None, None, :, :].to(power.device)
    mid_mask = mid_mask[None, None, None, :, :].to(power.device)
    high_mask = high_mask[None, None, None, :, :].to(power.device)

    low_energy = (power * low_mask).sum(dim=(2, 3, 4))
    mid_energy = (power * mid_mask).sum(dim=(2, 3, 4))
    high_energy = (power * high_mask).sum(dim=(2, 3, 4))

    return low_energy, mid_energy, high_energy


def new_extract_noise_features(
    pred_noise: torch.Tensor, noise: torch.Tensor | None, included_features: list[str]
) -> torch.Tensor:
    """
    Extracts features from the predicted noise and optionally from the actual noise.
    Args:
        pred_noise (torch.Tensor): The predicted noise tensor, shape (B, T, C, H, W).
        noise (torch.Tensor | None): The actual noise tensor, shape (B, T, C, H, W).
        included_features (list): List of features to extract. Options are:
            - 'noise_stats': Mean, std, skewness, kurtosis, and L2 norm of the predicted noise.
            - 'noise_fft': FFT magnitude, phase, and energy in low, mid, and high frequency bands of the predicted noise.
            - 'residual_stats': Mean, std, skewness, kurtosis, and L2 norm of the residual (predicted noise - actual noise).
            - 'cos_sim': Cosine similarity between the predicted noise and the actual noise.
    Returns:
        torch.Tensor: A tensor containing the extracted features for each sample in the batch, shape (B, T, F).
    """
    # tensor shape: (batch_size, sequence_length, channels, height, width) 8 20 4 64 64
    assert included_features, "included_features cannot be empty"
    assert all(
        f in ["noise_stats", "noise_fft", "residual_stats", "cos_sim"]
        for f in included_features
    ), (
        "included_features must be one of ['noise_stats', 'noise_fft', 'residual_stats', 'cos_sim']"
    )
    assert not (
        noise is None
        and ("residual_stats" in included_features or "cos_sim" in included_features)
    ), "residual_stats and cos_sim require noise"

    residual = pred_noise - noise if noise is not None else None

    features = []

    if "noise_stats" in included_features:
        pred_flat = pred_noise.flatten(2)

        pred_mean = pred_flat.mean(dim=2)
        pred_std = pred_flat.std(dim=2)
        pred_l2 = torch.linalg.norm(pred_flat, dim=2)

        pred_np = pred_flat.cpu().numpy()
        pred_skew = torch.from_numpy(scipy.stats.skew(pred_np, axis=2)).to(pred_noise.device, dtype=pred_mean.dtype)
        pred_kurt = torch.from_numpy(scipy.stats.kurtosis(pred_np, axis=2)).to(pred_noise.device, dtype=pred_mean.dtype)

        features += [pred_mean, pred_std, pred_l2, pred_skew, pred_kurt]

    if "noise_fft" in included_features:
        pred_fft = torch.fft.fft2(pred_noise, norm="ortho")
        pred_fft_mag = torch.abs(pred_fft).mean(dim=(2, 3, 4))
        pred_fft_phase = torch.angle(pred_fft).mean(dim=(2, 3, 4))
        pred_fft_low_energy, pred_fft_mid_energy, pred_fft_high_energy = compute_energy(pred_fft)

        features += [
            pred_fft_mag,
            pred_fft_phase,
            pred_fft_low_energy,
            pred_fft_mid_energy,
            pred_fft_high_energy,
        ]

    if "residual_stats" in included_features and residual is not None:
        residual_flat = residual.flatten(2)

        residual_mean = residual_flat.mean(dim=2)
        residual_std = residual_flat.std(dim=2)
        residual_l2 = torch.linalg.norm(residual_flat, dim=2)

        residual_np = residual_flat.cpu().numpy()
        residual_skew = torch.from_numpy(scipy.stats.skew(residual_np, axis=2)).to(pred_noise.device, dtype=residual_mean.dtype)
        residual_kurt = torch.from_numpy(scipy.stats.kurtosis(residual_np, axis=2)).to(pred_noise.device, dtype=residual_mean.dtype)

        features += [residual_mean, residual_std, residual_l2, residual_skew, residual_kurt]

    if "cos_sim" in included_features and noise is not None:
        pred_flat = pred_noise.flatten(2)
        noise_flat = noise.flatten(2)

        cos_sim = torch.nn.functional.cosine_similarity(pred_flat, noise_flat, dim=2)

        features += [cos_sim]

    # Stack all features along the last dimension
    stacked_features = torch.stack(features, dim=2)

    return stacked_features

class MyPipeline:
    def __init__(
        self,
        vae,
        unet,
        tokenizer,
        text_encoder,
        scheduler,
        total_timesteps=30,
        diffusion_steps=10,
        device="cuda",
    ):
        """
        Initializes the pipeline.

        Args:
            vae (VAE): The variational autoencoder model from Stable Diffusion Pipeline.
            unet (UNet): The U-Net model for denoising from Stable Diffusion Pipeline.
            tokenizer (Tokenizer): The tokenizer for text input from Stable Diffusion Pipeline.
            text_encoder (TextEncoder): The text encoder for generating text embeddings from Stable Diffusion Pipeline.
            scheduler (Scheduler): The scheduler for controlling the diffusion process from DDIM.
            total_timesteps (int, optional): The total number of diffusion timesteps. Defaults to 30.
            diffusion_steps (float, optional): The number of timesteps to use for diffusion. Defaults to 10.
            device (str, optional): The device to run the pipeline on. Defaults to 'cuda'.
        """
        self.vae = vae.to(device)
        self.unet = unet.to(device)
        self.scheduler = scheduler
        self.device = device

        self.total_timesteps = total_timesteps
        self.t_start = total_timesteps - diffusion_steps

        self.scheduler.set_timesteps(self.total_timesteps)

        text_encoder = text_encoder.to(device)
        with torch.no_grad():
            self.text_embeddings = text_encoder(
                tokenizer("", return_tensors="pt").input_ids.to(device)
            )[0]

    def __call__(self, batch: torch.Tensor):
        """
        Runs the pipeline on a batch of images.
        Args:
            batch (Tensor): A batch of images, shape (batch_size, channels, height, width).
        Returns:
            tuple: A tuple containing:
                - pred_noises_ts (Tensor): Tensor of predicted noises for each timestep, shape (B, T, C, H, W).
                - noises_ts (Tensor): Tensor of input noises for each timestep, shape (B, T, C, H, W).
        """
        batch = batch.to(self.device)

        # Prepare text embeddings for the batch
        batch_text_embeddings = self.text_embeddings.repeat(batch.shape[0], 1, 1)

        # Encode the image using VAE
        with torch.no_grad():
            vae_output = self.vae.encode(batch)

        latents = vae_output.latent_dist.sample() * self.vae.config.scaling_factor

        # Add noise to the latents
        noises_list = []
        latents_list = []
        noise = torch.randn_like(latents).to(self.device)
        for t in self.scheduler.timesteps[self.t_start :]:
            noises_list.append(noise)
            noisy_latents = self.scheduler.add_noise(latents, noise, t)
            latents_list.append(noisy_latents)

        pred_noises_list = []
        for t, lats in tqdm(
            zip(self.scheduler.timesteps[self.t_start :], latents_list),
            total=len(latents_list),
            desc="Denoising",
            leave=False,
        ):
            with torch.no_grad():
                noises_pred = self.unet(lats, t, batch_text_embeddings).sample
                pred_noises_list.append(noises_pred)

        pred_noises_ts, noises_ts = torch.stack(pred_noises_list), torch.stack(noises_list)
        pred_noises_ts = pred_noises_ts.permute(1, 0, 2, 3, 4)  # (B, T, C, H, W)
        noises_ts = noises_ts.permute(1, 0, 2, 3, 4)  # (B, T, C, H, W)

        return pred_noises_ts, noises_ts


class LSTMClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, num_classes=2):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0,
        )
        self.dropout = torch.nn.Dropout(0.2)
        self.fc = torch.nn.Linear(hidden_size, num_classes)

        self.__init_weights()

    def __init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                torch.nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                torch.nn.init.orthogonal_(param)
            elif "bias_ih" in name or "bias_hh" in name:
                torch.nn.init.zeros_(param)
                hidden_size = param.size(0) // 4
                param.data[hidden_size:2 * hidden_size] = 1.0

        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        logits = self.fc(x)
        return logits

    def embed(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        return x


class GRUClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, num_classes=2):
        super().__init__()
        self.gru = torch.nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0,
        )
        self.dropout = torch.nn.Dropout(0.2)
        self.fc = torch.nn.Linear(hidden_size, num_classes)

        self.__init_weights()

    def __init_weights(self):
        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:
                torch.nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                torch.nn.init.orthogonal_(param)
            elif "bias_ih" in name or "bias_hh" in name:
                torch.nn.init.zeros_(param)

        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        logits = self.fc(x)
        return logits

    def embed(self, x):
        x, _ = self.gru(x)
        x = x[:, -1, :]
        return x


class TransformerClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=4, num_classes=2):
        super().__init__()
        self.projector = torch.nn.Linear(input_size, hidden_size)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=4,
            dim_feedforward=hidden_size, batch_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.dropout = torch.nn.Dropout(0.2)
        self.fc = torch.nn.Linear(hidden_size, num_classes)

        self.__init_weights()

    def __init_weights(self):
        for _, module in self.named_modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, torch.nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)

        torch.nn.init.xavier_uniform_(self.projector.weight)
        torch.nn.init.zeros_(self.projector.bias)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = self.projector(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        logits = self.fc(x)
        return logits

    def embed(self, x):
        x = self.projector(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        return x

class FirstOnlyClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, num_classes=2):
        super().__init__()
        self.seq_1 = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2)
        )

        self.seq_list = []
        for _ in range(num_layers - 1):
            self.seq_list.append(torch.nn.Linear(hidden_size, hidden_size))
            self.seq_list.append(torch.nn.ReLU())
            self.seq_list.append(torch.nn.Dropout(0.2))
        self.seq_hidden = torch.nn.Sequential(*self.seq_list)

        self.fc = torch.nn.Linear(hidden_size, num_classes)

        self.__init_weights()

    def __init_weights(self):
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.zeros_(self.fc.bias)
        for layer in self.seq_1:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
        for layer in self.seq_hidden:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

    def forward(self, x):
        # Get only first step in sequence
        x = x[:, 0, :]
        x = self.seq_1(x)
        x = self.seq_hidden(x)
        logits = self.fc(x)
        return logits

    def embed(self, x):
        # Get only first step in sequence
        x = x[:, 0, :]
        x = self.seq_1(x)
        x = self.seq_hidden(x)
        return x


class LastOnlyClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, num_classes=2):
        super().__init__()
        self.seq_1 = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2)
        )

        self.seq_list = []
        for _ in range(num_layers - 1):
            self.seq_list.append(torch.nn.Linear(hidden_size, hidden_size))
            self.seq_list.append(torch.nn.ReLU())
            self.seq_list.append(torch.nn.Dropout(0.2))
        self.seq_hidden = torch.nn.Sequential(*self.seq_list)

        self.fc = torch.nn.Linear(hidden_size, num_classes)

        self.__init_weights()

    def __init_weights(self):
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.zeros_(self.fc.bias)
        for layer in self.seq_1:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
        for layer in self.seq_hidden:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

    def forward(self, x):
        # Get only last step in sequence
        x = x[:, -1, :]
        x = self.seq_1(x)
        x = self.seq_hidden(x)
        logits = self.fc(x)
        return logits

    def embed(self, x):
        # Get only last step in sequence
        x = x[:, -1, :]
        x = self.seq_1(x)
        x = self.seq_hidden(x)
        return x


def download_and_prepare_data(output_dir="", num_train_samples=-1, num_test_samples=-1):
    # Download the dataset from Kaggle
    logger.info("Downloading dataset from Kaggle...")
    datapath = kagglehub.dataset_download("yangsangtai/tiny-genimage")

    datapath = Path(datapath)
    output_dir = Path(output_dir)

    issubset = num_train_samples > 0 and num_test_samples > 0

    def get_random_or_all_files(path, num_samples):
        if issubset:
            random.seed(42)
            return random.sample(
                sorted(os.listdir(path)),
                num_samples,
            )
        else:
            return os.listdir(path)

    if issubset:
        logger.info(
            f"Creating a subset of the dataset with {num_train_samples} train samples "
            f"and {num_test_samples} test samples for each class..."
        )
        data_dir_name = f"data_{num_train_samples}_{num_test_samples}"
        data_aio_dir_name = f"data_aio_{num_train_samples}_{num_test_samples}"
    else:
        logger.info("Using the full dataset without subset...")
        data_dir_name = "data"
        data_aio_dir_name = "data_aio"

    # Symlinks data to a new structure
    logger.info(
        f"Symlinks data to a new structure at {output_dir / 'dataset' / data_dir_name}..."
    )

    for model_dir in os.listdir(datapath):
        for split in ["train", "val"]:
            for subdir, class_dir in [("nature", "0_real"), ("ai", "1_fake")]:
                (
                    output_dir
                    / "dataset"
                    / data_dir_name
                    / split
                    / model_dir
                    / class_dir
                ).mkdir(parents=True, exist_ok=True)

                for file in get_random_or_all_files(
                    datapath / model_dir / split / subdir,
                    num_train_samples if split == "train" else num_test_samples,
                ):
                    try:
                        (
                            output_dir
                            / "dataset"
                            / data_dir_name
                            / split
                            / model_dir
                            / class_dir
                            / file
                        ).symlink_to(datapath / model_dir / split / subdir / file)
                    except FileExistsError:
                        pass

    # Join all real images and fake images into 'data_aio'
    logger.info(
        f"Joining all real and fake images of all models into 2 classes at {output_dir / 'dataset' / data_aio_dir_name}..."
    )

    for split in ["train", "val"]:
        for model_dir in os.listdir(output_dir / "dataset" / data_dir_name / split):
            for class_dir in ["0_real", "1_fake"]:
                (output_dir / "dataset" / data_aio_dir_name / split / class_dir).mkdir(
                    parents=True, exist_ok=True
                )

                for file in os.listdir(
                    output_dir
                    / "dataset"
                    / data_dir_name
                    / split
                    / model_dir
                    / class_dir
                ):
                    try:
                        (
                            output_dir
                            / "dataset"
                            / data_aio_dir_name
                            / split
                            / class_dir
                            / f"{model_dir}_{file}"
                        ).symlink_to(
                            (
                                output_dir
                                / "dataset"
                                / data_dir_name
                                / split
                                / model_dir
                                / class_dir
                                / file
                            ).readlink()
                        )
                    except FileExistsError:
                        pass

    return {
        "data": str(output_dir / "dataset" / data_dir_name),
        "data_aio": str(output_dir / "dataset" / data_aio_dir_name),
    }


def prepare_pipeline(
    hf_repo="CompVis/stable-diffusion-v1-4",
    total_timesteps=30,
    diffusion_steps=10,
    device="cuda",
):
    """Prepares the Stable Diffusion pipeline with the specified parameters.
    Args:
        hf_repo (str): The Hugging Face repository to load the Stable Diffusion model from.
        total_timesteps (int): The total number of diffusion timesteps. Defaults to 30.
        diffusion_steps (int): The number of timesteps to use for diffusion. Defaults to 10.
        device (str): The device to run the pipeline on. Defaults to 'cuda'.
    Returns:
        MyPipeline: An instance of the MyPipeline class with the prepared models and configurations.
    """
    logger.info(
        f"Preparing pipeline with repo {hf_repo}, total_timesteps={total_timesteps}, diffusion_steps={diffusion_steps}, device={device}"
    )

    pipeline = StableDiffusionPipeline.from_pretrained(hf_repo)
    scheduler = DDIMScheduler.from_pretrained(hf_repo, subfolder="scheduler")
    vae = pipeline.vae.to(device)
    unet = pipeline.unet.to(device)
    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder.to(device)

    return MyPipeline(
        vae,
        unet,
        tokenizer,
        text_encoder,
        scheduler,
        total_timesteps,
        diffusion_steps,
        device,
    )


def prepare_dataloader(folder, batch_size=8):
    """Prepares a DataLoader for the specified folder containing images.
    Args:
        folder (str): The path to the folder containing images.
    Returns:
        DataLoader: A PyTorch DataLoader for the images in the specified folder.
    """
    logger.info(f"Preparing DataLoader for folder: {folder}")
    transform = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
            transforms.ConvertImageDtype(torch.float32),
        ]
    )

    dataset = torchvision.datasets.ImageFolder(root=folder, transform=transform)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return dataloader


def atomic_torch_save(obj, filepath):
    """Saves a PyTorch object to a file atomically.
    Args:
        obj: The PyTorch object to save.
        filepath (str): The path to the file where the object will be saved.
    """
    temp_filepath = f"{filepath}.tmp"
    torch.save(obj, temp_filepath)
    os.replace(temp_filepath, filepath)


def remove_slash(s):
    s = s.replace("\\", "/").split("/")
    return "_".join(s)


def run_pipeline_denoise(pipeline: MyPipeline, dataloader, output_dir, denoise_configs):
    """Runs the pipeline to denoise images in the DataLoader.
    Args:
        pipeline: The pipeline to use for denoising.
        dataloader: The DataLoader containing the images to denoise.
    Returns:
        str: The path to the output directory where denoised images will be saved.
    """

    logger.info("Running pipeline to denoise images...")

    dataset_root = dataloader.dataset.root
    dataset_root = remove_slash(dataset_root)

    denoise_configs = remove_slash(denoise_configs)

    count_batches = 0

    os.makedirs(
        f"{output_dir}/denoise_cache/{dataset_root}_{denoise_configs}", exist_ok=True
    )

    logger.info(
        f"Looking for and storing denoised data in {output_dir}/denoise_cache/{dataset_root}_{denoise_configs}"
    )

    for batch, labels in tqdm(dataloader):
        if not os.path.exists(
            f"{output_dir}/denoise_cache/{dataset_root}_{denoise_configs}/batch_{count_batches}.pt"
        ):
            pred_noises_ts, noises_ts = pipeline(batch)
            atomic_torch_save(
                (pred_noises_ts, noises_ts, labels),
                f"{output_dir}/denoise_cache/{dataset_root}_{denoise_configs}/batch_{count_batches}.pt",
            )

        count_batches += 1

    return f"{output_dir}/denoise_cache/{dataset_root}_{denoise_configs}"


def run_pipeline_extract_features(
    denoise_path, included_features
):
    """Extracts features from the predicted noises and actual noises using the pipeline.
    Args:
        denoise_path (str): Path to the directory containing denoised batches.
        included_features (list): List of features to include in the extraction.
    Returns:
        list: A list of tensors containing the extracted features for each sample in the batch.
    """

    logger.info("Extracting features from predicted noises and actual noises...")

    output_dir = denoise_path.split("/denoise_cache")[0]
    denoise_cache_folder_name = denoise_path.split("/")[-1]
    included_features_str = "_".join(included_features)

    if os.path.exists(
        f"{output_dir}/features_cache/{denoise_cache_folder_name}/features_{included_features_str}.pt"
    ):
        logger.info(
            f"Features already extracted and cached at {output_dir}/features_cache/{denoise_cache_folder_name}/features_{included_features_str}.pt"
        )
        return torch.load(
            f"{output_dir}/features_cache/{denoise_cache_folder_name}/features_{included_features_str}.pt",
            weights_only=False,
        )

    all_features = []
    all_labels = []

    all_batches = os.listdir(denoise_path)
    all_batches = [batch for batch in all_batches if batch.endswith(".pt")]

    try:
        for i in tqdm(range(0, len(all_batches), 4), desc="Extracting features"):
            batches = all_batches[i:i + 4]

            loaded_batches = [
                torch.load(f"{denoise_path}/{batch}", weights_only=False)
                for batch in batches
            ]
            pred_noises_ts, noises_ts, labels = zip(*loaded_batches)

            pred_noises_ts = torch.concat(pred_noises_ts)
            noises_ts = torch.concat(noises_ts)
            labels = torch.concat(labels)

            features = new_extract_noise_features(
                pred_noises_ts, noises_ts, included_features
            )
            all_features.append(features.detach().cpu())
            all_labels.append(labels.detach().cpu())

            del pred_noises_ts, noises_ts, labels
            gc.collect()
            torch.cuda.empty_cache()

    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        raise e

    all_features_ts = torch.concat(all_features).cuda()
    all_labels_ts = torch.concat(all_labels).cuda()

    os.makedirs(f"{output_dir}/features_cache/{denoise_cache_folder_name}", exist_ok=True)
    atomic_torch_save(
        (all_features_ts, all_labels_ts),
        f"{output_dir}/features_cache/{denoise_cache_folder_name}/features_{included_features_str}.pt",
    )

    return all_features_ts, all_labels_ts


def train_classifier(
    classifier,
    X_train,
    y_train,
    X_val,
    y_val,
    batch_size=8,
    epochs=50,
    lr=1e-3,
    weight_decay=1e-5,
    device="cuda",
):
    """Trains a classifier on the provided training data.
    Args:
        classifier: The classifier model to train.
        X_train: The training features.
        y_train: The training labels.
        X_val: The validation features.
        y_val: The validation labels.
        batch_size: The batch size for training.
        epochs: The number of training epochs.
        lr: The learning rate.
        weight_decay: The weight decay for the optimizer.
        device: The device to train on (e.g., 'cuda' or 'cpu').
    Returns:
        tuple: A tuple containing:
            - The trained classifier model.
            - The state dictionary of the best model.
    """

    logger.info(
        f"Training classifier with batch_size={batch_size}, epochs={epochs}, lr={lr}, weight_decay={weight_decay}, device={device}"
    )
    logger.info(
        f"Early stopping patience set to {EARLY_STOPPING_PATIENCE} epochs"
    )

    classifier.to(device)

    optimizer = torch.optim.AdamW(
        classifier.parameters(), lr=lr, weight_decay=weight_decay
    )
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-7, cooldown=3
    # )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer, T_0=10, T_mult=2, eta_min=1e-7
    # )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-7
    )
    criterion = torch.nn.CrossEntropyLoss()

    best_model_state_dict = None
    best_eval_loss = float("inf")
    best_epoch = 0

    for epoch in range(epochs):
        classifier.train()

        running_loss = 0
        correct = 0
        total = 0
        is_best = False

        for i in range(0, X_train.shape[0], batch_size):
            batch = X_train[i : i + batch_size].to(device).to(torch.float32)
            labels = y_train[i : i + batch_size].to(device)

            logits = classifier(batch)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * batch.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        with torch.no_grad():
            classifier.eval()

            val_running_loss = 0
            val_correct = 0
            val_total = 0

            for i in tqdm(range(0, X_val.shape[0], batch_size), leave=False):
                batch = X_val[i : i + batch_size].to(device).to(torch.float32)
                labels = y_val[i : i + batch_size].to(device)

                logits = classifier(batch)
                val_loss = criterion(logits, labels)

                val_running_loss += val_loss.item() * batch.size(0)

                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        # scheduler.step(val_running_loss / val_total)
        scheduler.step()

        if val_running_loss / val_total < best_eval_loss:
            best_eval_loss = val_running_loss / val_total
            best_model_state_dict = copy.deepcopy(classifier.state_dict().copy())
            is_best = True
            best_epoch = epoch

        logger.info(
            f"Epoch {epoch + 1:>2}: Loss = {running_loss / total:.6f}, Accuracy = {correct / total:.6f}, Val Loss = {val_running_loss / val_total:.6f}, Val Accuracy = {val_correct / val_total:.6f}, LR = {scheduler.get_last_lr()[0]:e}{', New best' if is_best else ''}"
        )

        mlflow.log_metrics(
            {
                "train_loss": running_loss / total,
                "train_accuracy": correct / total,
                "val_loss": val_running_loss / val_total,
                "val_accuracy": val_correct / val_total,
            },
            step=epoch + 1,
        )

        if epoch - best_epoch > EARLY_STOPPING_PATIENCE:
            logger.info(
                f"Early stopping triggered at epoch {epoch + 1}, no improvement in validation loss for {EARLY_STOPPING_PATIENCE} epochs."
            )
            break

    assert best_model_state_dict is not None, (
        "Training failed, either epochs == 0 or data is empty. No model was trained."
    )
    return classifier, best_model_state_dict


def evaluate_classifier(classifier, X_test, y_test, batch_size=8, device="cuda"):
    """Evaluates the classifier on the test data.
    Args:
        classifier (torch.nn.Module): The trained classifier.
        X_test (torch.Tensor): The test features.
        y_test (torch.Tensor): The test labels.
        batch_size (int): The batch size for evaluation.
        device (str): The device to run the evaluation on (e.g., 'cuda' or 'cpu').
    Returns:
        dict: A dictionary containing the classification report with precision, recall, f1-score, and support for each class.
    """

    logger.info(f"Evaluating classifier with batch_size={batch_size}, device={device}")

    classifier.to(device)
    classifier.eval()

    all_preds = []
    all_labels = []
    all_preds_probs = []

    with torch.no_grad():
        for i in range(0, X_test.shape[0], batch_size):
            batch = X_test[i : i + batch_size].to(device).to(torch.float32)
            labels = y_test[i : i + batch_size].to(device)

            logits = classifier(batch)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_preds_probs.extend(logits.softmax(dim=1)[:, 1].cpu().numpy())

    return accuracy_report(
        y_true=np.array(all_labels),
        y_pred=np.array(all_preds),
        y_pred_probs=np.array(all_preds_probs),
    )


def accuracy_report(y_true, y_pred, y_pred_probs=None):
    """
    Generates a report containing accuracy for each class, overall accuracy, and weighted F1 score.
    Args:
        y_true (list or np.ndarray): Ground truth labels.
        y_pred (list or np.ndarray): Predicted labels.
    Returns:
        dict: A dictionary containing accuracy for each class, overall accuracy, and weighted F1 score.
    """
    unique_classes = np.unique(y_true)
    accuracy_per_class = {}
    for cls in unique_classes:
        cls_indices = np.where(y_true == cls)[0]
        accuracy_per_class[f"accuracy_{cls}"] = accuracy_score(
            y_true[cls_indices], y_pred[cls_indices]
        )

    total_accuracy = accuracy_score(y_true, y_pred)

    weighted_f1_score = f1_score(y_true, y_pred, average="weighted")

    auc = roc_auc_score(y_true, y_pred_probs) if y_pred_probs is not None else None
    average_precision = (
        average_precision_score(y_true, y_pred_probs)
        if y_pred_probs is not None
        else None
    )

    return {
        **accuracy_per_class,
        "accuracy": total_accuracy,
        "f1": weighted_f1_score,
        "auc": float(auc) if auc is not None else None,
        "map": float(average_precision) if average_precision is not None else None,
    }


def visualise_tsne_pca(classifier, X, y, batch_size=32, device="cuda"):
    """
    Visualizes the embeddings of the classifier using t-SNE or PCA.
    Args:
        classifier (torch.nn.Module): The trained classifier.
        X (torch.Tensor): The input data for visualization.
        y (torch.Tensor): The labels for the input data.
        device (str): The device to run the classifier on. Defaults to 'cuda'.
    """
    classifier.to(device)
    classifier.eval()

    embeddings = []
    labels = []

    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            batch = X[i : i + batch_size].to(device).to(torch.float32)
            labels = y[i : i + batch_size].to(device)

            embs = classifier.embed(batch)
            embeddings.extend(embs.cpu().numpy())
            labels.extend(labels.cpu().numpy())

    embeddings = np.array(embeddings)
    labels = np.array(labels)

    pca = PCA(n_components=2)
    tsne = TSNE(n_components=2, random_state=42)

    X_pca = pca.fit_transform(embeddings)
    X_tsne = tsne.fit_transform(embeddings)

    plot_embedding(X_pca, labels, title="PCA Visualization")
    plot_embedding(X_tsne, labels, title="t-SNE Visualization")


def set_random_seed(seed):
    """Sets the random seed for reproducibility.
    Args:
        seed (int): The seed value to set.
    """
    logger.info(f"Setting random seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalise_features(features):
    """Normalizes the features by subtracting the mean and dividing by the standard deviation.
    Args:
        features (torch.Tensor): The features to normalize, shape (batch_size, num_features).
    Returns:
        torch.Tensor: The normalized features.
    """
    mean = features.mean(dim=(0, 1))
    std = features.std(dim=(0, 1))
    return (features - mean[None, None, :]) / (
        std[None, None, :] + 1e-8
    )  # Add small value to avoid division by zero


def get_classifier(
    classifier_type, input_size, hidden_size=64, num_layers=1, num_classes=2
):
    set_random_seed(42)  # Ensure reproducibility

    match classifier_type.lower():
        case "lstm":
            return LSTMClassifier(input_size, hidden_size, num_layers, num_classes)
        case "gru":
            return GRUClassifier(input_size, hidden_size, num_layers, num_classes)
        case "transformer":
            return TransformerClassifier(
                input_size, hidden_size, num_layers, num_classes
            )
        case "first":
            return FirstOnlyClassifier(input_size, hidden_size, num_layers, num_classes)
        case "last":
            return LastOnlyClassifier(input_size, hidden_size, num_layers, num_classes)
        case _:
            raise ValueError(f"Unknown classifier type: {classifier_type}")


def run_denoise(config, pipeline, train_dataloader, val_dataloader):
    """Runs the denoising process with the given configuration.
    Args:
        config (dict): A configuration dictionary containing all necessary parameters.
    """
    logger.info("Running denoising...")

    # Set random seeds for reproducibility
    set_random_seed(config["seed"])

    # Load and preprocess data
    train_denoise_path = run_pipeline_denoise(
        pipeline,
        train_dataloader,
        config["output_dir"],
        "_".join(str(config[k]) for k in sorted(list(CONFIG_KEYS_REQUIRE_DENOISE))),
    )
    val_denoise_path = run_pipeline_denoise(
        pipeline,
        val_dataloader,
        config["output_dir"],
        "_".join(str(config[k]) for k in sorted(list(CONFIG_KEYS_REQUIRE_DENOISE))),
    )

    gc.collect()
    torch.cuda.empty_cache()

    return (
        train_denoise_path,
        val_denoise_path,
    )


def run_extract_features_and_normalize(
    included_features,
    denoise_path,
    train_mean=None,
    train_std=None
):

    features_ts, labels_ts = run_pipeline_extract_features(
        denoise_path, included_features
    )

    if train_mean is None or train_std is None:
        train_mean = features_ts.mean(dim=(0, 1), keepdim=True)
        train_std = features_ts.std(dim=(0, 1), keepdim=True)

    features_ts = (features_ts - train_mean) / (
        train_std + 1e-8
    )  # Add small value to avoid division by zero

    return features_ts, labels_ts, train_mean, train_std


def run_train_classifier(
    config,
    train_features_ts,
    train_labels_ts,
    val_features_ts,
    val_labels_ts,
):

    logger.info(
        f"Extracted features: train={train_features_ts.shape}, val={val_features_ts.shape}"
    )
    logger.info(
        f"Train labels: {train_labels_ts.shape}, Val labels: {val_labels_ts.shape}"
    )
    logger.info(
        f"Train features mean: {train_features_ts.mean().item()}, std: {train_features_ts.std().item()}"
    )  # Debugging information for feature normalization
    logger.info(
        f"Val features mean: {val_features_ts.mean().item()}, std: {val_features_ts.std().item()}"
    )  # Debugging information for feature normalization

    # Initialize classifier
    classifier = get_classifier(
        classifier_type=config["classifier_type"],
        input_size=train_features_ts.shape[-1],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        num_classes=config["num_classes"],
    )
    classifier.to(config["device"])

    logger.info(f"Initialized classifier: {classifier}")

    with mlflow.start_run(
        run_name=f"{config['run_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ):
        mlflow.log_params(
            {
                "included_features": config["included_features"],
                "timesteps": config["timesteps"],
                "diffusion_steps": config["diffusion_steps"],
                "hidden_size": config["hidden_size"],
                "num_layers": config["num_layers"],
                # Adjusted for larger batch size
                "batch_size": config["batch_size"] * 2,
                "epochs": config["epochs"],
                "learning_rate": config["lr"],
                "weight_decay": config["weight_decay"],
            }
        )

        # Train classifier
        trained_classifier, best_model_state_dict = train_classifier(
            classifier,
            train_features_ts,
            train_labels_ts,
            val_features_ts,
            val_labels_ts,
            config["batch_size"] * 2,
            config["epochs"],
            config["lr"],
            config["weight_decay"],
            config["device"],
        )

        # Save the trained model
        os.makedirs(f"{config['output_dir']}/models", exist_ok=True)
        model_save_path = f"{config['output_dir']}/models/model_{config['run_name']}.pt"
        torch.save(best_model_state_dict, model_save_path)

        mlflow.log_artifact(model_save_path, artifact_path="models")

        trained_classifier.load_state_dict(best_model_state_dict)

        current_run_id = mlflow.active_run().info.run_id # type: ignore

    return trained_classifier, current_run_id

def run_evaluate_classifier(
    config,
    trained_classifier,
    test_features_ts,
    test_labels_ts,
    current_run_id,
):

    logger.info(
        f"Extracted features: test={test_features_ts.shape}"
    )
    logger.info(
        f"Test labels: {test_labels_ts.shape}"
    )

    logger.info(
        f"Test features mean: {test_features_ts.mean().item()}, std: {test_features_ts.std().item()}"
    )  # Debugging information for feature normalization

    with mlflow.start_run(run_id=current_run_id):

        # Evaluate classifier
        eval_report = evaluate_classifier(
            trained_classifier,
            test_features_ts,
            test_labels_ts,
            config["batch_size"] * 4,
            config["device"],
        )

        eval_report["run_name"] = config["run_name"]

        logger.info(f"Evaluation report: {eval_report}")

        # Save evaluation report
        eval_report_path = (
            f"{config['output_dir']}/reports/eval_report_{config['run_name']}.json"
        )

        os.makedirs(f"{config['output_dir']}/reports/", exist_ok=True)
        with open(eval_report_path, "w") as f:
            json.dump(eval_report, f)

        mlflow.log_artifact(eval_report_path, artifact_path="reports")

    logger.info(
        f"Run '{config['run_name']}' completed. Evaluation report saved to {eval_report_path}"
    )

    gc.collect()
    torch.cuda.empty_cache()

    return eval_report


def run_experiment(base_config, run_configs):
    """Execute multiple runs with the given configuration.
    Args:
        config (dict): A configuration dictionary containing all necessary parameters.
    """

    train_dataloader = prepare_dataloader(
        f"{base_config['data_dir']}/train", batch_size=base_config["batch_size"]
    )
    test_dataloader = prepare_dataloader(
        f"{base_config['data_dir']}/val", batch_size=base_config["batch_size"]
    )

    reports = []

    if os.path.exists(f"{base_config['output_dir']}/finished_models.txt"):
        with open(f"{base_config['output_dir']}/finished_models.txt", "r") as f:
            finished_models = set(f.read().splitlines())
    else:
        finished_models = set()

    for run_config in run_configs:

        # Merge base config with run-specific config
        merged_config = merge_configs(base_config, run_config)

        if merged_config["run_name"] in finished_models:
            logger.info(
                f"Skipping {merged_config['run_name']} as it has already been processed."
            )
            if (file := f"eval_report_{merged_config['run_name']}.json") in os.listdir(
                f"{merged_config['output_dir']}/reports"
            ):
                logger.info(f"Loading existing report for {file}...")
                with open(
                    f"{merged_config['output_dir']}/reports/{file}", "r"
                ) as f:
                    report = json.load(f)
                    reports.append(report)
                    logger.info(report)
            continue

        # Set random seeds for reproducibility
        set_random_seed(merged_config["seed"])

        # Initialize pipeline
        pipeline = prepare_pipeline(
            hf_repo=merged_config["hf_repo"],
            total_timesteps=merged_config["timesteps"],
            diffusion_steps=merged_config["diffusion_steps"],
            device=merged_config["device"],
        )

        train_denoise_path, val_denoise_path = run_denoise(
            merged_config, pipeline, train_dataloader, test_dataloader
        )
        train_features_ts, train_labels_ts, train_mean, train_std = run_extract_features_and_normalize(
            merged_config["included_features"], train_denoise_path
        )
        val_features_ts, val_labels_ts, _, _ = run_extract_features_and_normalize(
            merged_config["included_features"], val_denoise_path, train_mean, train_std
        )
        classifier, run_id = run_train_classifier(
            merged_config, train_features_ts, train_labels_ts, val_features_ts, val_labels_ts
        )
        eval_report = run_evaluate_classifier(
            merged_config, classifier, val_features_ts, val_labels_ts, run_id
        )
        reports.append(eval_report)

        # Save the run name to finished models
        finished_models.add(merged_config["run_name"])
        with open(
            f"{merged_config['output_dir']}/finished_models.txt", "w"
        ) as f:
            f.writelines(f"{model}\n" for model in finished_models)

        gc.collect()
        torch.cuda.empty_cache()

    return reports


def run_generalisation_experiment(base_config, run_configs):
    """Runs a generalisation experiment with the given configuration.
    Args:
        config (dict): A configuration dictionary containing all necessary parameters.
    """

    model_dirs = os.listdir(f"{base_config['data_dir']}/train")

    train_dataloaders = {
        model_dir: prepare_dataloader(f"{base_config['data_dir']}/train/{model_dir}")
        for model_dir in model_dirs
    }
    test_dataloaders = {
        model_dir: prepare_dataloader(f"{base_config['data_dir']}/val/{model_dir}")
        for model_dir in model_dirs
    }

    logger.info(
        f"Running generalisation experiment with {len(model_dirs)} model directories..."
    )

    # Set random seeds for reproducibility
    set_random_seed(base_config["seed"])

    # Initialize pipeline
    pipeline = prepare_pipeline(
        hf_repo=base_config["hf_repo"],
        total_timesteps=base_config["timesteps"],
        diffusion_steps=base_config["diffusion_steps"],
        device=base_config["device"],
    )

    model_denoise_paths = {}

    for model_dir, (train_dataloader, test_dataloader) in zip(
        model_dirs, zip(train_dataloaders.values(), test_dataloaders.values())
    ):
        denoise_paths = run_denoise(
            base_config, pipeline, train_dataloader, test_dataloader
        )

        model_denoise_paths[model_dir] = denoise_paths

    reports = []
    logger.info("Starting generalisation experiments...")

    if os.path.exists(f"{base_config['output_dir']}/finished_models.txt"):
        with open(f"{base_config['output_dir']}/finished_models.txt", "r") as f:
            finished_models = set(f.read().splitlines())
    else:
        finished_models = set()

    for this_model_dir in model_dirs:  # pnd is pipeline_and_data
        if this_model_dir in finished_models:
            logger.info(
                f"Skipping {this_model_dir} as it has already been processed."
            )
            for file in os.listdir(
                f"{base_config['output_dir']}/reports"
            ):
                if file.startswith(
                    _prefix := f"eval_report_generalisation_{this_model_dir}_to_"
                ):
                    logger.info(f"Loading existing report for {file}...")
                    with open(
                        f"{base_config['output_dir']}/reports/{file}", "r"
                    ) as f:
                        report = json.load(f)
                        report["original_model"] = this_model_dir
                        report["target_model"] = file[len(_prefix) :].split(".")[0]
                        reports.append(report)
            continue

        (
            this_train_denoise_path,
            this_val_denoise_path,
        ) = model_denoise_paths[this_model_dir]

        base_config["run_name"] = (
            f"generalisation_from_{this_model_dir}"
        )

        logger.info(
            f"Running generalisation experiment from {this_model_dir}..."
        )

        train_features_ts, train_labels_ts, train_mean, train_std = run_extract_features_and_normalize(
            base_config["included_features"], this_train_denoise_path
        )
        val_features_ts, val_labels_ts, _, _ = run_extract_features_and_normalize(
            base_config["included_features"], this_val_denoise_path, train_mean, train_std
        )
        classifier, run_id = run_train_classifier(
            base_config, train_features_ts, train_labels_ts, val_features_ts, val_labels_ts
        )

        # inner loop to iterate over each model directory
        for that_model_dir in model_dirs:
            _, that_val_denoise_path = model_denoise_paths[that_model_dir]

            base_config["run_name"] = (
                f"generalisation_{this_model_dir}_to_{that_model_dir}"
            )

            logger.info(
                f"Running generalisation experiment from {this_model_dir} to {that_model_dir}..."
            )

            # Train classifier on this model's training data and evaluate on that model's validation data
            test_features_ts, test_labels_ts, _, _ = run_extract_features_and_normalize(
                base_config["included_features"], that_val_denoise_path, train_mean, train_std
            )
            eval_report = run_evaluate_classifier(
                base_config, classifier, test_features_ts, test_labels_ts, run_id
            )

            eval_report["original_model"] = this_model_dir
            eval_report["target_model"] = that_model_dir

            reports.append(eval_report)

        # Save the finished model to avoid re-running
        finished_models.add(this_model_dir)
        with open(f"{base_config['output_dir']}/finished_models.txt", "w") as f:
            f.writelines(f"{model_dir}\n" for model_dir in finished_models)

    return reports


def merge_configs(base_config, experiment_config):
    """Merges the base configuration with the experiment-specific configuration.
    Args:
        base_config (dict): The base configuration dictionary.
        experiment_config (dict): The experiment-specific configuration dictionary.
    Returns:
        dict: The merged configuration dictionary.
    """
    merged_config = copy.deepcopy(base_config)
    merged_config.update(experiment_config)
    return merged_config


def normalise_config(config, dir_prefix=""):
    """Normalises the configuration dictionary.
    Args:
        config (dict): The configuration dictionary to normalise.
    Returns:
        dict: The normalised configuration dictionary.
    """
    if "output_dir" in config:
        config["output_dir"] = os.path.join(
            dir_prefix, config["output_dir"]
        )

    if IS_ON_GOOGLE_COLAB and "experiment_name" in config:
        config["experiment_name"] = f"{config['experiment_name']}_colab"

    def normalise_run_config(run_config):
        if not torch.cuda.is_available():
            run_config["device"] = "cpu"
        return run_config

    config["base"] = normalise_run_config(config["base"])

    for i, run_config in enumerate(config["runs"]):
        config["runs"][i] = normalise_run_config(run_config)

    return config


def validate_config(config):
    """Validates the configuration dictionary for required keys.
    Args:
        config (dict): The configuration dictionary to validate.
    Raises:
        ValueError: If any required key is missing in the configuration.
    """
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a dictionary.")

    if not set(config.keys()) == CONFIG_REQUIRED_KEYS:
        raise ValueError(
            f"Configuration must contain keys {CONFIG_REQUIRED_KEYS}. Found: {set(config.keys())}"
        )

    if not (
        isinstance(config["base"], dict)
        and isinstance(config["runs"], list)
        and all(isinstance(run, dict) for run in config["runs"])
    ):
        raise ValueError(
            "Configuration 'base' must be a dictionary and 'runs' must be a list of dictionaries."
        )

    if config["type"] not in CONFIG_ALLOWED_TYPE:
        raise ValueError(
            f"Configuration 'type' must be one of {CONFIG_ALLOWED_TYPE}. Found: {config['type']}"
        )

    if config["data_dir"] not in CONFIG_ALLOWED_DATA_DIR:
        raise ValueError(
            f"Configuration 'data_dir' must be one of {CONFIG_ALLOWED_DATA_DIR}. Found: {config['data_dir']}"
        )

    def validate_run_config(run_config):
        """Validates a single run configuration."""
        if not set(run_config.keys()).issubset(CONFIG_RUN_REQUIRED_KEYS):
            raise ValueError(
                f"Run configuration must contain only allowed keys: {CONFIG_RUN_REQUIRED_KEYS}. "
                f"Found unrecognized keys: {set(run_config.keys()) - CONFIG_RUN_REQUIRED_KEYS}"
            )

        if (
            "classifier_type" in run_config
            and run_config["classifier_type"] not in CONFIG_ALLOWED_CLASSIFIER_TYPE
        ):
            raise ValueError(
                f"Configuration 'classifier_type' must be one of {CONFIG_ALLOWED_CLASSIFIER_TYPE}. "
                f"Found: {run_config['classifier_type']}"
            )

        if "device" in run_config and run_config["device"] not in CONFIG_ALLOWED_DEVICE:
            raise ValueError(
                f"Configuration 'device' must be one of {CONFIG_ALLOWED_DEVICE}. "
                f"Found: {run_config['device']}"
            )

        if "included_features" in run_config and not set(
            run_config["included_features"]
        ).issubset(CONFIG_ALLOWED_FEATURES):
            raise ValueError(
                f"Configuration 'included_features' must be a subset of {CONFIG_ALLOWED_FEATURES}. "
                f"Found: {set(run_config['included_features']) - CONFIG_ALLOWED_FEATURES}"
            )

    validate_run_config(config["base"])

    for i, run_config in enumerate(config["runs"]):
        try:
            validate_run_config(run_config)
        except ValueError as e:
            raise ValueError(f"{e} at run {i}")


def main(config_path, dir_prefix, rerun_exp=False):
    """Main function to run an experiment.
    Args:
        config (dict): A configuration dictionary containing all necessary parameters.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    validate_config(config)

    config = merge_configs(CONFIG_DEFAULT, config)

    config = normalise_config(config, dir_prefix)

    os.makedirs(config["output_dir"], exist_ok=True)

    add_handlers(f"{config['output_dir']}/mypipeline.log")

    logger.info("################# Starting mypipeline... #################")

    mlflow.set_experiment(config["experiment_name"])

    logger.info(f"Executing config: {pformat(config, sort_dicts=False)}")

    if IS_ON_GOOGLE_COLAB:
        datadir_map = download_and_prepare_data(
            output_dir=".",
            num_train_samples=config["x_train_samples"],
            num_test_samples=config["x_test_samples"],
        )
    else:
        datadir_map = download_and_prepare_data(
            output_dir=config["output_dir"],
            num_train_samples=config["x_train_samples"],
            num_test_samples=config["x_test_samples"],
        )

    base_config = config["base"]
    run_configs = config["runs"] or [{}]
    base_config["data_dir"] = datadir_map[config["data_dir"]]
    base_config["output_dir"] = config["output_dir"]

    if rerun_exp:
        logger.info("Rerun experiment flag is set. Clearing previous results...")
        if os.path.exists(f"{base_config['output_dir']}/finished_models.txt"):
            os.remove(f"{base_config['output_dir']}/finished_models.txt")
        for folder in ["models", "reports"]:
            folder_path = f"{base_config['output_dir']}/{folder}"
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
                logger.info(f"Removed folder: {folder_path}")

    try:
        # Run experiment
        if config["type"] == "generic":
            reports = run_experiment(base_config, run_configs)
        elif config["type"] == "generalisation":
            reports = run_generalisation_experiment(base_config, run_configs)
        else:
            reports = []

        reports_df = pd.DataFrame(reports)
        reports_df.to_csv(f"{base_config['output_dir']}/eval_reports.csv", index=False)

        with mlflow.start_run(
            run_name=f"reporting_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ):
            mlflow.log_params(base_config)
            mlflow.log_table(
                reports_df,
                artifact_file="reports/eval_reports.json",
            )
            mlflow.log_artifact(
                f"{base_config['output_dir']}/eval_reports.csv",
                artifact_path="reports",
            )

        logger.info(
            f"Experiment {config['experiment_name']} completed. Evaluation reports saved to {base_config['output_dir']}/eval_reports.csv"
        )
        logger.info(f"Evaluation reports:\n{reports_df}")

    except Exception as e:
        logger.error(f"An error has occurred: {e}", exc_info=True)

    finally:
        gc.collect()
        torch.cuda.empty_cache()

        with mlflow.start_run(
            run_name=f"logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ):
            mlflow.log_artifact(f"{config['output_dir']}/mypipeline.log")


try:
    from google.colab import drive  # type:ignore

    if not os.path.exists("/content/drive"):
        drive.mount("/content/drive")

    IS_ON_GOOGLE_COLAB = True

except ImportError:
    IS_ON_GOOGLE_COLAB = False

EARLY_STOPPING_PATIENCE = 300


if __name__ == "__main__":
    parser = ArgumentParser(prog="MyPipeline Experiment")
    parser.add_argument("config")
    parser.add_argument("--dir-prefix", type=str, default="", help="Prefix for output directory")
    parser.add_argument("--rerun-exp", action="store_true", help="Rerun the experiment even if it has been completed before")

    args = parser.parse_args()

    main(args.config, args.dir_prefix, args.rerun_exp)

    # Quick code for running on Google Colab

    # main("/content/drive/MyDrive/mypipeline_exps_confs/exps/generalisation/generalisation_exp.yaml", "/content/drive/MyDrive/mypipeline_exps_7/")

    # for exp_file in os.listdir("/content/drive/MyDrive/mypipeline_exps_confs/exps/others"):
    #     main(f"/content/drive/MyDrive/mypipeline_exps_confs/exps/others/{exp_file}", "/content/drive/MyDrive/mypipeline_exps_7/")

    # main("/content/drive/MyDrive/mypipeline_exps_confs/exps/others/full_dataset_exp.yaml", "/content/drive/MyDrive/mypipeline_exps_7/")
    # main("/content/drive/MyDrive/mypipeline_exps_confs/exps/generalisation/generalisation_exp_sd15.yaml", "/content/drive/MyDrive/mypipeline_exps_7/")
    # main("/content/drive/MyDrive/mypipeline_exps_confs/exps/generalisation/generalisation_exp_sd21.yaml", "/content/drive/MyDrive/mypipeline_exps_7/")
