import shutil
import os
import pprint
from pathlib import Path

import kagglehub
from sklearn.metrics import balanced_accuracy_score, average_precision_score, roc_auc_score, f1_score
import pandas as pd
import torch
import numpy as np
import random

from dmimagedetection.test_code.main import runnig_tests

from universalfakedetect.validate import RealFakeDataset
from universalfakedetect.models import get_model

from aeroblade.src.aeroblade.high_level_funcs import compute_distances

def run_universal_face_detect(arch, ckpt, result_folder, real_path, fake_path):
    model = get_model(arch)
    state_dict = torch.load(ckpt, map_location='cpu')
    model.fc.load_state_dict(state_dict)
    model.eval()
    model.cuda()

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    dataset = RealFakeDataset(
        real_path=real_path,
        fake_path=fake_path,
        data_mode='ours',
        max_sample=1000,
        arch=arch,
        jpeg_quality=None,
        gaussian_sigma=None
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

    with torch.no_grad():
        y_true, y_pred = [], []
        for img, label in loader:
            in_tens = img.cuda()

            y_pred.extend(model(in_tens).sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())

        y_true, y_pred = np.array(y_true), np.array(y_pred)

    return y_true, y_pred


def run_evaluation_metrics(y_true, y_score, to_y_pred=lambda x: x > 0.5):
    y_pred = to_y_pred(y_score)
    return {
        'acc': balanced_accuracy_score(y_true, y_pred),
        'acc_fake': balanced_accuracy_score(y_true[y_true == 1], y_pred[y_true == 1]),
        'acc_real': balanced_accuracy_score(y_true[y_true == 0], y_pred[y_true == 0]),
        'auc': roc_auc_score(y_true, y_score),
        'ap': average_precision_score(y_true, y_score),
        'f1': f1_score(y_true, y_pred),
    }

def main():
    datapath = kagglehub.dataset_download("yangsangtai/tiny-genimage")
    print(f"Dataset downloaded to: {datapath}")

    os.makedirs('data/0_real', exist_ok=True)
    os.makedirs('data/1_fake', exist_ok=True)

    for dir in os.listdir(datapath):
        for file in os.listdir(f'{datapath}/{dir}/val/real'):
            shutil.copy(f'{datapath}/{dir}/val/real/{file}', f'data/0_real/{dir}_{file}')
        for file in os.listdir(f'{datapath}/{dir}/val/ai'):
            shutil.copy(f'{datapath}/{dir}/val/ai/{file}', f'data/1_fake/{dir}_{file}')

    os.makedirs('data_1000/0_real', exist_ok=True)
    os.makedirs('data_1000/1_fake', exist_ok=True)

    random.seed(42)  # For reproducibility
    random_1000_real = random.sample(os.listdir('data/0_real'), 1000)
    random_1000_fake = random.sample(os.listdir('data/1_fake'), 1000)
    for file in random_1000_real:
        shutil.copy(f'data/0_real/{file}', f'data_1000/0_real/{file}')
    for file in random_1000_fake:
        shutil.copy(f'data/1_fake/{file}', f'data_1000/1_fake/{file}')

    # ---------------- #
    # DMimageDetection #
    # ---------------- #

    os.makedirs('output/dmid', exist_ok=True)

    dmid_data = {'src': []}
    for dir in os.listdir('data'):
        for file in os.listdir(f'data/{dir}'):
            dmid_data['src'].append(f'{dir}/{file}')
    pd.DataFrame(dmid_data).to_csv('dmid_data.csv', index=False)

    runnig_tests(data_path="data/", output_dir="output/dmid/", weights_dir="weights/", csv_file="dmid_data.csv")

    df_dmid_real = pd.read_csv('output/dmid/0_real/0_real.csv')
    df_dmid_fake = pd.read_csv('output/dmid/1_fake/1_fake.csv')

    dmid_models = ['Grag2021_progan', 'Grag2021_latent']

    df_dmid = pd.concat([df_dmid_real, df_dmid_fake], ignore_index=True)

    dmid_scores = df_dmid[dmid_models[0]].to_numpy()
    dmid_labels = df_dmid['label'].to_numpy()
    dmid_metrics = run_evaluation_metrics(dmid_labels, dmid_scores)
    print("DMimageDetection Metrics:")
    pprint.pprint(dmid_metrics)
    pd.DataFrame(dmid_metrics).to_csv('output/dmid/dmid_metrics.csv', index=False)

    ganid_scores = df_dmid[dmid_models[1]].to_numpy()
    ganid_metrics = run_evaluation_metrics(dmid_labels, ganid_scores)
    print("GANimageDetection Metrics:")
    pprint.pprint(ganid_metrics)
    pd.DataFrame(ganid_metrics).to_csv('output/dmid/ganid_metrics.csv', index=False)

    # ------------------- #
    # UniversalFaceDetect #
    # ------------------- #

    os.makedirs('output/ufd', exist_ok=True)

    ufd_true, ufd_score = run_universal_face_detect(
        arch='CLIP:ViT-L/14',
        ckpt='universalfakedetect/pretrained_weights/fc_weights.pth',
        result_folder='output/ufd',
        real_path='data/0_real/',
        fake_path='data/1_fake/'
    )

    ufd_metrics = run_evaluation_metrics(ufd_true, ufd_score)
    print("UniversalFaceDetect Metrics:")
    pprint.pprint(ufd_metrics)
    pd.DataFrame(ufd_metrics).to_csv('output/ufd/ufd_metrics.csv', index=False)

    # ---- #
    # DIRE #
    # ---- #

    # os.environ["CUDA_VISIBLE_DEVICES"]="0"
    # os.environ["NCCL_P2P_DISABLE"]="1"

    # # !mpiexec --allow-run-as-root python dire/guided-diffusion/compute_dire.py
    # #          --model_path /content/DIRE/256x256_diffusion_uncond.pt --batch_size 16 --num_samples 1000 \
    # #          --timestep_respacing ddim20 --use_ddim True \
    # #          --images_dir /content/data_1000 --recons_dir /content/recons --dire_dir /content/dires \
    # #          --attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 \
    # #          --dropout 0.1 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 \
    # #          --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True

    # --------- #
    # AEROBLADE #
    # --------- #

    distances = compute_distances(
        dirs=[Path('data/0_real'), Path('data/1_fake')],
        transforms=['clean'],
        repo_ids=[
            "CompVis/stable-diffusion-v1-1",
            "stabilityai/stable-diffusion-2-base",
            "kandinsky-community/kandinsky-2-1",
        ],
        distance_metrics=['lpips_vgg_2'],
        amount=None,
        reconstruction_root=Path('aeroblade_reconstructions'),
        seed=1,
        batch_size=4,
        num_workers=2,
    )

    aeroblade_pred_real = distances[distances['dir'] == 'data/0_real'].distance.to_numpy()
    aeroblade_pred_fake = distances[distances['dir'] == 'data/1_fake'].distance.to_numpy()
    aeroblade_scores = np.concatenate([aeroblade_pred_real, aeroblade_pred_fake])
    aeroblade_labels = np.concatenate([np.zeros_like(aeroblade_pred_real), np.ones_like(aeroblade_pred_fake)])

    aeroblade_metrics = run_evaluation_metrics(aeroblade_labels, aeroblade_scores)
    print("AEROBLADE Metrics:")
    pprint.pprint(aeroblade_metrics)
    pd.DataFrame(aeroblade_metrics).to_csv('output/aeroblade/aeroblade_metrics.csv', index=False)

    # ---- #
    # DRCT #
    # ---- #

if __name__ == "__main__":
    main()
