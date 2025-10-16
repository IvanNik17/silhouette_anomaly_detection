import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from avenue_dataloader import AvenueMaskSequenceDataset
from model_lstm_unet import ConvLSTM_UNet_Predictor
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

def evaluate_model(model, dataset_root, model_path, label_path,
                   sequence_length=8, visualize_num=6, batch_size=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    gt_labels_full = np.load(label_path).squeeze()
    dataset = AvenueMaskSequenceDataset(dataset_root, sequence_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    start_idx = sequence_length - 1
    gt_labels = gt_labels_full[start_idx:start_idx + len(dataset)]

    all_scores, all_targets, all_preds, all_recons, all_last_inputs = [], [], [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):
            input_seq = batch["input_seq"].to(device)
            target_frame = batch["target_frame"].to(device)
            last_input = input_seq[:, -1]

            pred_logits, recon_logits = model(input_seq)
            pred_mask = torch.sigmoid(pred_logits)
            recon_mask = torch.sigmoid(recon_logits)

            for b in range(input_seq.size(0)):
                score = F.mse_loss(pred_mask[b], target_frame[b]).item()
                all_scores.append(score)
                all_targets.append(target_frame[b].cpu().squeeze().numpy())
                all_preds.append(pred_mask[b].cpu().squeeze().numpy())
                all_recons.append(recon_mask[b].cpu().squeeze().numpy())
                all_last_inputs.append(last_input[b].cpu().squeeze().numpy())

    scores = np.array(all_scores)
    norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    auc = roc_auc_score(gt_labels, norm_scores)
    ap = average_precision_score(gt_labels, norm_scores)
    print(f"Sequences: {len(norm_scores)}, ROC AUC: {auc:.4f}, AP: {ap:.4f}")

    plt.figure(figsize=(15, 5))
    plt.plot(norm_scores, label='Normalized Anomaly Score', color='tab:blue')
    plt.plot(gt_labels, label='Ground Truth', alpha=0.6, color='tab:orange')
    plt.xlabel('Sequence Index')
    plt.ylabel('Score / Label')
    plt.title('Anomaly Scores vs Ground Truth')
    plt.legend()
    plt.grid(True)
    plt.show()

    normal_idx = [i for i, l in enumerate(gt_labels) if l == 0]
    abnormal_idx = [i for i, l in enumerate(gt_labels) if l == 1]
    num_normals = min(visualize_num // 2, len(normal_idx))
    num_abnormals = min(visualize_num - num_normals, len(abnormal_idx))
    sample_idx = random.sample(normal_idx, num_normals) + random.sample(abnormal_idx, num_abnormals)

    titles = ["Last Input", "Reconstruction", "Target", "Prediction"]
    fig, axs = plt.subplots(len(sample_idx), 4, figsize=(12, 3 * len(sample_idx)))
    axs = np.array([axs]) if len(sample_idx) == 1 else np.reshape(axs, (len(sample_idx), 4))

    for i, idx in enumerate(sample_idx):
        color = 'red' if gt_labels[idx] == 1 else 'green'
        status = "ANOMALOUS" if gt_labels[idx] == 1 else "NORMAL"

        axs[i][0].imshow(all_last_inputs[idx], cmap='gray')
        axs[i][0].set_title(titles[0])
        axs[i][0].axis('off')
        axs[i][0].text(0.5, -0.15, f"GT: {status}", transform=axs[i][0].transAxes,
                       fontsize=10, color=color, ha='center', va='center',
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor=color, linewidth=2))

        axs[i][1].imshow(all_recons[idx], cmap='gray')
        axs[i][1].set_title(titles[1])
        axs[i][1].axis('off')

        axs[i][2].imshow(all_targets[idx], cmap='gray')
        axs[i][2].set_title(titles[2])
        axs[i][2].axis('off')

        axs[i][3].imshow(all_preds[idx], cmap='gray')
        axs[i][3].set_title(f"{titles[3]}\nScore={scores[idx]:.4f}")
        axs[i][3].axis('off')

    fig.suptitle(f"Prediction & Reconstruction (T={sequence_length})", y=1.02)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    dataset_name = ["avenue", "shanghai", "ped2", "ubnormal"]
    dataset_root = f"{dataset_name[2]}_masks"
    model_path = "mode_path"
    label_path = f"frame_labels_{dataset_name[2]}.npy"

    model = ConvLSTM_UNet_Predictor(base=32, z_dim=64, use_latent=False)
    evaluate_model(model, dataset_root, model_path, label_path, sequence_length=8, visualize_num=6, batch_size=1)
