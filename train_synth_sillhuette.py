import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

from synth_dataloader import SyntheticSequenceDataset
from model_lstm_unet import ConvLSTM_UNet_Predictor

class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32).view(1,1,3,3)
        self.sobel_y = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32).view(1,1,3,3)

    def forward(self, pred, target):
        device = pred.device
        sobel_x = self.sobel_x.to(device)
        sobel_y = self.sobel_y.to(device)
        grad_pred_x = F.conv2d(pred, sobel_x, padding=1)
        grad_pred_y = F.conv2d(pred, sobel_y, padding=1)
        grad_target_x = F.conv2d(target, sobel_x, padding=1)
        grad_target_y = F.conv2d(target, sobel_y, padding=1)
        grad_diff = (grad_pred_x - grad_target_x)**2 + (grad_pred_y - grad_target_y)**2
        return grad_diff.mean()

def combined_pred_loss(pred_logits, target, alpha=0.5):
    bce = F.binary_cross_entropy_with_logits(pred_logits, target)
    pred = torch.sigmoid(pred_logits)
    boundary = BoundaryLoss()(pred, target)
    return bce + alpha * boundary

def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch
    }, path)
    print(f"✅ Saved checkpoint: {path}")

def train_model(
    data_root,
    output_dir="outputs",
    sequence_length=8,
    batch_size=16,
    num_epochs=50,
    lr=1e-4,
    device="cuda" if torch.cuda.is_available() else "cpu",
    save_every=10,
    recon_weight=0.8,
    boundary_alpha=0.5
):
    os.makedirs(output_dir, exist_ok=True)
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "logs.csv")

    dataset = SyntheticSequenceDataset(
        root_dir=data_root,
        sequence_length=sequence_length,
        use_augmentation=False
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    print(f"Total training samples: {len(dataset)}")
    print(f"Using device: {device}")

    model = ConvLSTM_UNet_Predictor(base=32, use_latent=False).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_loss = float("inf")

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "pred_loss", "recon_loss", "total_loss"])

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_pred_loss = 0.0
        epoch_recon_loss = 0.0
        total_batches = len(dataloader)
        pbar = tqdm(dataloader, desc=f"Epoch [{epoch}/{num_epochs}]", unit="batch")

        for batch in pbar:
            input_seq = batch["input_seq"].to(device)
            target_frame = batch["target_frame"].to(device)
            optimizer.zero_grad()

            pred_logits, recon_logits = model(input_seq)
            loss_pred = combined_pred_loss(pred_logits, target_frame, alpha=boundary_alpha)
            last_input_frame = input_seq[:, -1]
            loss_recon = F.binary_cross_entropy_with_logits(recon_logits, last_input_frame) * recon_weight
            total_loss = loss_pred + loss_recon

            total_loss.backward()
            optimizer.step()

            epoch_pred_loss += loss_pred.item()
            epoch_recon_loss += loss_recon.item()

            pbar.set_postfix({
                "pred_loss": f"{loss_pred.item():.4f}",
                "recon_loss": f"{loss_recon.item():.4f}",
                "total_loss": f"{total_loss.item():.4f}"
            })

        avg_pred_loss = epoch_pred_loss / total_batches
        avg_recon_loss = epoch_recon_loss / total_batches
        avg_total_loss = avg_pred_loss + avg_recon_loss

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_pred_loss, avg_recon_loss, avg_total_loss])

        if epoch % save_every == 0:
            save_checkpoint(model, optimizer, epoch, os.path.join(ckpt_dir, f"model_epoch_{epoch}.pt"))

        if avg_total_loss < best_loss:
            best_loss = avg_total_loss
            save_checkpoint(model, optimizer, epoch, os.path.join(ckpt_dir, "best_model.pt"))

        print(f"Epoch {epoch} | Pred: {avg_pred_loss:.4f} | Recon: {avg_recon_loss:.4f} | Total: {avg_total_loss:.4f}")

    print("Training complete")
    print(f"Best model loss: {best_loss:.4f}")


if __name__ == "__main__":
    train_model(
        data_root="synth\data\to\load",
        output_dir="model\save\dir",
        sequence_length=8,
        batch_size=8,
        num_epochs=50,
        lr=1e-4
    )
