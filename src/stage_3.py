import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import cv2
from torchvision import transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path


class IrisDataset(Dataset):
    def __init__(self, root_dir: str, transform=None, class_filter=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}

        class_folders = [
            d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))
        ]
        if class_filter is not None:
            class_folders = [d for d in class_folders if d in class_filter]

        self.class_to_idx = {
            cls_name: idx for idx, cls_name in enumerate(sorted(class_folders))
        }

        for cls_name in class_folders:
            cls_dir = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                if img_name.endswith(".png"):
                    img_path = os.path.join(cls_dir, img_name)
                    self.samples.append((img_path, self.class_to_idx[cls_name]))

        print(
            f"Loaded dataset with {len(self.samples)} samples across {len(self.class_to_idx)} classes"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)
        return image, label

    def get_num_classes(self):
        return len(self.class_to_idx)


class IrisNet(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int = 512):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.backbone = models.resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Identity()

        self.embedding = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        embedding = self.embedding(x)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding


class RefinementNet(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=2048, output_dim=512):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.layers = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

        self.temperature = nn.Parameter(torch.ones(1) * 20.0)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.input_dim == self.output_dim:
            identity = x
            out = self.layers(x)
            out = out + identity
        else:
            out = self.layers(x)
        return F.normalize(out, p=2, dim=1)


class RefinementLoss(nn.Module):
    def __init__(self, beta=0.1):
        super().__init__()
        self.beta = beta

    def forward(self, embeddings, labels, temperature):
        sim_matrix = torch.matmul(embeddings, embeddings.T)

        labels_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
        pos_mask = labels_matrix.float()
        neg_mask = (~labels_matrix).float()

        pos_mask.fill_diagonal_(0)
        neg_mask.fill_diagonal_(0)

        scaled_sim_matrix = sim_matrix * temperature

        pos_pairs = scaled_sim_matrix * pos_mask
        pos_loss = torch.sum(F.relu(1 - pos_pairs) ** 2 * pos_mask) / (
            pos_mask.sum() + 1e-6
        )

        neg_pairs = scaled_sim_matrix * neg_mask
        neg_mean = torch.sum(neg_pairs * neg_mask) / (neg_mask.sum() + 1e-6)
        neg_var = torch.sum(((neg_pairs - neg_mean) ** 2) * neg_mask) / (
            neg_mask.sum() + 1e-6
        )

        total_loss = pos_loss + torch.abs(neg_mean) + self.beta * neg_var

        return total_loss, {
            "pos_loss": pos_loss.item(),
            "neg_mean": neg_mean.item(),
            "neg_var": neg_var.item(),
            "temperature": temperature.item(),
        }


def plot_similarity_histogram(
    same_class_sims,
    diff_class_sims,
    embeddings,
    labels,
    epoch,
    save_dir="histograms_refined",
):
    plt.clf()
    plt.close("all")

    os.makedirs(save_dir, exist_ok=True)

    labels_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
    mask_same = torch.triu(labels_matrix, diagonal=1)
    mask_diff = torch.triu(~labels_matrix, diagonal=1)

    indices = torch.nonzero(torch.triu(torch.ones_like(labels_matrix), diagonal=1))
    same_indices = indices[mask_same[indices[:, 0], indices[:, 1]]]
    diff_indices = indices[mask_diff[indices[:, 0], indices[:, 1]]]

    binary = embeddings > 0

    hamming_matrix = (binary.unsqueeze(1) != binary.unsqueeze(0)).float().mean(dim=-1)

    same_hamming = hamming_matrix[mask_same].cpu().numpy()
    diff_hamming = hamming_matrix[mask_diff].cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

    ax1.hist(
        same_class_sims,
        bins=50,
        alpha=0.5,
        label=f"Same Class (μ={same_class_sims.mean():.3f}, σ={same_class_sims.std():.3f})",
        density=True,
        color="blue",
    )
    ax1.hist(
        diff_class_sims,
        bins=50,
        alpha=0.5,
        label=f"Different Class (μ={diff_class_sims.mean():.3f}, σ={diff_class_sims.std():.3f})",
        density=True,
        color="red",
    )

    ax1.set_title("Cosine Similarity Distribution")
    ax1.set_xlabel("Cosine Similarity")
    ax1.set_ylabel("Density")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.hist(
        same_hamming,
        bins=50,
        alpha=0.5,
        label=f"Same Class (μ={same_hamming.mean():.3f}, σ={same_hamming.std():.3f})",
        density=True,
        color="blue",
    )
    ax2.hist(
        diff_hamming,
        bins=50,
        alpha=0.5,
        label=f"Different Class (μ={diff_hamming.mean():.3f}, σ={diff_hamming.std():.3f})",
        density=True,
        color="red",
    )

    ax2.set_title("Normalized Hamming Distance Distribution")
    ax2.set_xlabel("Hamming Distance")
    ax2.set_ylabel("Density")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"Refined Embedding Distributions - Epoch {epoch+1}", y=1.02)
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, f"distribution_hist_epoch_{epoch+1:04d}.png"),
        bbox_inches="tight",
    )
    plt.close()


def extract_and_train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = {
        "train_dir": "./IITD_folders/IITD_folders_train_only/",
        "inference_dir": "./IITD_folders/IITD_folders_inference_only_folders/",
        "embedding_dim": 1024,
        "batch_size": 460,
        "epochs": 1000000,
        "lr": 1e-4,
        "beta": 0.1,
        "output_dim": 512,
    }

    os.makedirs("checkpoints_refined", exist_ok=True)

    class_folders = [
        d
        for d in os.listdir(config["train_dir"])
        if os.path.isdir(os.path.join(config["train_dir"], d))
    ]
    num_classes = len(class_folders)
    print(f"Found {num_classes} classes in training directory")

    original_model = IrisNet(
        num_classes=num_classes, embedding_dim=config["embedding_dim"]
    ).to(device)
    checkpoint = torch.load("stage_2.pt", map_location=device)
    original_model.load_state_dict(checkpoint["model_state_dict"])
    original_model.eval()

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    print("Extracting embeddings from training set...")
    train_dataset = IrisDataset(config["train_dir"], transform=transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    train_embeddings = []
    train_labels = []

    with torch.no_grad():
        for images, labels in tqdm(train_loader):
            images = images.to(device)
            embeddings = original_model(images)
            train_embeddings.append(embeddings.cpu())
            train_labels.append(labels)

    train_embeddings = torch.cat(train_embeddings, dim=0)
    train_labels = torch.cat(train_labels, dim=0)

    print("Extracting embeddings from histogram dataset (2000 random samples)...")
    hist_dataset = IrisDataset(config["inference_dir"], transform=transform)

    hist_loader = DataLoader(
        hist_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    hist_embeddings = []
    hist_labels = []

    with torch.no_grad():
        for images, labels in tqdm(hist_loader):
            images = images.to(device)
            embeddings = original_model(images)
            hist_embeddings.append(embeddings.cpu())
            hist_labels.append(labels)

    hist_embeddings = torch.cat(hist_embeddings, dim=0)
    hist_labels = torch.cat(hist_labels, dim=0)

    refinement_model = RefinementNet(
        input_dim=config["embedding_dim"],
        hidden_dim=2048,
        output_dim=config["output_dim"],
    ).to(device)

    criterion = RefinementLoss(beta=config["beta"]).to(device)

    optimizer = torch.optim.AdamW(
        refinement_model.parameters(), lr=config["lr"], weight_decay=0.01
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=1e-7)

    best_loss = float("inf")
    batch_size = 64

    for epoch in range(config["epochs"]):
        refinement_model.train()
        total_loss = 0

        perm = torch.randperm(len(train_embeddings))
        n_batches = len(train_embeddings) // batch_size

        pbar = tqdm(range(n_batches), desc=f'Epoch {epoch+1}/{config["epochs"]}')
        for i in pbar:
            idx = perm[i * batch_size : (i + 1) * batch_size]
            batch_embeddings = train_embeddings[idx].to(device)
            batch_labels = train_labels[idx].to(device)

            refined = refinement_model(batch_embeddings)
            loss, metrics = criterion(
                refined, batch_labels, refinement_model.temperature
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(refinement_model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(
                {
                    "loss": loss.item(),
                    "pos_loss": metrics["pos_loss"],
                    "neg_mean": metrics["neg_mean"],
                    "temp": metrics["temperature"],
                }
            )

        avg_loss = total_loss / n_batches

        if (epoch + 1) % 500 == 0:
            refinement_model.eval()
            with torch.no_grad():
                refined_hist = refinement_model(hist_embeddings.to(device))
                sim_matrix = torch.matmul(refined_hist, refined_hist.T)
                labels_matrix = hist_labels.unsqueeze(0) == hist_labels.unsqueeze(1)

                mask_same = torch.triu(labels_matrix, diagonal=1)
                mask_diff = torch.triu(~labels_matrix, diagonal=1)

                same_class_sims = sim_matrix[mask_same].cpu().numpy()
                diff_class_sims = sim_matrix[mask_diff].cpu().numpy()

                plot_similarity_histogram(
                    same_class_sims, diff_class_sims, refined_hist, hist_labels, epoch
                )

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": refinement_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": avg_loss,
                    },
                    f"checkpoints_refined/checkpoint_epoch_{epoch+1}.pt",
                )

        print(f"\nEpoch {epoch+1}")
        print(f"Training Loss: {avg_loss:.4f}")

        if (epoch + 1) % 500 == 0:
            print(f"Histogram Stats:")
            print(
                f"Same Class Mean: {same_class_sims.mean():.3f}, Std: {same_class_sims.std():.3f}"
            )
            print(
                f"Diff Class Mean: {diff_class_sims.mean():.3f}, Std: {diff_class_sims.std():.3f}"
            )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": refinement_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                },
                "best_refinement_model.pt",
            )
            print(f"Saved new best model with loss: {avg_loss:.4f}")

        scheduler.step()


if __name__ == "__main__":
    extract_and_train()
