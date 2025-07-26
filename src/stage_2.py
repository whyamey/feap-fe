import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torch.nn import Parameter
import math
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
from sklearn.model_selection import train_test_split
import json


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


class IrisAugmentation:
    def __init__(self, img_dimension=320):
        self.img_dimension = img_dimension
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def rotate_image(self, image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(
            image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR
        )
        return np.ascontiguousarray(result)

    def augment(self, image, aug_type):
        image = cv2.resize(image, (self.img_dimension, self.img_dimension))

        if aug_type == 0:
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            image = cv2.filter2D(image, -1, kernel)
        elif aug_type == 1:
            image = self.rotate_image(image, 30)
        elif aug_type == 2:
            image = self.rotate_image(image, 120)
        elif aug_type == 3:
            image = self.rotate_image(image, 320)
        elif aug_type == 4:
            image = self.rotate_image(image, 230)
        elif aug_type == 5:
            image = np.ascontiguousarray(np.flip(image, 0))
        elif aug_type == 6:
            image = np.ascontiguousarray(np.flip(image, 1))
        elif aug_type == 7:
            image = image.copy()
            image[0:70] = 0
        elif aug_type == 8:
            cl1 = self.clahe.apply(image[:, :, 0])
            cl1[cl1 < 10] = 0
            image = image.copy()
            image[:, :, 0] = cl1
            image[:, :, 1] = cl1
            image[:, :, 2] = cl1
        elif aug_type == 9:
            image = cv2.blur(image, (3, 3))

        return np.ascontiguousarray(image)


class AugmentedIrisDataset(Dataset):
    def __init__(self, base_dataset):
        self.dataset = base_dataset
        self.augmenter = IrisAugmentation()
        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_path, label = self.dataset.samples[idx]

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        aug_type = np.random.randint(0, 10)
        image = self.augmenter.augment(image, aug_type)

        image = np.ascontiguousarray(image)

        image = self.to_tensor(image)

        return image, label


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


class CircleLoss(nn.Module):
    def __init__(self, m: float, gamma: float, soft_margin: float = 0.1):
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_margin = soft_margin
        self.soft_plus = nn.Softplus()

    def forward(self, sp: torch.Tensor, sn: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sp: similarity to positive classes, shape [N, 1]
            sn: similarity to negative classes, shape [N, K-1]
        """
        ap = torch.clamp_min(-sp.detach() + 1 + self.m, min=0.0)
        an = torch.clamp_min(sn.detach() + self.m, min=0.0)

        delta_p = 1 - self.m + self.soft_margin * torch.sigmoid(sp - 0.8)
        delta_n = self.m - self.soft_margin * torch.sigmoid(-sn - 0.2)

        logit_p = -ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        std_reg = 0.1 * (torch.std(sp) + torch.std(sn))

        loss = self.soft_plus(
            torch.logsumexp(logit_n, dim=1) + torch.logsumexp(logit_p, dim=1)
        )

        return loss.mean() + std_reg


def get_similarity_matrix(embeddings: torch.Tensor, labels: torch.Tensor) -> tuple:
    """
    Calculate positive and negative similarities for Circle loss
    """
    n = embeddings.size(0)
    similarity_matrix = torch.matmul(embeddings, embeddings.T)

    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    mask_pos = labels_equal.triu(diagonal=1)
    mask_neg = (~labels_equal).triu(diagonal=1)

    pos_sims = []
    for i in range(n):
        pos_mask = mask_pos[i]
        if pos_mask.sum() > 0:
            pos_sim = similarity_matrix[i][pos_mask].mean().unsqueeze(0)
        else:
            pos_sim = torch.tensor([0.9], device=embeddings.device)
        pos_sims.append(pos_sim)

    sp = torch.stack(pos_sims).unsqueeze(1)

    neg_sims = []
    max_neg_count = mask_neg.sum(dim=1).max().item()

    for i in range(n):
        neg_mask = mask_neg[i]
        curr_negs = similarity_matrix[i][neg_mask]

        if len(curr_negs) < max_neg_count:
            padding = torch.full(
                (max_neg_count - len(curr_negs),), -1.0, device=embeddings.device
            )
            curr_negs = torch.cat([curr_negs, padding])
        elif len(curr_negs) > max_neg_count:
            curr_negs = curr_negs[:max_neg_count]

        neg_sims.append(curr_negs.unsqueeze(0))

    sn = torch.cat(neg_sims, dim=0)

    return sp, sn


def plot_similarity_histogram(
    same_class_sims, diff_class_sims, epoch, save_dir="histograms"
):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))

    plt.hist(
        same_class_sims,
        bins=50,
        alpha=0.5,
        label=f"Same Class (μ={same_class_sims.mean():.3f}, σ={same_class_sims.std():.3f})",
        density=True,
        color="blue",
    )
    plt.hist(
        diff_class_sims,
        bins=50,
        alpha=0.5,
        label=f"Different Class (μ={diff_class_sims.mean():.3f}, σ={diff_class_sims.std():.3f})",
        density=True,
        color="red",
    )

    plt.title(f"Embedding Similarity Distribution - Epoch {epoch+1}")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)

    stats_text = (
        f"Same Class:\n"
        f"  Mean: {same_class_sims.mean():.3f}\n"
        f"  Std: {same_class_sims.std():.3f}\n"
        f"Different Class:\n"
        f"  Mean: {diff_class_sims.mean():.3f}\n"
        f"  Std: {diff_class_sims.std():.3f}\n"
        f"  Abs Mean: {np.abs(diff_class_sims).mean():.3f}"
    )
    plt.text(
        0.02,
        0.98,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.savefig(os.path.join(save_dir, f"similarity_hist_epoch_{epoch+1}.png"))
    plt.close()


def create_train_val_split(dataset_path, val_ratio=0.2):
    """Create and save train/val split based on classes"""
    classes = [
        d
        for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ]
    train_classes, val_classes = train_test_split(
        classes, test_size=val_ratio, random_state=42
    )

    split_info = {
        "train_classes": sorted(train_classes),
        "val_classes": sorted(val_classes),
    }

    with open("dataset_split.json", "w") as f:
        json.dump(split_info, f, indent=4)

    return split_info


def train_iris_model(
    config, resume_training=False, checkpoint_path="models/stage_1.pt"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    if not os.path.exists("dataset_split.json"):
        split_info = create_train_val_split(config["iitd_left_dir"])
    else:
        with open("dataset_split.json", "r") as f:
            split_info = json.load(f)

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = IrisDataset(
        config["iitd_left_dir"],
        transform=transform,
        class_filter=split_info["train_classes"],
    )
    train_dataset = AugmentedIrisDataset(train_dataset)

    val_dataset = IrisDataset(
        config["iitd_left_dir"],
        transform=transform,
        class_filter=split_info["val_classes"],
    )

    hist_dataset = IrisDataset(config["iitd_right_dir"], transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    hist_loader = DataLoader(
        hist_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = IrisNet(
        num_classes=train_dataset.dataset.get_num_classes(),
        embedding_dim=config["embedding_dim"],
    ).to(device)

    criterion = CircleLoss(m=config["margin"], gamma=config["gamma"]).to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters()},
            {"params": criterion.parameters(), "lr": config["lr"] * 0.1},
        ],
        lr=config["lr"],
        weight_decay=0.01,
    )

    start_epoch = 0
    best_val_loss = float("inf")

    if resume_training and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        start_epoch = 0
        best_val_loss = float("inf")
        print(f"Loaded model weights only, starting fresh training with Circle Loss")

    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=1e-7)

    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]} - Training')

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            embeddings = model(images)
            sp, sn = get_similarity_matrix(embeddings, labels)
            loss = criterion(sp, sn)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(device), labels.to(device)
                embeddings = model(images)
                sp, sn = get_similarity_matrix(embeddings, labels)
                loss = criterion(sp, sn)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(f"\nEpoch {epoch+1}")
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")

        model.eval()
        with torch.no_grad():
            all_embeddings = []
            all_labels = []

            for images, labels in tqdm(hist_loader, desc="Computing Histogram"):
                images = images.to(device)
                embeddings = model(images)
                all_embeddings.append(embeddings.cpu())
                all_labels.append(labels)

            all_embeddings = torch.cat(all_embeddings, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            similarity_matrix = torch.matmul(all_embeddings, all_embeddings.T)

            labels_matrix = all_labels.unsqueeze(0) == all_labels.unsqueeze(1)
            mask_same = torch.triu(labels_matrix, diagonal=1)
            mask_diff = torch.triu(~labels_matrix, diagonal=1)

            same_class_sims = similarity_matrix[mask_same].numpy()
            diff_class_sims = similarity_matrix[mask_diff].numpy()

            plot_similarity_histogram(same_class_sims, diff_class_sims, epoch)

        if (epoch + 1) % 5 == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_loss,
                "config": config,
            }
            checkpoint_name = os.path.join(
                checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt"
            )
            torch.save(checkpoint, checkpoint_name)
            print(f"Saved periodic checkpoint to {checkpoint_name}")

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_loss,
                "config": config,
            }
            torch.save(checkpoint, os.path.join(checkpoint_dir, "best_model_circle.pt"))
            print(f"Saved best model with val_loss: {val_loss:.4f}")

        scheduler.step()


if __name__ == "__main__":
    config = {
        "iitd_left_dir": "./IITD_folders/IITD_folders_train_only/",
        "iitd_right_dir": "./IITD_folders/IITD_folders_inference_only_folders/",
        "embedding_dim": 1024,
        "batch_size": 64,
        "epochs": 1000,
        "lr": 2e-5,
        "margin": 0.25,
        "gamma": 80,
        "soft_margin": 0.05,
    }

    train_iris_model(config, resume_training=True)
