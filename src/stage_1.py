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


class Unified_Cross_Entropy_Loss(nn.Module):
    def __init__(self, in_features, out_features, m=0.4, s=64, l=1.0, r=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        self.l = l
        self.r = r
        self.bias = Parameter(torch.FloatTensor(1))
        nn.init.constant_(self.bias, math.log(out_features * r * 10))
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.kaiming_normal_(
            self.weight, a=1, mode="fan_in", nonlinearity="leaky_relu"
        )

    def forward(self, input, label):
        cos_theta = F.linear(
            F.normalize(input, eps=1e-5), F.normalize(self.weight, eps=1e-5)
        )

        cos_m_theta_p = self.s * (cos_theta - self.m) - self.bias
        cos_m_theta_n = self.s * cos_theta - self.bias
        p_loss = torch.log(1 + torch.exp(-cos_m_theta_p.clamp(min=-self.s, max=self.s)))
        n_loss = (
            torch.log(1 + torch.exp(cos_m_theta_n.clamp(min=-self.s, max=self.s)))
            * self.l
        )

        one_hot = torch.zeros_like(cos_theta, dtype=torch.bool)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        loss = one_hot * p_loss + (~one_hot) * n_loss
        return loss.sum(dim=1).mean()


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
    config,
    resume_training=False,
    checkpoint_path="checkpoints/checkpoint_epoch_1000.pt",
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

    criterion = Unified_Cross_Entropy_Loss(
        in_features=config["embedding_dim"],
        out_features=train_dataset.dataset.get_num_classes(),
        m=config["margin"],
        s=config["scale"],
        l=config["loss_weight"],
        r=config["radius"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        [{"params": model.parameters()}, {"params": criterion.parameters()}],
        lr=config["lr"],
        weight_decay=0.01,
    )

    start_epoch = 0
    best_val_loss = float("inf")

    if resume_training:
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_loss = checkpoint["val_loss"]
            print(f"Resuming from epoch {start_epoch}")
        else:
            print(f"No checkpoint found at {checkpoint_path}")
            return

    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=1e-7)

    def save_checkpoint(epoch, model, optimizer, scheduler, val_loss, is_best=False):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
            "config": config,
        }

        if (epoch + 1) % 20 == 0:
            checkpoint_name = os.path.join(
                checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt"
            )
            torch.save(checkpoint, checkpoint_name)
            print(f"Saved periodic checkpoint to {checkpoint_name}")

        if is_best:
            best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_model_path)
            print(f"Saved best model with val_loss: {val_loss:.4f}")

    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]} - Training')

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            embeddings = model(images)
            loss = criterion(embeddings, labels)

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
                loss = criterion(embeddings, labels)
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

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        save_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            val_loss=val_loss,
            is_best=is_best,
        )

        scheduler.step()


if __name__ == "__main__":
    config = {
        "iitd_left_dir": "./IITD_folders/IITD_folders_train_only/",
        "iitd_right_dir": "./IITD_folders/IITD_folders_inference_only_folders/",
        "embedding_dim": 1024,
        "batch_size": 64,
        "epochs": 2000,
        "lr": 1e-4,
        "margin": 0.4,
        "scale": 64,
        "loss_weight": 1.0,
        "radius": 1.0,
    }

    train_iris_model(config, resume_training=False)
