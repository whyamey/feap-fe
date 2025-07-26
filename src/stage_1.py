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
import csv


def hard_negative_mining_loss(embeddings, labels, margin_neg):
    sim_matrix = torch.matmul(embeddings, embeddings.t())

    batch_size = embeddings.size(0)
    labels_exp = labels.view(-1, 1)
    mask_neg = labels_exp != labels_exp.t()
    mask_neg.fill_diagonal_(False)
    if not mask_neg.any():
        return torch.tensor(0.0, device=embeddings.device)

    sim_matrix_masked = sim_matrix.clone()
    sim_matrix_masked[~mask_neg] = -2.0
    sim_hn, _ = sim_matrix_masked.max(dim=1)

    loss = torch.clamp(sim_hn - margin_neg, min=0).mean()
    return loss


def generate_and_plot_histograms(
    model, hist_dataset, device, epoch, save_dir="histograms"
):
    print("Generating and plotting histograms...")
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    hist_loader = DataLoader(
        hist_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True
    )

    all_embeddings = []
    all_labels = []
    all_paths = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(
            tqdm(hist_loader, desc="Computing Embeddings")
        ):
            images = images.to(device)
            embeddings = model(images)
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())

            batch_start = i * hist_loader.batch_size
            batch_end = min(batch_start + hist_loader.batch_size, len(hist_dataset))
            batch_paths = [
                hist_dataset.samples[idx][0] for idx in range(batch_start, batch_end)
            ]
            all_paths.extend(batch_paths)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    num_samples = len(all_labels)

    labels_matrix = all_labels.unsqueeze(0) == all_labels.unsqueeze(1)
    mask_same = torch.triu(labels_matrix, diagonal=1)
    mask_diff = torch.triu(~labels_matrix, diagonal=1)

    sim_matrix_cos = torch.matmul(all_embeddings, all_embeddings.T)
    same_class_sims_cos = sim_matrix_cos[mask_same].numpy()
    diff_class_sims_cos = sim_matrix_cos[mask_diff].numpy()

    binary_embeddings = torch.sign(all_embeddings)
    embedding_dim = binary_embeddings.shape[1]

    integer_ham_dist = 0.5 * (
        embedding_dim - torch.matmul(binary_embeddings, binary_embeddings.T)
    )
    sim_matrix_ham = integer_ham_dist / embedding_dim

    same_class_sims_ham = sim_matrix_ham[mask_same].numpy()
    diff_class_sims_ham = sim_matrix_ham[mask_diff].numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    ax1.hist(
        same_class_sims_cos,
        bins=100,
        alpha=0.6,
        density=True,
        label="Same Class",
        color="blue",
    )
    ax1.hist(
        diff_class_sims_cos,
        bins=100,
        alpha=0.6,
        density=True,
        label="Different Class",
        color="red",
    )
    ax1.set_title(f"Cosine Similarity Distribution - Epoch {epoch+1}")
    ax1.set_xlabel("Cosine Similarity")
    ax1.set_ylabel("Density")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    stats_text_cos = (
        f"Same Class (blue):\n  Mean: {same_class_sims_cos.mean():.4f}, Std: {same_class_sims_cos.std():.4f}\n"
        f"Different Class (red):\n  Mean: {diff_class_sims_cos.mean():.4f}, Std: {diff_class_sims_cos.std():.4f}"
    )
    ax1.text(
        0.02,
        0.98,
        stats_text_cos,
        transform=ax1.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax2.hist(
        same_class_sims_ham,
        bins=50,
        range=(0, 1),
        alpha=0.6,
        density=True,
        label="Same Class",
        color="blue",
    )
    ax2.hist(
        diff_class_sims_ham,
        bins=50,
        range=(0, 1),
        alpha=0.6,
        density=True,
        label="Different Class",
        color="red",
    )
    ax2.set_title(f"Fractional Hamming Distance Distribution - Epoch {epoch+1}")
    ax2.set_xlabel("Fractional Hamming Distance")
    ax2.set_ylabel("Density")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    stats_text_ham = (
        f"Same Class (blue):\n  Mean: {same_class_sims_ham.mean():.4f}, Std: {same_class_sims_ham.std():.4f}\n"
        f"Different Class (red):\n  Mean: {diff_class_sims_ham.mean():.4f}, Std: {diff_class_sims_ham.std():.4f}"
    )
    ax2.text(
        0.02,
        0.98,
        stats_text_ham,
        transform=ax2.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, f"combined_hist_epoch_{epoch+1:04d}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    return {
        "mean_cos_same": same_class_sims_cos.mean(),
        "std_cos_same": same_class_sims_cos.std(),
        "mean_cos_diff": diff_class_sims_cos.mean(),
        "std_cos_diff": diff_class_sims_cos.std(),
        "mean_ham_same": same_class_sims_ham.mean(),
        "std_ham_same": same_class_sims_ham.std(),
        "mean_ham_diff": diff_class_sims_ham.mean(),
        "std_ham_diff": diff_class_sims_ham.std(),
    }


class IrisDataset(Dataset):
    def __init__(
        self, root_dir: str, transform=None, exclude_classes=None, eye_type="L"
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.exclude_classes = exclude_classes or []
        self.eye_type = eye_type.upper()

        class_folders = [
            d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))
        ]
        class_folders = [d for d in class_folders if d not in self.exclude_classes]
        self.class_to_idx = {
            cls_name: idx for idx, cls_name in enumerate(sorted(class_folders))
        }

        for cls_name in class_folders:
            cls_dir = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                if img_name.endswith(f"_{self.eye_type}.bmp") or img_name.endswith(
                    f"_{self.eye_type}.png"
                ):
                    img_path = os.path.join(cls_dir, img_name)
                    self.samples.append((img_path, self.class_to_idx[cls_name]))

        print(
            f"Loaded dataset with {len(self.samples)} samples across {len(self.class_to_idx)} classes"
        )
        print(f"Eye type: {self.eye_type}")
        if self.exclude_classes:
            print(f"Excluded classes: {self.exclude_classes}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            image = self.transform(image)
        return image, label

    def get_num_classes(self):
        return len(self.class_to_idx)


class FlatIrisDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}

        image_files = [f for f in os.listdir(root_dir) if f.endswith(".png")]
        class_ids = set(f[:4] for f in image_files)
        self.class_to_idx = {
            cls_id: idx for idx, cls_id in enumerate(sorted(class_ids))
        }

        for img_name in image_files:
            img_path = os.path.join(root_dir, img_name)
            class_id = img_name[:4]
            self.samples.append((img_path, self.class_to_idx[class_id]))

        print(
            f"Loaded {len(self.samples)} samples across {len(self.class_to_idx)} classes from {root_dir}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            image = self.transform(image)
        return image, label

    def get_num_classes(self):
        return len(self.class_to_idx)


class IrisAugmentation:
    def __init__(self, img_dimension=256):
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
            image = self.clahe.apply(image)
            image[image < 10] = 0
        elif aug_type == 9:
            image = cv2.blur(image, (3, 3))

        return np.ascontiguousarray(image)


class AugmentedIrisDataset(Dataset):
    def __init__(self, base_dataset):
        self.dataset = base_dataset
        self.augmenter = IrisAugmentation()
        self.to_tensor = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.485], std=[0.229])]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_path, label = self.dataset.samples[idx]

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        aug_type = np.random.randint(0, 10)
        image = self.augmenter.augment(image, aug_type)

        image = np.ascontiguousarray(image)
        image = self.to_tensor(image)

        return image, label


class IrisNet(nn.Module):
    def __init__(
        self, num_classes: int, embedding_dim: int = 512, pretrained: bool = True
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)

        self.backbone.classifier = nn.Identity()

        self.embedding = nn.Sequential(
            nn.BatchNorm1d(1280),
            nn.Dropout(0.4),
            nn.Linear(1280, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, x):
        x = x.expand(-1, 3, -1, -1)

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

        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        loss = (one_hot * p_loss) + ((1.0 - one_hot) * n_loss)
        return loss.sum(dim=1).mean()


def train_iris_model(config, resume_training=False, checkpoint_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    csv_log_path = "training_log.csv"
    log_file_exists = os.path.exists(csv_log_path)
    csv_file = open(csv_log_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    header = [
        "epoch",
        "total_loss",
        "uce_loss",
        "hn_loss",
        "mean_cos_same",
        "std_cos_same",
        "mean_cos_diff",
        "std_cos_diff",
        "mean_ham_same",
        "std_ham_same",
        "mean_ham_diff",
        "std_ham_diff",
    ]
    if not log_file_exists or os.path.getsize(csv_log_path) == 0:
        csv_writer.writerow(header)

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229]),
        ]
    )

    exclude_classes = config.get("exclude_classes", [])

    train_dataset = IrisDataset(
        config["data_dir"],
        transform=None,
        exclude_classes=exclude_classes,
        eye_type="L",
    )
    train_dataset = AugmentedIrisDataset(train_dataset)

    hist_dataset = IrisDataset(
        config["data_dir"],
        transform=transform,
        exclude_classes=exclude_classes,
        eye_type="R",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    model = IrisNet(
        num_classes=train_dataset.dataset.get_num_classes(),
        embedding_dim=config["embedding_dim"],
        pretrained=config["use_pretrained"],
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
    if resume_training and checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}")

    scheduler = CosineAnnealingLR(
        optimizer, T_max=config["epochs"] - start_epoch, eta_min=1e-7
    )

    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        total_loss_epoch, uce_loss_epoch, hn_loss_epoch = 0, 0, 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]} - Training')

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            embeddings = model(images)
            uce_loss = criterion(embeddings, labels)

            hn_loss = hard_negative_mining_loss(
                embeddings, labels, margin_neg=config["hn_margin"]
            )
            total_loss = uce_loss + config["hn_weight"] * hn_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss_epoch += total_loss.item()
            uce_loss_epoch += uce_loss.item()
            hn_loss_epoch += hn_loss.item()
            pbar.set_postfix(
                {
                    "loss": total_loss.item(),
                    "uce": uce_loss.item(),
                    "hn": hn_loss.item(),
                }
            )

        avg_total_loss = total_loss_epoch / len(train_loader)
        avg_uce_loss = uce_loss_epoch / len(train_loader)
        avg_hn_loss = hn_loss_epoch / len(train_loader)

        print(f"\nEpoch {epoch+1} Summary:")
        print(
            f"  Avg Total Loss: {avg_total_loss:.4f} (UCE: {avg_uce_loss:.4f}, HN: {avg_hn_loss:.4f})"
        )

        scheduler.step()

        if (epoch + 1) % 1 == 0:
            hist_stats = generate_and_plot_histograms(
                model=model, hist_dataset=hist_dataset, device=device, epoch=epoch
            )

            log_row = {
                "total_loss": avg_total_loss,
                "uce_loss": avg_uce_loss,
                "hn_loss": avg_hn_loss,
                **hist_stats,
            }
            csv_writer.writerow([log_row.get(h, "") for h in header])
            csv_file.flush()

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": config,
            }
            checkpoint_name = os.path.join(
                checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt"
            )
            torch.save(checkpoint, checkpoint_name)
            print(f"Saved checkpoint to {checkpoint_name}")

    csv_file.close()
    print("Training finished.")


if __name__ == "__main__":
    config = {
        "data_dir": "/share/user/feap/iitd_segmented/",
        "exclude_classes": ["140", "53", "74", "94", "37", "38", "156", "019", "017"],
        "embedding_dim": 1024,
        "batch_size": 64,
        "epochs": 2000,
        "lr": 1e-4,
        "margin": 0.4,
        "scale": 64,
        "loss_weight": 1.0,
        "radius": 1.0,
        "use_pretrained": True,
        "hn_margin": 0.1,
        "hn_weight": 0.5,
    }

    train_iris_model(config, resume_training=False)
