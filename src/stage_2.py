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
    model, hist_dataset, device, epoch, save_dir="histograms_primary"
):
    print(f"\nGenerating and plotting primary model histograms for epoch {epoch+1}...")
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    hist_loader = DataLoader(
        hist_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True
    )

    all_embeddings, all_labels, all_paths = [], [], []
    with torch.no_grad():
        for i, (images, labels) in enumerate(
            tqdm(hist_loader, desc="Primary Eval Embeddings")
        ):
            images = images.to(device)
            embeddings = model(images)
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())
            batch_start = i * hist_loader.batch_size
            batch_end = min(batch_start + hist_loader.batch_size, len(hist_dataset))
            all_paths.extend(
                [hist_dataset.samples[idx][0] for idx in range(batch_start, batch_end)]
            )

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

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
    ax1.set_title(f"Primary Cosine Sim - Epoch {epoch+1}")
    ax1.set_xlabel("Cosine Similarity")
    stats_text_cos = (
        f"Same: M={same_class_sims_cos.mean():.4f}, S={same_class_sims_cos.std():.4f}\n"
        f"Diff: M={diff_class_sims_cos.mean():.4f}, S={diff_class_sims_cos.std():.4f}"
    )
    ax1.text(
        0.02,
        0.98,
        stats_text_cos,
        transform=ax1.transAxes,
        va="top",
        bbox=dict(boxstyle="round", fc="white", alpha=0.8),
    )
    ax1.legend()

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
    ax2.set_title(f"Primary Hamming Dist - Epoch {epoch+1}")
    ax2.set_xlabel("Fractional Hamming Distance")
    stats_text_ham = (
        f"Same: M={same_class_sims_ham.mean():.4f}, S={same_class_sims_ham.std():.4f}\n"
        f"Diff: M={diff_class_sims_ham.mean():.4f}, S={diff_class_sims_ham.std():.4f}"
    )
    ax2.text(
        0.02,
        0.98,
        stats_text_ham,
        transform=ax2.transAxes,
        va="top",
        bbox=dict(boxstyle="round", fc="white", alpha=0.8),
    )
    ax2.legend()

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, f"primary_hist_epoch_{epoch+1:04d}.png"), dpi=300
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


def generate_and_plot_refiner_histograms(
    refiner_model,
    eval_embeddings,
    eval_paths,
    eval_precomputed,
    device,
    epoch,
    save_dir="histograms_refiner",
):
    print(f"\nGenerating and plotting refiner model histograms for epoch {epoch+1}...")
    os.makedirs(save_dir, exist_ok=True)
    refiner_model.eval()

    with torch.no_grad():
        refined_embeddings = refiner_model(eval_embeddings.to(device)).cpu()

    binary_embeddings = torch.sign(refined_embeddings)
    embedding_dim = binary_embeddings.shape[1]
    dot_product_matrix = torch.matmul(binary_embeddings, binary_embeddings.t())
    ham_dist_matrix = 0.5 * (embedding_dim - dot_product_matrix) / embedding_dim

    mask_same = eval_precomputed["mask_same"]
    mask_diff = eval_precomputed["mask_diff"]
    valid_diff_indices = eval_precomputed["valid_diff_indices"]

    same_class_dists = ham_dist_matrix[mask_same].numpy()
    diff_class_dists = ham_dist_matrix[mask_diff].numpy()

    plt.figure(figsize=(12, 8))
    plt.hist(
        same_class_dists,
        bins=50,
        range=(0, 1),
        alpha=0.7,
        density=True,
        label="Same Class",
    )
    plt.hist(
        diff_class_dists,
        bins=50,
        range=(0, 1),
        alpha=0.7,
        density=True,
        label="Different Class",
    )
    plt.title(f"Refined Hamming Distance - Epoch {epoch+1}")
    plt.xlabel("Fractional Hamming Distance")
    plt.ylabel("Density")

    stats_text = (
        f"Same Class (blue):\n  Mean: {same_class_dists.mean():.4f}, Std: {same_class_dists.std():.4f}\n"
        f"Different Class (red):\n  Mean: {diff_class_dists.mean():.4f}, Std: {diff_class_dists.std():.4f}"
    )
    plt.text(
        0.98,
        0.98,
        stats_text,
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(
        os.path.join(save_dir, f"refiner_hist_epoch_{epoch+1:04d}.png"), dpi=300
    )
    plt.close()

    return {
        "mean_ham_same_refined": same_class_dists.mean(),
        "std_ham_same_refined": same_class_dists.std(),
        "mean_ham_diff_refined": diff_class_dists.mean(),
        "std_ham_diff_refined": diff_class_dists.std(),
    }


class IrisDataset(Dataset):
    def __init__(self, root_dir, transform=None, eye_type="L", exclude_classes=None):
        self.root_dir = root_dir
        self.transform = transform
        self.eye_type = eye_type.upper()
        self.exclude_classes = exclude_classes or []
        self.samples = []
        self.class_to_idx = {}

        class_folders = sorted(
            [
                d
                for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
            ]
        )

        original_class_count = len(class_folders)
        class_folders = [d for d in class_folders if d not in self.exclude_classes]

        self.class_to_idx = {name: i for i, name in enumerate(class_folders)}

        for cls_name in class_folders:
            cls_dir = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                if img_name.endswith(f"_{self.eye_type}.bmp") or img_name.endswith(
                    f"_{self.eye_type}.png"
                ):
                    self.samples.append(
                        (os.path.join(cls_dir, img_name), self.class_to_idx[cls_name])
                    )

        print(
            f"Loaded {self.eye_type}-eye dataset: {len(self.samples)} samples from {len(self.class_to_idx)} classes."
        )
        if self.exclude_classes:
            print(
                f"  -> Excluded {original_class_count - len(class_folders)} classes: {self.exclude_classes}"
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
    def __init__(self, dim=256):
        self.dim = dim
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def rotate(self, img, angle):
        center = tuple(np.array(img.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        return np.ascontiguousarray(
            cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
        )

    def augment(self, img):
        img = cv2.resize(img, (self.dim, self.dim))
        aug_type = np.random.randint(0, 10)
        if aug_type == 0:
            img = cv2.filter2D(
                img, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            )
        elif aug_type == 1:
            img = self.rotate(img, 30)
        elif aug_type == 2:
            img = self.rotate(img, 120)
        elif aug_type == 5:
            img = np.ascontiguousarray(np.flip(img, 0))
        elif aug_type == 6:
            img = np.ascontiguousarray(np.flip(img, 1))
        elif aug_type == 8:
            img = self.clahe.apply(img)
        elif aug_type == 9:
            img = cv2.blur(img, (3, 3))
        return np.ascontiguousarray(img)


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
        img_path, label = self.dataset.samples[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = self.augmenter.augment(image)
        return self.to_tensor(image), label


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class IrisNet(nn.Module):
    def __init__(self, num_classes, embedding_dim=512, pretrained=True):
        super().__init__()
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
        embedding = self.embedding(torch.flatten(x, 1))
        return F.normalize(embedding, p=2, dim=1)


class Unified_Cross_Entropy_Loss(nn.Module):
    def __init__(self, in_features, out_features, m=0.4, s=64):
        super().__init__()
        self.in_features, self.out_features, self.m, self.s = (
            in_features,
            out_features,
            m,
            s,
        )
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.kaiming_normal_(
            self.weight, a=1, mode="fan_in", nonlinearity="leaky_relu"
        )

    def forward(self, input, label):
        cos_theta = F.linear(
            F.normalize(input, eps=1e-5), F.normalize(self.weight, eps=1e-5)
        )
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        m_hot = self.m * one_hot
        phi = cos_theta - m_hot
        loss = F.cross_entropy(self.s * phi, label)
        return loss


class AttentionRefiner(nn.Module):
    def __init__(self, input_dim=1024, output_dim=512):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        hidden_squeeze_dim = input_dim // 16
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_squeeze_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_squeeze_dim, input_dim),
            nn.Sigmoid(),
        )

        self.residual_block = nn.Sequential(
            nn.Linear(input_dim, output_dim // 2),
            nn.GELU(),
            nn.Linear(output_dim // 2, output_dim),
        )

        if self.input_dim != self.output_dim:
            self.projection_shortcut = nn.Linear(input_dim, output_dim, bias=False)

        self.final_norm = nn.LayerNorm(output_dim)
        self.final_act = nn.Tanh()

    def forward(self, x):
        identity = x
        attention_weights = self.attention(x)
        x_att = x * attention_weights
        residual = self.residual_block(x_att)

        if self.input_dim != self.output_dim:
            identity = self.projection_shortcut(identity)

        refined_output = self.final_norm(identity + residual)
        return self.final_act(refined_output)


class HammingRefinementLoss(nn.Module):
    def __init__(self, lambda_neg=1.0, target_dist=0.5):
        super().__init__()
        self.lambda_neg = lambda_neg
        self.target_dist = target_dist

    def forward(self, refined_embeddings, labels):
        soft_binary_embeddings = refined_embeddings
        dot_product = torch.matmul(soft_binary_embeddings, soft_binary_embeddings.t())

        embedding_dim = soft_binary_embeddings.shape[1]

        normalized_embeds = F.normalize(soft_binary_embeddings, p=2, dim=1)
        dot_product_norm = torch.matmul(normalized_embeds, normalized_embeds.t())
        ham_dist = 0.5 * (1.0 - dot_product_norm)

        labels_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask_pos = torch.triu(labels_matrix, diagonal=1)
        mask_neg = torch.triu(~labels_matrix, diagonal=1)

        pos_dists = ham_dist[mask_pos]
        neg_dists = ham_dist[mask_neg]

        loss_pos = pos_dists.mean() if pos_dists.numel() > 0 else 0.0

        loss_neg = (
            ((neg_dists - self.target_dist) ** 2).mean()
            if neg_dists.numel() > 0
            else 0.0
        )

        return loss_pos + self.lambda_neg * loss_neg, loss_pos, loss_neg


def run_full_training_pipeline(
    primary_config, refiner_config, primary_checkpoint_path=None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    primary_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229]),
        ]
    )
    exclude_list = primary_config.get("exclude_classes", [])
    train_dataset_L = IrisDataset(
        primary_config["data_dir"],
        transform=primary_transform,
        eye_type="L",
        exclude_classes=exclude_list,
    )
    hist_dataset_R = IrisDataset(
        primary_config["data_dir"],
        transform=primary_transform,
        eye_type="R",
        exclude_classes=exclude_list,
    )
    num_classes = train_dataset_L.get_num_classes()

    model = IrisNet(
        num_classes=num_classes,
        embedding_dim=primary_config["embedding_dim"],
        pretrained=primary_config["use_pretrained"],
    ).to(device)

    if primary_checkpoint_path and os.path.exists(primary_checkpoint_path):
        print("\n" + "=" * 50)
        print("    PART 1: Skipping Primary Training - Loading Model")
        print("=" * 50 + "\n")
        checkpoint = torch.load(primary_checkpoint_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print(f"Primary model loaded successfully from {primary_checkpoint_path}")
    else:
        print(
            "Primary checkpoint not found. Please provide a valid path to skip training."
        )
        return

    print("\n" + "=" * 50)
    print("      PART 2: Extracting Embeddings for Refiner")
    print("=" * 50 + "\n")
    model.eval()

    def extract_embeddings_from_dataset(dataset_to_extract):
        loader = DataLoader(
            dataset_to_extract,
            batch_size=primary_config["batch_size"] * 2,
            shuffle=False,
            num_workers=4,
        )
        embeddings_list, labels_list, paths_list = [], [], []
        with torch.no_grad():
            for images, labels in tqdm(
                loader, desc=f"Extracting {dataset_to_extract.eye_type}-eye embeddings"
            ):
                embeddings_list.append(model(images.to(device)).cpu())
                labels_list.append(labels)
        paths_list = [s[0] for s in dataset_to_extract.samples]
        return torch.cat(embeddings_list), torch.cat(labels_list), paths_list

    train_embeddings_L, train_labels_L, _ = extract_embeddings_from_dataset(
        train_dataset_L
    )
    eval_embeddings_R, eval_labels_R, eval_paths_R = extract_embeddings_from_dataset(
        hist_dataset_R
    )
    print(
        f"Extracted {len(train_embeddings_L)} train embeddings (L) and {len(eval_embeddings_R)} eval embeddings (R)."
    )

    print("\n" + "=" * 50)
    print("        PART 3: Training RefinementNet")
    print("=" * 50 + "\n")

    os.makedirs("refiner_checkpoints", exist_ok=True)

    print("Pre-computing static evaluation data (masks and indices)...")
    eval_labels_matrix = eval_labels_R.unsqueeze(0) == eval_labels_R.unsqueeze(1)
    eval_mask_same = torch.triu(eval_labels_matrix, diagonal=1)
    eval_mask_diff = torch.triu(~eval_labels_matrix, diagonal=1)
    num_eval_samples = len(eval_labels_R)
    diff_indices = torch.triu_indices(num_eval_samples, num_eval_samples, offset=1)
    diff_mask_flat = eval_mask_diff[diff_indices[0], diff_indices[1]]
    valid_diff_indices = diff_indices[:, diff_mask_flat]
    eval_precomputed = {
        "mask_same": eval_mask_same,
        "mask_diff": eval_mask_diff,
        "valid_diff_indices": valid_diff_indices,
    }
    print("Pre-computation complete.")

    refiner_train_dataset = EmbeddingDataset(train_embeddings_L, train_labels_L)
    refiner_train_loader = DataLoader(
        refiner_train_dataset, batch_size=refiner_config["batch_size"], shuffle=True
    )

    refiner_model = AttentionRefiner(
        input_dim=primary_config["embedding_dim"],
        output_dim=refiner_config["refined_dim"],
    ).to(device)

    refiner_criterion = HammingRefinementLoss(
        lambda_neg=refiner_config["lambda_neg"],
        target_dist=refiner_config["target_dist"],
    )
    refiner_optimizer = torch.optim.AdamW(
        refiner_model.parameters(), lr=refiner_config["lr"]
    )
    refiner_scheduler = CosineAnnealingLR(
        refiner_optimizer, T_max=refiner_config["epochs"], eta_min=1e-7
    )

    refiner_csv_path = "refiner_log.csv"
    with open(refiner_csv_path, "w", newline="") as refiner_csv_file:
        refiner_csv_writer = csv.writer(refiner_csv_file)
        refiner_header = [
            "epoch",
            "total_loss",
            "pos_pair_loss",
            "neg_pair_loss",
            "mean_ham_same_refined",
            "std_ham_same_refined",
            "mean_ham_diff_refined",
            "std_ham_diff_refined",
        ]
        refiner_csv_writer.writerow(refiner_header)
        for epoch in range(refiner_config["epochs"]):
            refiner_model.train()
            total_loss_epoch, pos_loss_epoch, neg_loss_epoch = 0, 0, 0
            pbar = tqdm(
                refiner_train_loader,
                desc=f'Refiner Epoch {epoch+1}/{refiner_config["epochs"]}',
            )
            for base_embeds, labels in pbar:
                base_embeds, labels = base_embeds.to(device), labels.to(device)
                refiner_optimizer.zero_grad()
                refined_embeds = refiner_model(base_embeds)
                loss, loss_p, loss_n = refiner_criterion(refined_embeds, labels)
                loss.backward()
                refiner_optimizer.step()
                total_loss_epoch += loss.item()
                pos_loss_epoch += (
                    loss_p.item() if isinstance(loss_p, torch.Tensor) else loss_p
                )
                neg_loss_epoch += (
                    loss_n.item() if isinstance(loss_n, torch.Tensor) else loss_n
                )
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

            avg_total_loss = total_loss_epoch / len(refiner_train_loader)
            avg_pos_loss = pos_loss_epoch / len(refiner_train_loader)
            avg_neg_loss = neg_loss_epoch / len(refiner_train_loader)
            print(
                f"\nEpoch {epoch+1} Training Summary: Avg Total Loss: {avg_total_loss:.4f}"
            )

            refiner_scheduler.step()

            if (epoch + 1) % refiner_config["eval_frequency"] == 0 or (
                epoch + 1
            ) == refiner_config["epochs"]:
                refiner_stats = generate_and_plot_refiner_histograms(
                    refiner_model,
                    eval_embeddings_R,
                    eval_paths_R,
                    eval_precomputed,
                    device,
                    epoch,
                )

                log_row = {
                    "epoch": epoch + 1,
                    "total_loss": avg_total_loss,
                    "pos_pair_loss": avg_pos_loss,
                    "neg_pair_loss": avg_neg_loss,
                    **refiner_stats,
                }
                refiner_csv_writer.writerow(
                    [log_row.get(h, "") for h in refiner_header]
                )
                refiner_csv_file.flush()
                print(
                    f"  -> Full evaluation for epoch {epoch+1} complete and logged to CSV."
                )

                checkpoint_path = os.path.join(
                    "refiner_checkpoints", f"refiner_checkpoint_epoch_{epoch+1}.pt"
                )
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": refiner_model.state_dict(),
                        "optimizer_state_dict": refiner_optimizer.state_dict(),
                        "scheduler_state_dict": refiner_scheduler.state_dict(),
                    },
                    checkpoint_path,
                )
                print(f"  -> Saved refiner checkpoint to {checkpoint_path}")

    torch.save(refiner_model.state_dict(), "refinernet_final.pt")
    print(
        "Refinement network training finished. Final model saved to refinernet_final.pt"
    )


if __name__ == "__main__":
    PRIMARY_CHECKPOINT_TO_LOAD = "checkpoint_stage_one.pt"

    primary_config = {
        "data_dir": "/share/user/feap/iitd_segmented/",
        "embedding_dim": 1024,
        "batch_size": 64,
        "epochs": 150,
        "lr": 1e-4,
        "margin": 0.4,
        "scale": 64,
        "use_pretrained": True,
        "hn_margin": 0.1,
        "hn_weight": 0.5,
        "exclude_classes": [
            "017",
            "019",
            "037",
            "038",
            "053",
            "074",
            "094",
            "140",
            "156",
        ],
        "eval_frequency": 30,
    }

    refiner_config = {
        "refined_dim": 512,
        "epochs": 15000,
        "batch_size": 2048,
        "lr": 1e-4,
        "lambda_neg": 12.0,
        "target_dist": 0.5,
        "eval_frequency": 100,
    }

    run_full_training_pipeline(
        primary_config,
        refiner_config,
        primary_checkpoint_path=PRIMARY_CHECKPOINT_TO_LOAD,
    )
