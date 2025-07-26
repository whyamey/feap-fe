import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from torchvision import transforms
import cv2
import shutil
from torchvision import transforms, models
from pathlib import Path


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

    def forward(self, x):
        if self.input_dim == self.output_dim:
            identity = x
            out = self.layers(x)
            out = out + identity
        else:
            out = self.layers(x)
        return F.normalize(out, p=2, dim=1)


def compute_similarity_stats_batched(embeddings, labels, batch_size=1000):
    """Compute similarity statistics in batches to avoid memory issues"""
    device = embeddings.device
    n = len(embeddings)
    same_class_sims = []
    diff_class_sims = []

    for i in tqdm(range(0, n, batch_size), desc="Computing similarities"):
        batch_end = min(i + batch_size, n)
        batch_embeddings = embeddings[i:batch_end]
        batch_labels = labels[i:batch_end]

        sim_matrix = torch.matmul(batch_embeddings, embeddings.T)

        labels_matrix = batch_labels.unsqueeze(1) == labels.unsqueeze(0)

        for j in range(batch_end - i):
            row_sims = sim_matrix[j]
            row_labels = labels_matrix[j]

            if i == 0:
                row_sims = torch.cat([row_sims[:j], row_sims[j + 1 :]])
                row_labels = torch.cat([row_labels[:j], row_labels[j + 1 :]])

            same_class_sims.append(row_sims[row_labels].cpu())
            diff_class_sims.append(row_sims[~row_labels].cpu())

        del sim_matrix, labels_matrix
        torch.cuda.empty_cache()

    same_class_sims = torch.cat(same_class_sims).numpy()
    diff_class_sims = torch.cat(diff_class_sims).numpy()

    return same_class_sims, diff_class_sims


def compute_hamming_stats_batched(embeddings, labels, batch_size=100):
    """Compute Hamming distance statistics in batches, processing one row at a time"""
    device = embeddings.device
    n = len(embeddings)
    same_class_hamming = []
    diff_class_hamming = []

    binary_embeddings = (embeddings > 0).float()

    for i in tqdm(range(n), desc="Computing Hamming distances"):
        current_embedding = binary_embeddings[i : i + 1]
        current_label = labels[i]

        for j in range(0, n, batch_size):
            batch_end = min(j + batch_size, n)
            if batch_end <= i:
                continue

            if j == 0 and i < batch_end:
                batch_embeddings = binary_embeddings[i + 1 : batch_end]
                batch_labels = labels[i + 1 : batch_end]
            else:
                batch_embeddings = binary_embeddings[j:batch_end]
                batch_labels = labels[j:batch_end]

            if len(batch_embeddings) == 0:
                continue

            differences = (current_embedding != batch_embeddings).float().mean(dim=-1)
            same_class_mask = current_label == batch_labels

            same_class_hamming.extend(differences[same_class_mask].cpu().tolist())
            diff_class_hamming.extend(differences[~same_class_mask].cpu().tolist())

            del differences, same_class_mask
            torch.cuda.empty_cache()

    return np.array(same_class_hamming), np.array(diff_class_hamming)


def plot_similarity_histogram(
    same_class_sims, diff_class_sims, same_class_hamming, diff_class_hamming, save_path
):
    plt.clf()
    plt.close("all")

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
        same_class_hamming,
        bins=50,
        alpha=0.5,
        label=f"Same Class (μ={same_class_hamming.mean():.3f}, σ={same_class_hamming.std():.3f})",
        density=True,
        color="blue",
    )
    ax2.hist(
        diff_class_hamming,
        bins=50,
        alpha=0.5,
        label=f"Different Class (μ={diff_class_hamming.mean():.3f}, σ={diff_class_hamming.std():.3f})",
        density=True,
        color="red",
    )

    ax2.set_title("Normalized Hamming Distance Distribution")
    ax2.set_xlabel("Hamming Distance")
    ax2.set_ylabel("Density")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Final Embedding Distributions", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def save_embeddings_by_class(embeddings_dict, base_output_dir):
    """Save embeddings organized by class in both binary and float formats"""
    float_dir = os.path.join(base_output_dir, "iitd_embed_cosine")
    binary_dir = os.path.join(base_output_dir, "iitd_embed_hamming")

    for class_id, class_data in embeddings_dict.items():
        float_class_dir = os.path.join(float_dir, class_id)
        binary_class_dir = os.path.join(binary_dir, class_id)
        os.makedirs(float_class_dir, exist_ok=True)
        os.makedirs(binary_class_dir, exist_ok=True)

        for file_name, embedding in class_data.items():
            float_embedding = embedding.cpu().numpy()
            binary_embedding = (float_embedding > 0).astype(int)

            float_path = os.path.join(float_class_dir, file_name + ".txt")
            with open(float_path, "w") as f:
                f.write(",".join(map(str, float_embedding)) + ",")

            binary_path = os.path.join(binary_class_dir, file_name + ".txt")
            with open(binary_path, "w") as f:
                f.write(",".join(map(str, binary_embedding)) + ",")


def rotate_image(image, angle):
    """Rotate an image by the given angle in degrees"""
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image


def apply_gamma_correction(image, gamma=1.2):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(
        np.uint8
    )
    return cv2.LUT(image, table)


def adjust_channel_intensity(image, channel_idx=0, factor=1.05):
    result = image.copy().astype(np.float32)
    result[:, :, channel_idx] = np.clip(result[:, :, channel_idx] * factor, 0, 255)
    return result.astype(np.uint8)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = {
        "data_dir": "./IITD_folders/IITD_folders_inference_only_folders/",
        "embedding_dim": 1024,
        "output_dim": 512,
        "batch_size": 32,
        "output_dir": "./iitd_embeddings_421000_nvote",
    }

    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs("final_histograms", exist_ok=True)

    classes = [
        d
        for d in os.listdir(config["data_dir"])
        if os.path.isdir(os.path.join(config["data_dir"], d))
    ]
    num_classes = len(classes)
    print(f"Found {num_classes} classes in inference directory")

    original_model = IrisNet(
        num_classes=num_classes, embedding_dim=config["embedding_dim"]
    ).to(device)
    refinement_model = RefinementNet(
        input_dim=config["embedding_dim"],
        hidden_dim=2048,
        output_dim=config["output_dim"],
    ).to(device)

    original_checkpoint = torch.load("stage_2.pt", map_location=device)
    refinement_checkpoint = torch.load("stage_3.pt", map_location=device)

    original_model.load_state_dict(original_checkpoint["model_state_dict"])
    refinement_model.load_state_dict(refinement_checkpoint["model_state_dict"])

    original_model.eval()
    refinement_model.eval()

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    embeddings_dict = {}
    all_embeddings = []
    all_labels = []
    class_to_idx = {}

    print("Processing images and generating embeddings...")
    for class_id in tqdm(os.listdir(config["data_dir"])):
        class_dir = os.path.join(config["data_dir"], class_id)
        if not os.path.isdir(class_dir):
            continue

        if class_id not in class_to_idx:
            class_to_idx[class_id] = len(class_to_idx)

        for img_name in os.listdir(class_dir):
            if not img_name.endswith(".png"):
                continue

            file_name = os.path.splitext(img_name)[0]

            img_path = os.path.join(class_dir, img_name)
            image = cv2.imread(img_path)
            if image is None:
                print(f"Failed to load image: {img_path}")
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image_variations = []

            image_variations.append(image.copy())

            image_variations.append(cv2.flip(image, 1))

            image_variations.append(rotate_image(image, 5))
            image_variations.append(rotate_image(image, -5))
            image_variations.append(rotate_image(cv2.flip(image, 1), 5))
            image_variations.append(rotate_image(cv2.flip(image, 1), -5))

            bright_up = cv2.convertScaleAbs(image, alpha=1.1, beta=0)
            bright_up_flip = cv2.convertScaleAbs(cv2.flip(image, 1), alpha=1.1, beta=0)
            bright_down = cv2.convertScaleAbs(image, alpha=0.9, beta=0)
            bright_down_flip = cv2.convertScaleAbs(
                cv2.flip(image, 1), alpha=0.9, beta=0
            )
            image_variations.extend(
                [bright_up, bright_down, bright_down_flip, bright_up_flip]
            )
            slight_blur = cv2.GaussianBlur(image, (3, 3), 0.5)
            slight_blur_flip = cv2.GaussianBlur(cv2.flip(image, 1), (3, 3), 0.5)
            image_variations.append(slight_blur)
            image_variations.append(slight_blur_flip)

            height, width = image.shape[:2]
            src_points = np.float32(
                [[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]]
            )
            offset = width * 0.015
            dst_points = np.float32(
                [
                    [0 + offset, 0],
                    [width - 1 - offset, 0],
                    [0, height - 1],
                    [width - 1, height - 1],
                ]
            )
            transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            warped_img = cv2.warpPerspective(image, transform_matrix, (width, height))
            warped_img_f = cv2.warpPerspective(
                cv2.flip(image, 1), transform_matrix, (width, height)
            )
            image_variations.append(warped_img)
            image_variations.append(warped_img_f)

            tx, ty = 3, 2
            translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
            translated = cv2.warpAffine(image, translation_matrix, (width, height))
            translated_f = cv2.warpAffine(
                cv2.flip(image, 1), translation_matrix, (width, height)
            )
            image_variations.append(translated)
            image_variations.append(translated_f)

            gamma_up = apply_gamma_correction(image, gamma=1.15)
            gamma_down = apply_gamma_correction(image, gamma=0.85)
            gamma_up_f = apply_gamma_correction(cv2.flip(image, 1), gamma=1.15)
            gamma_down_f = apply_gamma_correction(cv2.flip(image, 1), gamma=0.85)
            image_variations.extend([gamma_up, gamma_down, gamma_down_f, gamma_up_f])

            embeddings_list = []
            binary_list = []

            for img_var in image_variations:
                tensor_var = transform(img_var).unsqueeze(0).to(device)

                with torch.no_grad():
                    embedding_var = refinement_model(original_model(tensor_var))
                    embeddings_list.append(embedding_var)

                    binary_var = (embedding_var > 0).float()
                    binary_list.append(binary_var)

            binary_stack = torch.cat(binary_list, dim=0)

            vote_sum = binary_stack.sum(dim=0, keepdim=True)

            majority_threshold = len(image_variations) / 2
            final_binary = (vote_sum > majority_threshold).float()

            final_embedding = torch.zeros_like(embeddings_list[0])
            embeddings_stack = torch.cat(embeddings_list, dim=0)

            for dim in range(final_binary.shape[1]):
                if final_binary[0, dim] == 1:
                    mask = binary_stack[:, dim] == 1
                    if mask.sum() > 0:
                        final_embedding[0, dim] = embeddings_stack[mask, dim].mean()
                    else:
                        final_embedding[0, dim] = embeddings_list[0][0, dim]
                else:
                    mask = binary_stack[:, dim] == 0
                    if mask.sum() > 0:
                        final_embedding[0, dim] = embeddings_stack[mask, dim].mean()
                    else:
                        final_embedding[0, dim] = embeddings_list[0][0, dim]

            final_embedding = F.normalize(final_embedding, p=2, dim=1)

            if class_id not in embeddings_dict:
                embeddings_dict[class_id] = {}
            embeddings_dict[class_id][file_name] = final_embedding.squeeze()

            all_embeddings.append(final_embedding)
            all_labels.append(class_to_idx[class_id])

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.tensor(all_labels, device=device)

    print("Computing similarity statistics...")
    same_class_sims, diff_class_sims = compute_similarity_stats_batched(
        all_embeddings, all_labels, batch_size=1000
    )

    print("Computing Hamming distance statistics...")
    same_class_hamming, diff_class_hamming = compute_hamming_stats_batched(
        all_embeddings, all_labels, batch_size=100
    )

    print("Generating and saving histogram...")
    plot_similarity_histogram(
        same_class_sims,
        diff_class_sims,
        same_class_hamming,
        diff_class_hamming,
        "final_histograms/final_distribution_hist_goners_gone.png",
    )

    print("Saving embeddings organized by class...")
    save_embeddings_by_class(embeddings_dict, config["output_dir"])

    print("Processing complete!")
    print(f"Embeddings saved in: {config['output_dir']}")
    print(f"Histograms saved in: final_histograms/")


if __name__ == "__main__":
    main()
