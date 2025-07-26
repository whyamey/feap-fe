import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt


class IrisNet(nn.Module):
    def __init__(self, num_classes, embedding_dim=1024, pretrained=False):
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
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        x = self.backbone(x)
        embedding = self.embedding(torch.flatten(x, 1))
        return F.normalize(embedding, p=2, dim=1)


class AttentionRefiner(nn.Module):
    def __init__(self, input_dim=1024, output_dim=512):
        super().__init__()
        self.input_dim, self.output_dim = input_dim, output_dim
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


def rotate_image(image, angle):
    h, w = image.shape[:2]
    rot_mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    return cv2.warpAffine(image, rot_mat, (w, h))


def apply_gamma_correction(image, gamma=1.2):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype(
        "uint8"
    )
    return cv2.LUT(image, table)


def compute_similarity_stats_batched(embeddings, labels, batch_size=500):
    device, n = embeddings.device, len(embeddings)
    same_class_sims, diff_class_sims = [], []
    for i in tqdm(range(0, n, batch_size), desc="Computing Cosine Similarities"):
        batch_end = min(i + batch_size, n)
        batch_embeds, batch_labels = embeddings[i:batch_end], labels[i:batch_end]
        sim_matrix = torch.matmul(batch_embeds, embeddings.t())
        labels_matrix = batch_labels.unsqueeze(1) == labels.unsqueeze(0)
        for j in range(batch_end - i):
            global_idx = i + j
            row_sims, row_labels_match = (
                sim_matrix[j, global_idx + 1 :],
                labels_matrix[j, global_idx + 1 :],
            )
            if row_sims.numel() > 0:
                same_class_sims.append(row_sims[row_labels_match].cpu())
                diff_class_sims.append(row_sims[~row_labels_match].cpu())
    return torch.cat(same_class_sims).numpy(), torch.cat(diff_class_sims).numpy()


def compute_hamming_stats_batched_sign(embeddings, labels, batch_size=500):
    device, n = embeddings.device, len(embeddings)
    same_class_ham, diff_class_ham = [], []

    binary_embeddings = torch.sign(embeddings)
    embedding_dim = embeddings.shape[1]

    for i in tqdm(range(0, n, batch_size), desc="Computing Hamming Distances"):
        batch_end = min(i + batch_size, n)
        batch_embeds_bin, batch_labels = (
            binary_embeddings[i:batch_end],
            labels[i:batch_end],
        )

        dot_prod_matrix = torch.matmul(batch_embeds_bin, binary_embeddings.t())
        ham_dist_matrix = 0.5 * (embedding_dim - dot_prod_matrix) / embedding_dim

        labels_matrix = batch_labels.unsqueeze(1) == labels.unsqueeze(0)
        for j in range(batch_end - i):
            global_idx = i + j
            row_dists, row_labels_match = (
                ham_dist_matrix[j, global_idx + 1 :],
                labels_matrix[j, global_idx + 1 :],
            )
            if row_dists.numel() > 0:
                same_class_ham.append(row_dists[row_labels_match].cpu())
                diff_class_ham.append(row_dists[~row_labels_match].cpu())

    return torch.cat(same_class_ham).numpy(), torch.cat(diff_class_ham).numpy()


def plot_distributions(
    same_sims,
    diff_sims,
    same_ham,
    diff_ham,
    save_path,
):

    plt.close("all")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    fig.suptitle("Final N-Vote Embedding Distributions", fontsize=20)

    same_sim_label = f"Same Class (μ={same_sims.mean():.4f}, σ={same_sims.std():.4f})"
    diff_sim_label = (
        f"Different Class (μ={diff_sims.mean():.4f}, σ={diff_sims.std():.4f})"
    )

    ax1.hist(
        same_sims, bins=100, alpha=0.7, density=True, label=same_sim_label, color="blue"
    )
    ax1.hist(
        diff_sims, bins=100, alpha=0.7, density=True, label=diff_sim_label, color="red"
    )
    ax1.set_title("Cosine Similarity Distribution", fontsize=16)
    ax1.set_xlabel("Cosine Similarity", fontsize=12)
    ax1.set_ylabel("Density", fontsize=12)
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    same_ham_label = f"Same Class (μ={same_ham.mean():.4f}, σ={same_ham.std():.4f})"
    diff_ham_label = (
        f"Different Class (μ={diff_ham.mean():.4f}, σ={diff_ham.std():.4f})"
    )

    ax2.hist(
        same_ham,
        bins=50,
        range=(0, 1),
        alpha=0.7,
        density=True,
        label=same_ham_label,
        color="blue",
    )
    ax2.hist(
        diff_ham,
        bins=50,
        range=(0, 1),
        alpha=0.7,
        density=True,
        label=diff_ham_label,
        color="red",
    )
    ax2.set_title("Fractional Hamming Distance Distribution", fontsize=16)
    ax2.set_xlabel("Fractional Hamming Distance", fontsize=12)
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(save_path, dpi=300)
    print(f"\nHistogram plot saved to: {save_path}")


def save_embeddings_by_class(embeddings_dict, base_output_dir):
    print(f"\nSaving embeddings to {base_output_dir}...")
    float_dir = os.path.join(base_output_dir, "iitd_embed_cosine")
    binary_dir = os.path.join(base_output_dir, "iitd_embed_hamming")
    for class_id, class_data in tqdm(
        embeddings_dict.items(), desc="Saving embedding files"
    ):
        os.makedirs(os.path.join(float_dir, class_id), exist_ok=True)
        os.makedirs(os.path.join(binary_dir, class_id), exist_ok=True)
        for file_name, embedding in class_data.items():
            float_embedding = embedding.cpu().numpy()
            binary_embedding = ((np.sign(float_embedding) + 1) / 2).astype(int)

            float_filepath = os.path.join(float_dir, class_id, file_name)
            float_string = ",".join([f"{x:.8f}" for x in float_embedding]) + ","
            with open(float_filepath, "w") as f:
                f.write(float_string)

            binary_filepath = os.path.join(binary_dir, class_id, file_name)
            binary_string = ",".join([str(x) for x in binary_embedding]) + ","
            with open(binary_filepath, "w") as f:
                f.write(binary_string)


def main():
    config = {
        "data_dir": "/share/user/feap/iitd_segmented/",
        "primary_checkpoint_path": "checkpoint_stage_one.pt",
        "refiner_checkpoint_path": "checkpoint_stage_two.pt",
        "primary_embedding_dim": 1024,
        "refined_embedding_dim": 512,
        "output_dir": "./iitd_embeddings_nvote_effnet",
        "histograms_dir": "./final_histograms_effnet",
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
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(config["histograms_dir"], exist_ok=True)

    primary_model = IrisNet(
        num_classes=224, embedding_dim=config["primary_embedding_dim"]
    ).to(device)
    refiner_model = AttentionRefiner(
        input_dim=config["primary_embedding_dim"],
        output_dim=config["refined_embedding_dim"],
    ).to(device)
    print(f"Loading primary model from {config['primary_checkpoint_path']}...")
    primary_checkpoint = torch.load(
        config["primary_checkpoint_path"], map_location=device
    )
    primary_model.load_state_dict(
        primary_checkpoint.get("model_state_dict", primary_checkpoint)
    )
    print(f"Loading refiner model from {config['refiner_checkpoint_path']}...")
    refiner_checkpoint = torch.load(
        config["refiner_checkpoint_path"], map_location=device
    )
    refiner_model.load_state_dict(
        refiner_checkpoint.get("model_state_dict", refiner_checkpoint)
    )
    primary_model.eval()
    refiner_model.eval()

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229]),
        ]
    )
    embeddings_dict, all_final_embeddings, all_labels, class_to_idx = {}, [], [], {}
    print("\nStarting N-Vote embedding generation...")
    all_class_folders = sorted(
        [
            d
            for d in os.listdir(config["data_dir"])
            if os.path.isdir(os.path.join(config["data_dir"], d))
        ]
    )
    exclude_list = set(config.get("exclude_classes", []))
    class_folders = [d for d in all_class_folders if d not in exclude_list]
    print(
        f"Found {len(all_class_folders)} total classes. Excluding {len(exclude_list)}. Processing {len(class_folders)}."
    )

    for class_id in tqdm(class_folders, desc="Processing Classes"):
        class_dir = os.path.join(config["data_dir"], class_id)
        if class_id not in class_to_idx:
            class_to_idx[class_id] = len(class_to_idx)
        for img_name in os.listdir(class_dir):
            if not (img_name.endswith(".bmp") or img_name.endswith(".png")):
                continue
            if "_R" not in img_name:
                continue
            file_name_base = os.path.splitext(img_name)[0]
            image = cv2.imread(os.path.join(class_dir, img_name))
            if image is None:
                continue

            image_variations, img_flipped, img_flipped_h = (
                [],
                cv2.flip(image, 0),
                cv2.flip(image, 1),
            )

            image_variations.extend(
                [
                    image.copy(),
                    img_flipped.copy(),
                    img_flipped_h.copy(),
                    rotate_image(image, 5),
                    rotate_image(image, -5),
                    rotate_image(img_flipped, 5),
                    rotate_image(img_flipped, -5),
                    rotate_image(img_flipped_h, -5),
                ]
            )

            embeddings_list, binary_list = [], []
            with torch.no_grad():
                for img_var in image_variations:
                    img_gray_var = cv2.cvtColor(img_var, cv2.COLOR_BGR2GRAY)
                    tensor_var = transform(img_gray_var).unsqueeze(0).to(device)
                    primary_embed = primary_model(tensor_var)
                    refined_embed = refiner_model(primary_embed)
                    embeddings_list.append(refined_embed)
                    binary_list.append(torch.sign(refined_embed))

            embeddings_stack = torch.cat(embeddings_list, dim=0)
            binary_stack = torch.cat(binary_list, dim=0)
            vote_sum = binary_stack.sum(dim=0, keepdim=True)
            final_binary = torch.sign(vote_sum)
            final_binary[final_binary == 0] = 1
            final_embedding = torch.zeros_like(embeddings_list[0])
            for dim in range(final_binary.shape[1]):
                mask = binary_stack[:, dim] == final_binary[0, dim]
                final_embedding[0, dim] = (
                    embeddings_stack[mask, dim].mean()
                    if mask.sum() > 0
                    else embeddings_stack[:, dim].mean()
                )
            final_embedding = F.normalize(final_embedding, p=2, dim=1)

            if class_id not in embeddings_dict:
                embeddings_dict[class_id] = {}
            embeddings_dict[class_id][file_name_base] = final_embedding.squeeze()
            all_final_embeddings.append(final_embedding)
            all_labels.append(class_to_idx[class_id])

    all_final_embeddings = torch.cat(all_final_embeddings, dim=0)
    all_labels = torch.tensor(all_labels, device=device)

    same_class_sims, diff_class_sims = compute_similarity_stats_batched(
        all_final_embeddings, all_labels
    )
    same_class_ham, diff_class_ham = compute_hamming_stats_batched_sign(
        all_final_embeddings, all_labels
    )

    plot_distributions(
        same_class_sims,
        diff_class_sims,
        same_class_ham,
        diff_class_ham,
        save_path=os.path.join(
            config["histograms_dir"], "final_nvote_distributions.png"
        ),
    )

    save_embeddings_by_class(embeddings_dict, config["output_dir"])
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
