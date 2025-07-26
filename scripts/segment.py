import os
import cv2
import numpy as np
from pathlib import Path
import re


def extract_mask_info(mask_filename):
    pattern = r"OperatorA_(\d{3})-[AB]_(\d{2})\.tiff"
    match = re.match(pattern, mask_filename)
    if match:
        class_num = match.group(1)
        iris_num = match.group(2)
        return class_num, iris_num
    return None, None


def find_matching_iris_image(iris_base_dir, class_num, iris_num):
    class_dir = os.path.join(iris_base_dir, class_num)
    if not os.path.exists(class_dir):
        return None, None

    for filename in os.listdir(class_dir):
        if filename.endswith(".bmp"):
            file_pattern = r"(\d{2})_[LR]\.bmp"
            match = re.match(file_pattern, filename)
            if match and match.group(1) == iris_num:
                return os.path.join(class_dir, filename), filename

    return None, None


def apply_mask_to_image(image_path, mask_path):
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Error: Could not read image {image_path}")
        return None

    if mask is None:
        print(f"Error: Could not read mask {mask_path}")
        return None

    if image.shape[:2] != mask.shape:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    masked_image = cv2.bitwise_and(image, mask_3channel)

    return masked_image


def process_iris_segmentation(mask_dir, iris_dir, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(".tiff")]

    processed_count = 0
    skipped_count = 0

    for mask_file in mask_files:
        print(f"Processing mask: {mask_file}")
        class_num, iris_num = extract_mask_info(mask_file)

        if class_num is None or iris_num is None:
            print(f"Could not parse mask filename: {mask_file}")
            skipped_count += 1
            continue

        iris_image_path, iris_filename = find_matching_iris_image(
            iris_dir, class_num, iris_num
        )

        if iris_image_path is None:
            print(
                f"Could not find matching iris image for class {class_num}, iris {iris_num}"
            )
            skipped_count += 1
            continue

        print(f"  Found matching iris: {iris_image_path}")

        mask_path = os.path.join(mask_dir, mask_file)
        segmented_image = apply_mask_to_image(iris_image_path, mask_path)

        if segmented_image is None:
            print(f"  Failed to apply mask")
            skipped_count += 1
            continue

        output_class_dir = os.path.join(output_dir, class_num)
        Path(output_class_dir).mkdir(parents=True, exist_ok=True)

        output_path = os.path.join(output_class_dir, iris_filename)
        cv2.imwrite(output_path, segmented_image)

        print(f"  Saved segmented image: {output_path}")
        processed_count += 1

    print(f"\nProcessing complete!")
    print(f"Successfully processed: {processed_count} images")
    print(f"Skipped: {skipped_count} images")


if __name__ == "__main__":
    MASK_DIR = "path/to/your/mask/directory"
    IRIS_DIR = "path/to/your/iris/directory"
    OUTPUT_DIR = "iitd_segmented"

    if not os.path.exists(MASK_DIR):
        print(f"Error: Mask directory does not exist: {MASK_DIR}")
        exit(1)

    if not os.path.exists(IRIS_DIR):
        print(f"Error: Iris directory does not exist: {IRIS_DIR}")
        exit(1)

    process_iris_segmentation(MASK_DIR, IRIS_DIR, OUTPUT_DIR)
