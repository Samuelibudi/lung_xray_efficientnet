"""
Fine-tune Google EfficientNet-B7 on the Kaggle Chest X-Ray Pneumonia dataset.

The script is designed to run primarily on Kaggle's GPU environment while
being developed and maintained locally in Visual Studio Code.

Dataset:
    paultimothymooney/chest-xray-pneumonia

Pretrained model:
    google/efficientnet-b7

The dataset is downloaded through kagglehub inside the Kaggle environment.
It is not downloaded to the local development machine.

The fine-tuned Hugging Face model is saved in Hugging Face-compatible format
and can subsequently be downloaded to the local computer.

Author:
    Your Name

Project:
    Lung X-Ray Pneumonia Classification
"""

from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import (
    AutoImageProcessor,
    EfficientNetForImageClassification,
)

import kagglehub


# ============================================================================
# Configuration
# ============================================================================

MODEL_NAME = "google/efficientnet-b7"

DATASET_HANDLE = "paultimothymooney/chest-xray-pneumonia"

OUTPUT_DIR = Path("/kaggle/working/efficientnet-b7-pneumonia")

IMAGE_SIZE = 600

BATCH_SIZE = 8

NUM_EPOCHS = 5

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 2

RANDOM_SEED = 42

CLASS_NAMES = [
    "NORMAL",
    "PNEUMONIA",
]


# ============================================================================
# Reproducibility
# ============================================================================


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# Device
# ============================================================================


def get_device() -> torch.device:
    """Select CUDA GPU when available, otherwise use CPU.

    Returns:
        PyTorch device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA GPU not available. Using CPU.")

    return device


# ============================================================================
# Dataset
# ============================================================================


class ChestXrayDataset(Dataset):
    """Dataset for the Kaggle Chest X-Ray Pneumonia dataset."""

    def __init__(
        self,
        image_paths: list[Path],
        labels: list[int],
        transform=None,
    ) -> None:
        """Initialize the dataset.

        Args:
            image_paths: List of image file paths.
            labels: Integer labels corresponding to images.
            transform: Image preprocessing and augmentation pipeline.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.image_paths)

    def __getitem__(self, index: int):
        """Return an image tensor and its label.

        Args:
            index: Dataset index.

        Returns:
            Tuple containing image tensor and integer label.
        """
        image_path = self.image_paths[index]
        label = self.labels[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def find_dataset_root(dataset_path: Path) -> Path:
    """Locate the chest_xray directory inside the downloaded dataset.

    Args:
        dataset_path: Root path returned by kagglehub.

    Returns:
        Path to the chest_xray directory.

    Raises:
        FileNotFoundError: If the expected directory cannot be found.
    """
    possible_roots = [
        dataset_path / "chest_xray",
        dataset_path,
    ]

    for root in possible_roots:
        if (root / "train").exists():
            return root

    raise FileNotFoundError(
        f"Could not find the expected dataset structure in {dataset_path}."
    )


def collect_images(
    dataset_root: Path,
    split: str,
) -> tuple[list[Path], list[int]]:
    """Collect image paths and labels for a dataset split.

    Args:
        dataset_root: Root directory of the chest X-ray dataset.
        split: Dataset split, such as train, val, or test.

    Returns:
        Tuple containing image paths and integer labels.
    """
    split_directory = dataset_root / split

    image_paths = []
    labels = []

    for label, class_name in enumerate(CLASS_NAMES):
        class_directory = split_directory / class_name

        if not class_directory.exists():
            raise FileNotFoundError(
                f"Class directory not found: {class_directory}"
            )

        for image_path in class_directory.iterdir():
            if image_path.suffix.lower() in {
                ".jpg",
                ".jpeg",
                ".png",
            }:
                image_paths.append(image_path)
                labels.append(label)

    return image_paths, labels


# ============================================================================
# Transforms
# ============================================================================


def create_transforms(
    image_processor,
) -> tuple[transforms.Compose, transforms.Compose]:
    """Create training and validation preprocessing pipelines.

    Args:
        image_processor: Hugging Face image processor.

    Returns:
        Training and validation transforms.
    """
    image_mean = image_processor.image_mean
    image_std = image_processor.image_std

    train_transform = transforms.Compose(
        [
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            ),
            transforms.RandomHorizontalFlip(
                p=0.5
            ),
            transforms.RandomRotation(
                degrees=7
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=image_mean,
                std=image_std,
            ),
        ]
    )

    validation_transform = transforms.Compose(
        [
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=image_mean,
                std=image_std,
            ),
        ]
    )

    return train_transform, validation_transform


# ============================================================================
# Data loaders
# ============================================================================


def create_data_loaders(
    dataset_root: Path,
    image_processor,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create training, validation, and test data loaders.

    Args:
        dataset_root: Dataset root directory.
        image_processor: Hugging Face image processor.

    Returns:
        Training, validation, and test DataLoaders.
    """
    train_transform, validation_transform = create_transforms(
        image_processor
    )

    train_paths, train_labels = collect_images(
        dataset_root,
        "train",
    )

    validation_paths, validation_labels = collect_images(
        dataset_root,
        "val",
    )

    test_paths, test_labels = collect_images(
        dataset_root,
        "test",
    )

    train_dataset = ChestXrayDataset(
        train_paths,
        train_labels,
        train_transform,
    )

    validation_dataset = ChestXrayDataset(
        validation_paths,
        validation_labels,
        validation_transform,
    )

    test_dataset = ChestXrayDataset(
        test_paths,
        test_labels,
        validation_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(validation_dataset)}")
    print(f"Test images: {len(test_dataset)}")

    return (
        train_loader,
        validation_loader,
        test_loader,
    )


# ============================================================================
# Model
# ============================================================================


def create_model() -> EfficientNetForImageClassification:
    """Load pretrained EfficientNet-B7 and configure its classifier.

    Returns:
        Configured EfficientNet-B7 model.
    """
    id2label = {
        0: "NORMAL",
        1: "PNEUMONIA",
    }

    label2id = {
        "NORMAL": 0,
        "PNEUMONIA": 1,
    }

    model = EfficientNetForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    return model


# ============================================================================
# Training
# ============================================================================


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    device,
) -> float:
    """Train the model for one epoch.

    Args:
        model: EfficientNet classification model.
        data_loader: Training DataLoader.
        optimizer: PyTorch optimizer.
        device: Training device.

    Returns:
        Average training loss.
    """
    model.train()

    total_loss = 0.0

    progress_bar = tqdm(
        data_loader,
        desc="Training",
    )

    for images, labels in progress_bar:
        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad()

        outputs = model(
            pixel_values=images,
            labels=labels,
        )

        loss = outputs.loss

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    return total_loss / len(data_loader)


# ============================================================================
# Validation
# ============================================================================


def evaluate_model(
    model,
    data_loader,
    device,
) -> dict:
    """Evaluate the model.

    Args:
        model: EfficientNet classification model.
        data_loader: Evaluation DataLoader.
        device: Evaluation device.

    Returns:
        Dictionary containing evaluation metrics.
    """
    model.eval()

    predictions = []
    true_labels = []

    total_loss = 0.0

    with torch.no_grad():
        for images, labels in tqdm(
            data_loader,
            desc="Evaluating",
        ):
            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            outputs = model(
                pixel_values=images,
                labels=labels,
            )

            total_loss += outputs.loss.item()

            predicted_labels = torch.argmax(
                outputs.logits,
                dim=1,
            )

            predictions.extend(
                predicted_labels.cpu().numpy()
            )

            true_labels.extend(
                labels.cpu().numpy()
            )

    accuracy = accuracy_score(
        true_labels,
        predictions,
    )

    precision = precision_score(
        true_labels,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        true_labels,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        true_labels,
        predictions,
        zero_division=0,
    )

    return {
        "loss": total_loss / len(data_loader),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predictions": predictions,
        "labels": true_labels,
    }


# ============================================================================
# Model saving
# ============================================================================


def save_model(
    model,
    image_processor,
    output_directory: Path,
) -> None:
    """Save the fine-tuned model and image processor.

    Args:
        model: Fine-tuned EfficientNet model.
        image_processor: Hugging Face image processor.
        output_directory: Destination directory.
    """
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_pretrained(
        output_directory
    )

    image_processor.save_pretrained(
        output_directory
    )

    print(
        f"Model saved to: {output_directory}"
    )


# ============================================================================
# Reporting
# ============================================================================


def print_evaluation_report(
    results: dict,
) -> None:
    """Print classification metrics.

    Args:
        results: Evaluation results dictionary.
    """
    print("\nEvaluation Results")
    print("==================")

    print(
        f"Loss:       {results['loss']:.4f}"
    )

    print(
        f"Accuracy:   {results['accuracy']:.4f}"
    )

    print(
        f"Precision:  {results['precision']:.4f}"
    )

    print(
        f"Recall:     {results['recall']:.4f}"
    )

    print(
        f"F1-score:   {results['f1']:.4f}"
    )

    print("\nConfusion Matrix")
    print("================")

    matrix = confusion_matrix(
        results["labels"],
        results["predictions"],
    )

    print(matrix)

    print("\nClassification Report")
    print("=====================")

    report = classification_report(
        results["labels"],
        results["predictions"],
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    print(report)


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    """Execute the complete model fine-tuning pipeline."""
    print("=" * 70)
    print("EfficientNet-B7 Lung X-Ray Fine-Tuning")
    print("=" * 70)

    set_seed(RANDOM_SEED)

    device = get_device()

    print("\nDownloading/accessing Kaggle dataset...")

    dataset_path = Path(
        kagglehub.dataset_download(
            DATASET_HANDLE
        )
    )

    print(
        f"Dataset location: {dataset_path}"
    )

    dataset_root = find_dataset_root(
        dataset_path
    )

    print(
        f"Dataset root: {dataset_root}"
    )

    print("\nLoading Hugging Face image processor...")

    image_processor = AutoImageProcessor.from_pretrained(
        MODEL_NAME
    )

    print("\nCreating data loaders...")

    (
        train_loader,
        validation_loader,
        test_loader,
    ) = create_data_loaders(
        dataset_root,
        image_processor,
    )

    print("\nLoading pretrained EfficientNet-B7...")

    model = create_model()

    model.to(device)

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"Total parameters: "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_f1 = 0.0

    print("\nStarting fine-tuning...")

    for epoch in range(NUM_EPOCHS):
        print(
            f"\nEpoch "
            f"{epoch + 1}/{NUM_EPOCHS}"
        )

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
        )

        validation_results = evaluate_model(
            model,
            validation_loader,
            device,
        )

        print(
            f"\nTraining loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Validation loss: "
            f"{validation_results['loss']:.4f}"
        )

        print(
            f"Validation accuracy: "
            f"{validation_results['accuracy']:.4f}"
        )

        print(
            f"Validation F1: "
            f"{validation_results['f1']:.4f}"
        )

        if validation_results["f1"] > best_f1:
            best_f1 = validation_results["f1"]

            print(
                "\nNew best model found. "
                "Saving checkpoint..."
            )

            save_model(
                model,
                image_processor,
                OUTPUT_DIR,
            )

    print("\nLoading best saved model...")

    model = EfficientNetForImageClassification.from_pretrained(
        OUTPUT_DIR
    )

    model.to(device)

    print("\nEvaluating on test set...")

    test_results = evaluate_model(
        model,
        test_loader,
        device,
    )

    print_evaluation_report(
        test_results
    )

    print("\nTraining complete.")
    print(
        f"Final model directory: "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()