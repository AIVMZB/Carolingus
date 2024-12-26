import os
import yaml
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
from datetime import datetime
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np

from .dataset import SyntaxEmbeddingDataset
from .model import SyntaxEncoder
from .syntax_loss import SyntaxLoss


@dataclass
class TrainResult:
    metrics: dict[str, list]
    trained_model: SyntaxEncoder


def create_sampler(dataset, subset_size):
    indices = np.random.choice(len(dataset), size=subset_size, replace=False)
    return SubsetRandomSampler(indices)


def create_dataloader(dataset, subset_size, batch_size):
    sampler = create_sampler(dataset, subset_size)
    return DataLoader(dataset, sampler=sampler, batch_size=batch_size)


def get_image_paths(dataset_root: str) -> list:
    image_paths = []
    for document in os.listdir(dataset_root):
        for image_path in os.listdir(os.path.join(dataset_root, document)):
            image_paths.append(
                os.path.join(dataset_root, document, image_path)
            )

    return image_paths


def train(
    train_dataset: SyntaxEmbeddingDataset,
    val_dataset: SyntaxEmbeddingDataset,
    steps_per_epoch: int = 200,
    batch_size: int = 8,
    model: SyntaxEncoder | None = None,
    lr: float = 0.0008,
    margin: float = 0.5,
    epochs: int = 50
) -> TrainResult:
    if model is None:
        model = SyntaxEncoder(64)
    model = model.cuda()

    loss = SyntaxLoss(margin=margin)
    optimizer = Adam(model.parameters(), lr=lr)

    loss_history = []
    val_loss_history = []

    for epoch in range(1, epochs + 1):
        epoch_loss = 0
        train_dataloader = create_dataloader(train_dataset, steps_per_epoch, batch_size)
        val_dataloader = create_dataloader(val_dataset, steps_per_epoch, batch_size)

        model.train()
        for data in tqdm(train_dataloader):
            first_image = data["first_image"]
            second_image = data["second_image"]
            str_dist = data["str_distance"]

            optimizer.zero_grad()
            first_out = model(first_image)
            second_out = model(second_image)

            loss_value = loss(first_out, second_out, str_dist)
            loss_value.backward()
            epoch_loss += loss_value.item()

            optimizer.step()

        epoch_loss /= len(train_dataloader)
        loss_history.append(epoch_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_dataloader:
                first_image = data["first_image"]
                second_image = data["second_image"]
                str_dist = data["str_distance"]
                first_out = model(first_image)
                second_out = model(second_image)

                loss_value = loss(first_out, second_out, str_dist).item()
                val_loss += loss_value

        val_loss /= len(val_dataloader)
        val_loss_history.append(val_loss)

        print(f"[EPOCH {epoch}] Loss - {epoch_loss} | Val Loss - {val_loss}")

    return TrainResult(
        metrics={
            "train_loss": loss_history,
            "val_loss": val_loss_history
        },
        trained_model=model
    )


def get_train_val_datasets(
    dataset_root: str,
    img_size: tuple[int, int] = (120, 120),
    val_size: float = 0.2,
    seed: int = 42
) -> tuple[SyntaxEmbeddingDataset, SyntaxEmbeddingDataset]:
    total_images = []
    for document in os.listdir(dataset_root):
        for image in os.listdir(os.path.join(dataset_root, document)):
            total_images.append(
                os.path.join(dataset_root, document, image)
            )

    train_images, val_images = train_test_split(
        total_images, test_size=val_size, random_state=seed
    )
    train_dataset = SyntaxEmbeddingDataset(
        dataset_root, img_size, "RGB", train_images
    )
    val_dataset = SyntaxEmbeddingDataset(
        dataset_root, img_size, "RGB", val_images
    )

    return train_dataset, val_dataset


def training_pipeline(config: str | dict):
    if isinstance(config, str):
        config = yaml.load(open(config), yaml.SafeLoader)

    train_dataset, val_dataset = get_train_val_datasets(
        config["DATASET_PATH"],
        (config["IMG_SIZE"], config["IMG_SIZE"]),
        seed=config["SEED"]
    )

    os.makedirs(config["SAVE_PATH"], exist_ok=True)
    save_dir = os.path.join(
        config["SAVE_PATH"], datetime.now().strftime("%Y-%m-%d %H_%M_%S"))
    os.makedirs(save_dir, exist_ok=True)

    result = train(
        train_dataset,
        val_dataset,
        config["STEPS_PER_EPOCH"],
        config["BATCH_SIZE"],
        None,
        config["LR"],
        config["MARGIN"],
        config["EPOCHS"]
    )

    torch.save(result.trained_model.state_dict(), os.path.join(save_dir, "model.pth"))

    plt.title("Loss plot")
    plt.plot(result.metrics["train_loss"][1:], label="Train loss")
    plt.plot(result.metrics["val_loss"][1:], label="Validation loss")
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(save_dir, "losses.png"))
