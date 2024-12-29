import os
import yaml
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn import functional as F
from tqdm import tqdm
from datetime import datetime
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import json

from .dataset import SyntaxEmbeddingTripletDataset, ImgToWordDataset
from .model import SyntaxEncoder
from .syntax_loss import SyntaxLoss
from .metrics import find_closest_words, find_gt_closest_words, measure


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
    train_dataset: SyntaxEmbeddingTripletDataset,
    val_dataset: SyntaxEmbeddingTripletDataset,
    steps_per_epoch: int = 200,
    batch_size: int = 8,
    model: SyntaxEncoder | None = None,
    embed_dim: int = 64,
    lr: float = 0.0008,
    margin: float = 0.5,
    epochs: int = 50
) -> TrainResult:
    if model is None:
        model = SyntaxEncoder(embed_dim)
    model = model.cuda()

    loss_function = torch.nn.TripletMarginLoss(margin=margin)
    optimizer = Adam(model.parameters(), lr=lr)

    loss_history = []
    val_loss_history = []

    for epoch in range(1, epochs + 1):
        epoch_loss = 0
        train_dataloader = create_dataloader(train_dataset, steps_per_epoch, batch_size)
        val_dataloader = create_dataloader(val_dataset, steps_per_epoch // 2, batch_size)

        model.train()
        for anchor, positive, negative in tqdm(train_dataloader):
            optimizer.zero_grad()
            anchor_emb = model(anchor)
            positive_emb = model(positive)
            negative_emb = model(negative)

            loss_value = loss_function(anchor_emb, positive_emb, negative_emb)
            loss_value.backward()
            epoch_loss += loss_value.item()

            optimizer.step()

        epoch_loss /= len(train_dataloader)
        loss_history.append(epoch_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for anchor, positive, negative in val_dataloader:
                anchor_emb = model(anchor)
                positive_emb = model(positive)
                negative_emb = model(negative)

                loss_value = loss_function(anchor_emb, positive_emb, negative_emb).item()
                val_loss += loss_value

        val_loss /= len(val_dataloader)
        val_loss_history.append(val_loss)

        print(f"[EPOCH {epoch} / {epochs}] Loss - {epoch_loss} | Val Loss - {val_loss}")

    return TrainResult(
        metrics={
            "train_loss": loss_history,
            "val_loss": val_loss_history
        },
        trained_model=model
    )


def get_train_val_images(
    dataset_root: str, 
    val_size: float = 0.2,
    seed: int = 42
) -> tuple[list, list]:
    total_images = []
    for document in os.listdir(dataset_root):
        if "." in document:
            continue
        
        for image in os.listdir(os.path.join(dataset_root, document)):
            total_images.append(
                os.path.join(dataset_root, document, image)
            )

    train_images, val_images = train_test_split(
        total_images, test_size=val_size, random_state=seed
    )
    return train_images, val_images


def save_train_results(train_results: TrainResult, save_dir: str, config: dict):
    os.makedirs(save_dir, exist_ok=True)

    torch.save(train_results.trained_model.state_dict(), os.path.join(save_dir, "model.pth"))

    plt.title("Loss plot")
    plt.plot(train_results.metrics["train_loss"][1:], label="Train loss")
    plt.plot(train_results.metrics["val_loss"][1:], label="Validation loss")
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(save_dir, "losses.png"))

    with open(os.path.join(save_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f, yaml.SafeDumper)


def make_embedding_file(model: SyntaxEncoder, save_dir: str, dataset: ImgToWordDataset):
    embeddings: dict[str, list] = {}
    model.eval()
    print("Making embeddings...")
    with torch.no_grad():
        for i in tqdm(range(len(dataset) - 1)):
            data = dataset[i]
            word, image = data["word"], data["image"]
            vec = model(image)[0]
            if word in embeddings:
                embeddings[word].append(vec.cpu().numpy())
            else:
                embeddings[word] = [vec.cpu().numpy()]

    print("Calculating averages...")
    for word, vecs in embeddings.items():
        if len(vecs) > 1:
            vecs = np.array(vecs).mean(axis=0).tolist()
        elif len(vecs) == 1:
            vecs = vecs[0].tolist()

        embeddings[word] = vecs

    json.dump(embeddings, open(os.path.join(save_dir, "embeddings.json"), "w"), indent=2)

    return embeddings


def training_pipeline(config: str | dict):
    if isinstance(config, str):
        config = yaml.load(open(config), yaml.SafeLoader)

    if config["PRETRAINED"] is not None:
        model = SyntaxEncoder(output_dim=config["EMBED_DIM"])
        model.load_state_dict(torch.load(config["PRETRAINED"], weights_only=True))
    else: 
        model = None

    train_df = pd.read_csv(os.path.join(config["DATASET_PATH"], "train.csv"))
    val_df = pd.read_csv(os.path.join(config["DATASET_PATH"], "val.csv"))
    test_df = pd.read_csv(os.path.join(config["DATASET_PATH"], "test.csv"))
    
    train_dataset = SyntaxEmbeddingTripletDataset(
        config["DATASET_PATH"],
        train_df,
        (config["IMG_SIZE"], config["IMG_SIZE"]),
        config["IMG_FORMAT"]
    )
    val_dataset = SyntaxEmbeddingTripletDataset(
        config["DATASET_PATH"],
        val_df,
        (config["IMG_SIZE"], config["IMG_SIZE"]),
        config["IMG_FORMAT"]
    )

    save_dir = os.path.join(config["SAVE_PATH"], datetime.now().strftime("%Y-%m-%d %H_%M_%S"))
    result = train(
        train_dataset,
        val_dataset,
        config["STEPS_PER_EPOCH"],
        config["BATCH_SIZE"],
        model,
        config["EMBED_DIM"],
        config["LR"],
        config["MARGIN"],
        config["EPOCHS"]
    )
    save_train_results(result, save_dir, config)

    train_img2word_dataset = ImgToWordDataset(
        config["DATASET_PATH"],
        config["IMG_FORMAT"],
        train_df["anchor"].tolist()
    )
    val_img2word_dataset = ImgToWordDataset(
        config["DATASET_PATH"],
        config["IMG_FORMAT"],
        val_df["anchor"].tolist()
    )
    test_img2word_dataset = ImgToWordDataset(
        config["DATASET_PATH"],
        config["IMG_FORMAT"],
        test_df["anchor"].tolist()
    )
    embeddings = make_embedding_file(result.trained_model, save_dir, train_img2word_dataset)

    print("Validating...")
    nearest_precision = measure(result.trained_model, embeddings, val_img2word_dataset)
    print(f"The metric value is {nearest_precision}")

    result.trained_model.eval()
    print("Testing...")
    with torch.no_grad():
        for _ in range(10):
            val_data = test_img2word_dataset[np.random.randint(0, len(test_img2word_dataset))]
            val_word = val_data["word"]
            val_image = val_data["image"]

            vec = result.trained_model(val_image)[0]
            closest_words = find_closest_words(vec, embeddings, margin=2, max_words=5)
            gt_closest_words = find_gt_closest_words(val_word, list(embeddings.keys()), threshold=2, sort=True)
            print(f"The closes words to \"{val_word}\" are \n{closest_words}")
            print(f"Ground truth closest words are\n{gt_closest_words}")
            print("-------------------------------------")

# TODO: find hard pairs and train on them
