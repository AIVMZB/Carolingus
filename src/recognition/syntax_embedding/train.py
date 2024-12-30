import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn import functional as F
from tqdm import tqdm
from torch.optim import Adam
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import json

from .dataset import SyntaxEmbeddingTripletDataset, ImgToWordDataset
from .model import SyntaxEncoder
from .metrics import string_distance


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
    # TODO: Make online hard training https://omoindrot.github.io/triplet-loss#offline-and-online-triplet-mining
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


def save_train_results(train_results: TrainResult, save_dir: str, tag: str = ""):
    torch.save(train_results.trained_model.state_dict(), os.path.join(save_dir, f"{tag}model.pth"))

    plt.title("Loss plot")
    plt.plot(train_results.metrics["train_loss"][1:], label="Train loss")
    plt.plot(train_results.metrics["val_loss"][1:], label="Validation loss")
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"{tag}losses.png"))


def make_embedding_files(model: SyntaxEncoder, save_dir: str, dataset: ImgToWordDataset):
    embeddings_per_word: dict[str, list] = {}
    embeddings_per_image: dict[str, list] = {}
    model.eval()
    print("Making embeddings...")
    with torch.no_grad():
        for i in tqdm(range(len(dataset) - 1)):
            data = dataset[i]
            word, image, image_path = data["word"], data["image"], data["image_path"]
            vec = model(image)[0]
            if word in embeddings_per_word:
                embeddings_per_word[word].append(vec.cpu().numpy())
            else:
                embeddings_per_word[word] = [vec.cpu().numpy()]

            embeddings_per_image[image_path] = vec.cpu().numpy().tolist()

    json.dump(
        embeddings_per_image, 
        open(os.path.join(save_dir, "embeddings_per_image.json"), "w"), 
        indent=2
    )

    print("Calculating averages...")
    for word, vecs in embeddings_per_word.items():
        if len(vecs) > 1:
            vecs = np.array(vecs).mean(axis=0).tolist()
        elif len(vecs) == 1:
            vecs = vecs[0].tolist()

        embeddings_per_word[word] = vecs

    json.dump(embeddings_per_word, open(os.path.join(save_dir, "embeddings.json"), "w"), indent=2)

    return embeddings_per_word

