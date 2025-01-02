from datetime import datetime
import pandas as pd
import numpy as np
import json
import torch
import yaml
import os

from .model import SyntaxEncoder
from .dataset import (
    SyntaxEmbeddingTripletDataset,
    ImgToWordDataset,
    HardTripletsDataset,
)
from .train import train, save_train_results, make_save_embeddings, TrainResult
from .metrics import measure, find_closest_words, find_gt_closest_words
from .common import get_create_embeddings_dir


def read_triplet_dataset(
    dataset_path: str, img_size: int, img_format: str, subsets: list
) -> SyntaxEmbeddingTripletDataset | list:
    datasets = []
    for subset in subsets:
        df = pd.read_csv(os.path.join(dataset_path, f"{subset}.csv"))
        datasets.append(
            SyntaxEmbeddingTripletDataset(
                dataset_path, df, (img_size, img_size), img_format
            )
        )

    if len(datasets) == 1:
        return datasets[0]

    return datasets


def read_hard_triplet_dataset(
    dataset_path: str,
    embeddings: dict,
    subset: str,
    str_dist_threshold: int,
    img_size: int,
    img_format: str,
) -> HardTripletsDataset:
    df = pd.read_csv(os.path.join(dataset_path, f"{subset}.csv"))

    return HardTripletsDataset(
        dataset_root=dataset_path,
        embeddings=embeddings,
        str_dist_threshold=str_dist_threshold,
        subset_images=df["anchor"].tolist(),
        image_format=img_format,
        img_size=img_size,
    )


def read_img2word_dataset(
    dataset_path: str,
    img_size: int,
    img_format: str,
    add_batch_dim: bool = False,
    augment: bool = False,
    subsets: list = ["train"],
) -> ImgToWordDataset | list:
    datasets = []
    for subset in subsets:
        df = pd.read_csv(os.path.join(dataset_path, f"{subset}.csv"))
        datasets.append(
            ImgToWordDataset(
                dataset_root=dataset_path,
                image_format=img_format,
                img_size=img_size,
                images=df["anchor"].tolist(),
                add_batch_dim=add_batch_dim,
                augment=augment,
            )
        )

    if len(datasets) == 1:
        return datasets[0]

    return datasets


def write_metrics(file_name: str, **metrics):
    with open(file_name, "w") as f:
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value}", file=f)


def test(
    model: SyntaxEncoder,
    dataset: ImgToWordDataset,
    embeddings: dict,
    file_name: str,
    n_samples: int = 10,
    verbose: bool = True,
):
    if n_samples == -1:
        n_samples = len(dataset)

    model.eval()
    with torch.no_grad():
        for _ in range(n_samples):
            test_data = dataset[np.random.randint(0, len(dataset))]
            test_word = test_data["word"]
            test_image = test_data["image"]

            vec = model(test_image)[0]
            closest_words = find_closest_words(vec, embeddings, margin=2, max_words=5)
            gt_closest_words = find_gt_closest_words(
                test_word, list(embeddings.keys()), threshold=2, sort=True
            )
            with open(file_name, "a") as f:
                pred_message = (
                    f'Predicted words, similar to "{test_word}": {closest_words}'
                )
                gt_message = f"Ground truth similar words are: {gt_closest_words}"
                print(pred_message, file=f)
                print(gt_message, file=f)
                print("-" * 60, file=f)

                if verbose:
                    print(pred_message)
                    print(gt_message)
                    print("-" * 60)


def post_training_routine(config: dict, save_dir: str, train_results: TrainResult):

    save_train_results(train_results, save_dir)

    train_img2word_dataset, val_img2word_dataset, test_img2word_dataset = (
        read_img2word_dataset(
            config["DATASET_PATH"],
            config["IMG_SIZE"],
            config["IMG_FORMAT"],
            add_batch_dim=True,
            subsets=["train", "val", "test"],
        )
    )

    train_embeddings = make_save_embeddings(
        train_results.trained_model, save_dir, train_img2word_dataset, subset="train"
    )
    val_embeddings = make_save_embeddings(
        train_results.trained_model, save_dir, val_img2word_dataset, subset="val"
    )

    print("Validating...")
    nearest_precision = measure(
        val_embeddings, train_embeddings, config["MARGIN"], threshold=2, max_words=3
    )

    print(f"The metric value is {nearest_precision}")
    write_metrics(
        os.path.join(save_dir, "metrics.txt"), nearest_precision=nearest_precision
    )

    print("Testing...")
    test(
        train_results.trained_model,
        test_img2word_dataset,
        train_embeddings,
        os.path.join(save_dir, "test_results.txt"),
        -1,
    )


def training_pipeline(config: str | dict, save_dir: str) -> SyntaxEncoder:
    if isinstance(config, str):
        config = yaml.load(open(config), yaml.SafeLoader)

    yaml.dump(config, open(os.path.join(save_dir, "config.yaml"), "w"), yaml.SafeDumper)

    if config["PRETRAINED"] is not None:
        model = SyntaxEncoder(output_dim=config["EMBED_DIM"])
        model.load_state_dict(torch.load(config["PRETRAINED"], weights_only=True))
    else:
        model = None

    train_dataset, val_dataset = read_triplet_dataset(
        config["DATASET_PATH"],
        config["IMG_SIZE"],
        config["IMG_FORMAT"],
        ["train", "val"],
    )

    result = train(
        train_dataset,
        val_dataset,
        config["SIMPLE_TRAIN"]["STEPS_PER_EPOCH"],
        config["SIMPLE_TRAIN"]["BATCH_SIZE"],
        model,
        config["EMBED_DIM"],
        config["SIMPLE_TRAIN"]["LR"],
        config["MARGIN"],
        config["SIMPLE_TRAIN"]["EPOCHS"],
    )
    post_training_routine(config, save_dir, result)

    return result.trained_model


def hard_training_pipeline(config: str | dict, model: SyntaxEncoder, save_dir: str):
    if isinstance(config, str):
        config = yaml.load(open(config), yaml.SafeLoader)

    embeddings_dir = get_create_embeddings_dir(save_dir)
    train_embeddings = json.load(
        open(os.path.join(embeddings_dir, "train-embeddings.json"))
    )

    train_dataset = read_hard_triplet_dataset(
        config["DATASET_PATH"],
        train_embeddings,
        "train",
        config["HARD_TRAIN"]["STR_DIST_THRESHOLD"],
        config["IMG_SIZE"],
        config["IMG_FORMAT"],
    )
    val_dataset = read_triplet_dataset(
        config["DATASET_PATH"],
        config["IMG_SIZE"],
        config["IMG_FORMAT"],
        subsets=["val"],
    )

    result = train(
        train_dataset,
        val_dataset,
        config["HARD_TRAIN"]["STEPS_PER_EPOCH"],
        config["HARD_TRAIN"]["BATCH_SIZE"],
        model,
        config["EMBED_DIM"],
        config["HARD_TRAIN"]["LR"],
        config["MARGIN"],
        config["HARD_TRAIN"]["EPOCHS"],
    )
    save_dir = os.path.join(save_dir, "hard_train")
    os.makedirs(save_dir)

    post_training_routine(config, save_dir, result)


def main_pipeline(config: str | dict):
    if isinstance(config, str):
        config = yaml.load(open(config), yaml.SafeLoader)

    save_dir = os.path.join(
        config["SAVE_PATH"], datetime.now().strftime("%Y-%m-%d %H_%M_%S")
    )
    os.makedirs(save_dir, exist_ok=True)
    model = training_pipeline(config, save_dir)
    hard_training_pipeline(config, model, save_dir)
