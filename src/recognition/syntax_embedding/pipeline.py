from datetime import datetime
import pandas as pd
import numpy as np
import shutil
import torch
import yaml
import os

from .model import SyntaxEncoder
from .dataset import SyntaxEmbeddingTripletDataset, ImgToWordDataset
from .train import train, save_train_results, make_embedding_files
from .metrics import measure, find_closest_words, find_gt_closest_words


def read_train_datasets(dataset_path: str, img_size: int, img_format: str) -> tuple:
    train_df = pd.read_csv(os.path.join(dataset_path, "train.csv"))
    val_df = pd.read_csv(os.path.join(dataset_path, "val.csv"))
    test_df = pd.read_csv(os.path.join(dataset_path, "test.csv"))

    train_dataset = SyntaxEmbeddingTripletDataset(
        dataset_path,
        train_df,
        (img_size, img_size),
        img_format
    )
    val_dataset = SyntaxEmbeddingTripletDataset(
        dataset_path,
        val_df,
        (img_size, img_size),
        img_format
    )
    test_dataset = SyntaxEmbeddingTripletDataset(
        dataset_path,
        test_df,
        (img_size, img_size),
        img_format
    )

    return train_dataset, val_dataset, test_dataset


def read_inference_dataset(dataset_path: str, img_size: int, img_format: str) -> tuple:
    train_df = pd.read_csv(os.path.join(dataset_path, "train.csv"))
    val_df = pd.read_csv(os.path.join(dataset_path, "val.csv"))
    test_df = pd.read_csv(os.path.join(dataset_path, "test.csv"))
    train_img2word_dataset = ImgToWordDataset(
        dataset_path,
        img_format,
        img_size,
        train_df["anchor"].tolist()
    )
    val_img2word_dataset = ImgToWordDataset(
        dataset_path,
        img_format,
        img_size,
        val_df["anchor"].tolist()
    )
    test_img2word_dataset = ImgToWordDataset(
        dataset_path,
        img_format,
        img_size,
        test_df["anchor"].tolist()
    )

    return train_img2word_dataset, val_img2word_dataset, test_img2word_dataset


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
    verbose: bool = True
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
            closest_words = find_closest_words(
                vec, embeddings, margin=2, max_words=5
            )
            gt_closest_words = find_gt_closest_words(
                test_word, list(embeddings.keys()), threshold=2, sort=True
            )
            with open(file_name, "a") as f:
                pred_message = f"Predicted words, similar to \"{test_word}\": {closest_words}"
                gt_message = f"Ground truth similar words are: {gt_closest_words}"
                print(pred_message, file=f)
                print(gt_message, file=f)
                print("-" * 60, file=f)

                if verbose:
                    print(pred_message)
                    print(gt_message)
                    print("-" * 60)


def training_pipeline(config: str | dict):
    if isinstance(config, str):
        config = yaml.load(open(config), yaml.SafeLoader)

    save_dir = os.path.join(config["SAVE_PATH"], datetime.now().strftime("%Y-%m-%d %H_%M_%S"))
    os.makedirs(save_dir, exist_ok=True)
    yaml.dump(config, open(os.path.join(save_dir, "config.yaml"), "w"), yaml.SafeDumper)

    if config["PRETRAINED"] is not None:
        model = SyntaxEncoder(output_dim=config["EMBED_DIM"])
        model.load_state_dict(torch.load(
            config["PRETRAINED"], weights_only=True))
    else:
        model = None

    train_dataset, val_dataset, _ = read_train_datasets(
        config["DATASET_PATH"],
        config["IMG_SIZE"],
        config["IMG_FORMAT"]
    )

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
    save_train_results(result, save_dir)

    train_img2word_dataset, val_img2word_dataset, test_img2word_dataset = read_inference_dataset(
        config["DATASET_PATH"],
        config["IMG_SIZE"],
        config["IMG_FORMAT"]
    )

    embeddings = make_embedding_files(
        result.trained_model, save_dir, train_img2word_dataset)

    print("Validating...")
    nearest_precision = measure(
        result.trained_model, embeddings, val_img2word_dataset
    )
    print(f"The metric value is {nearest_precision}")
    write_metrics(os.path.join(save_dir, "metrics.txt"),
                  nearest_precision=nearest_precision)

    print("Testing...")
    test(
        result.trained_model,
        test_img2word_dataset,
        embeddings,
        os.path.join(save_dir, "test_results.txt"),
        -1
    )
