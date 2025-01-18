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
from .metrics import measure, find_gt_closest_words
from .common import get_create_embeddings_dir, IndexedEmbeddings


def read_triplet_dataset(
    dataset_path: str,
    img_size: int,
    img_format: str,
    subsets: list,
    balanced: bool = False,
) -> SyntaxEmbeddingTripletDataset | list:
    datasets = []
    for subset in subsets:
        if balanced:
            df = pd.read_csv(os.path.join(dataset_path, f"{subset}-balanced.csv"))
        else:
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
    balanced: bool = False,
) -> HardTripletsDataset:
    df = pd.read_csv(os.path.join(dataset_path, f"{subset}.csv"))

    return HardTripletsDataset(
        dataset_root=dataset_path,
        embeddings=embeddings,
        str_dist_threshold=str_dist_threshold,
        balanced=balanced,
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
    balanced: bool = False,
) -> ImgToWordDataset | list:
    datasets = []
    for subset in subsets:
        if balanced:
            df = pd.read_csv(os.path.join(dataset_path, f"{subset}-balanced.csv"))
        else:
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
    indexed_embeddings: IndexedEmbeddings,
    file_name: str,
    margin: float,
    max_words: int,
    str_dist_thresh: int,
    n_samples: int = 10,
    verbose: bool = True,
):
    """
    Performs testing of the model. Saves the results to the file.
    Args:
        model (SyntaxEncoder): Model to test.
        dataset (ImgToWordDataset): Dataset to test on.
        indexed_embeddings (IndexedEmbeddings): Indexed embeddings of the training dataset.
        file_name (str): File to save the results to.
        margin (float): Margin for the similarity.
        max_words (int): Maximum number of words to show.
        str_dist_thresh (int): Threshold for the string distance.
        n_samples (int): Number of samples to test on.
        verbose (bool): Whether to print the results.
    """
    if n_samples == -1:
        n_samples = len(dataset)

    model.eval()
    with torch.no_grad():
        for _ in range(n_samples):
            test_data = dataset[np.random.randint(0, len(dataset))]
            test_word = test_data["word"]
            test_image = test_data["image"]

            vec = model(test_image)[0]
            closest_words = indexed_embeddings.find_closest_words(
                vec, margin=margin, max_words=max_words
            )
            gt_closest_words = find_gt_closest_words(
                test_word,
                indexed_embeddings.words,
                threshold=str_dist_thresh,
                sort=True,
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
    """
    Actions to perform after the training is done.
    Args:
        config (dict): Configuration dictionary.
        save_dir (str): Directory to save the results to.
        train_results (TrainResult): Results of the training.
    """

    save_train_results(train_results, save_dir)

    train_img2word_dataset, val_img2word_dataset, test_img2word_dataset = (
        read_img2word_dataset(
            config["DATASET_PATH"],
            config["IMG_SIZE"],
            config["IMG_FORMAT"],
            add_batch_dim=True,
            subsets=["train", "val", "test"],
            balanced=False,
        )
    )

    train_embeddings = make_save_embeddings(
        train_results.trained_model, save_dir, train_img2word_dataset, subset="train"
    )
    val_embeddings = make_save_embeddings(
        train_results.trained_model, save_dir, val_img2word_dataset, subset="val"
    )

    print("Validating...")
    indexed_train_embeddings = IndexedEmbeddings(train_embeddings)
    nearest_precision = measure(
        val_embeddings,
        indexed_train_embeddings,
        config["MARGIN_THRESHOLD"],
        threshold=2,
        max_words=3,
    )

    print(f"The metric value is {nearest_precision}")
    write_metrics(
        os.path.join(save_dir, "metrics.txt"), nearest_precision=nearest_precision
    )

    print("Testing...")
    test(
        train_results.trained_model,
        test_img2word_dataset,
        indexed_train_embeddings,
        os.path.join(save_dir, "test_results.txt"),
        config["MARGIN_THRESHOLD"],
        max_words=3,
        str_dist_thresh=2,
        n_samples=20,
    )


def training_pipeline(
    config: str | dict, save_dir: str, model: SyntaxEncoder | None = None
) -> SyntaxEncoder:
    """
    Training pipeline for syntax embedding model. Uses whole dataset
    Args:
        config (str | dict): Path to the config file or the config itself.
        save_dir (str): Directory to save the results to.
        model (SyntaxEncoder | None): Model to train. Defaults to None.
    Returns:
        SyntaxEncoder: Trained model.
    """
    if isinstance(config, str):
        config = yaml.load(open(config), yaml.SafeLoader)

    yaml.dump(config, open(os.path.join(save_dir, "config.yaml"), "w"), yaml.SafeDumper)

    if model is None and config["PRETRAINED"] is not None:
        model = SyntaxEncoder(output_dim=config["EMBED_DIM"])
        model.load_state_dict(torch.load(config["PRETRAINED"], weights_only=True))

    train_dataset, val_dataset = read_triplet_dataset(
        config["DATASET_PATH"],
        config["IMG_SIZE"],
        config["IMG_FORMAT"],
        ["train", "val"],
        balanced=config["SIMPLE_TRAIN"]["BALANCED"],
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


def hard_training_pipeline(
    config: str | dict, model: SyntaxEncoder, save_dir: str, repeat: int = 1
) -> SyntaxEncoder:
    """
    Pipeline for training hard triplets.
    Args:
        config (str | dict): Path to the config file or the config itself.
        model (SyntaxEncoder): Model to train.
        save_dir (str): Directory to save the results to.
        repeat (int): Number of the current repeat.
    Returns:
        SyntaxEncoder: Trained model.
    """
    if isinstance(config, str):
        config = yaml.load(open(config), yaml.SafeLoader)

    if repeat == 1:
        embeddings_dir = get_create_embeddings_dir(save_dir)
    else:
        embeddings_dir = os.path.join(save_dir, f"hard_train-{repeat - 1}/embeddings")

    train_embeddings = json.load(
        open(os.path.join(embeddings_dir, "train-embeddings.json"))
    )
    val_embeddings = json.load(
        open(os.path.join(embeddings_dir, "val-embeddings.json"))
    )

    train_dataset = read_hard_triplet_dataset(
        config["DATASET_PATH"],
        train_embeddings,
        "train",
        config["HARD_TRAIN"]["STR_DIST_THRESHOLD"],
        config["IMG_SIZE"],
        config["IMG_FORMAT"],
        balanced=config["HARD_TRAIN"]["BALANCED"],
    )
    # val_dataset = read_triplet_dataset(
    #     config["DATASET_PATH"],
    #     config["IMG_SIZE"],
    #     config["IMG_FORMAT"],
    #     subsets=["val"],
    #     balanced=config["HARD_TRAIN"]["BALANCED"],
    # )
    val_dataset = read_hard_triplet_dataset(
        config["DATASET_PATH"],
        val_embeddings,
        "val",
        config["HARD_TRAIN"]["STR_DIST_THRESHOLD"],
        config["IMG_SIZE"],
        config["IMG_FORMAT"],
        balanced=config["HARD_TRAIN"]["BALANCED"],
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
    save_dir = os.path.join(save_dir, f"hard_train-{repeat}")
    os.makedirs(save_dir)

    post_training_routine(config, save_dir, result)

    return result.trained_model


def main_pipeline(config: str | dict):
    """
    Main pipeline for training syntax embedding model.
    Args:
        config (str | dict): Path to the config file or the config itself.
    """
    if isinstance(config, str):
        config = yaml.load(open(config), yaml.SafeLoader)

    save_dir = os.path.join(
        config["SAVE_PATH"], datetime.now().strftime("%Y-%m-%d %H_%M_%S")
    )

    os.makedirs(save_dir, exist_ok=True)
    model = None
    for i in range(config["REPEAT"]):
        save_dir_of_repeat = os.path.join(save_dir, f"repeat-{i + 1}")
        os.makedirs(save_dir_of_repeat)

        model = training_pipeline(config, save_dir_of_repeat, model)
        for j in range(config["HARD_TRAIN"]["REPEAT"]):
            model = hard_training_pipeline(config, model, save_dir_of_repeat, repeat=j + 1)
