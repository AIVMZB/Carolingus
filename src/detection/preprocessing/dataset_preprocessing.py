import os
import shutil
import cv2

from .preprocessor import ImagePreprocessor


def preprocess_folder(
    base_dir: str, new_dir: str, preprocessor: ImagePreprocessor
) -> None:
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)

    for filename in os.listdir(base_dir):
        if filename.endswith(".txt"):
            shutil.copy(
                os.path.join(base_dir, filename), os.path.join(new_dir, filename)
            )
        else:
            img = cv2.imread(os.path.join(base_dir, filename))
            img = preprocessor.process(img)
            try:
                cv2.imwrite(os.path.join(new_dir, filename), img)
            except Exception as e:
                print(
                    f"[ERROR] Failed to save {os.path.join(new_dir, filename)}. Error message: "
                )
                print(e)


def create_preprocessed_dataset(
    preprocessor: ImagePreprocessor, base_dataset_path: str, new_dataset_path: str
) -> None:
    if os.path.exists(new_dataset_path):
        raise FileExistsError(
            f"The {new_dataset_path} is already used delete this directory or pass unused path"
        )
    if not os.path.exists(base_dataset_path):
        raise FileNotFoundError(f"{base_dataset_path} not found")

    os.mkdir(new_dataset_path)

    base_subdirs = [
        subdir for subdir in os.scandir(base_dataset_path) if subdir.is_dir()
    ]

    for subdir in base_subdirs:
        new_dir = os.path.join(new_dataset_path, subdir.name)
        preprocess_folder(subdir, new_dir, preprocessor)
