import os
import shutil
from typing import Optional, NamedTuple


class DetectObject(NamedTuple):
    class_number: int
    bbox_data: list


def clear_dataset_from_classes(
    dataset_path: str,
    new_dataset_path: str,
    classes_to_remove: tuple,
    train_dir: str,
    validation_dir: str,
    test_dir: Optional[str] = None,
) -> None:

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset is not found in {dataset_path}")

    os.mkdir(new_dataset_path)

    clear_folder_from_classes(
        os.path.join(dataset_path, train_dir),
        os.path.join(new_dataset_path, train_dir),
        classes_to_remove,
    )
    clear_folder_from_classes(
        os.path.join(dataset_path, validation_dir),
        os.path.join(new_dataset_path, validation_dir),
        classes_to_remove,
    )

    if test_dir:
        clear_folder_from_classes(
            os.path.join(dataset_path, test_dir),
            os.path.join(new_dataset_path, test_dir),
            classes_to_remove,
        )


def clear_folder_from_classes(
    folder_path: str, new_folder_class: str, classes_to_remove: tuple
) -> None:
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found in {folder_path}")

    os.mkdir(new_folder_class)

    for file in os.listdir(folder_path):
        if not file.endswith(".txt"):
            shutil.copyfile(
                os.path.join(folder_path, file), os.path.join(new_folder_class, file)
            )
            continue

        parsed_objects = parse_label_file(
            os.path.join(folder_path, file), classes_to_remove
        )
        objects_to_file(os.path.join(new_folder_class, file), parsed_objects)


def parse_label_file(file_path: str, classes_to_remove: list) -> list[DetectObject]:
    detect_objects = []
    with open(file_path, "r") as f:
        for line in f:
            values = line.split(" ")
            class_number = int(values[0])

            if class_number in classes_to_remove:
                continue

            bbox_data = [float(coord) for coord in values[1:]]
            detect_objects.append(DetectObject(class_number, bbox_data))

    return detect_objects


def objects_to_file(filepath: str, detect_objects: list[DetectObject]) -> None:
    with open(filepath, "w") as f:
        for detect_object in detect_objects:
            f.write(str(detect_object.class_number))
            f.write(" ")

            for i in range(len(detect_object.bbox_data)):
                if i != len(detect_object.bbox_data) - 1:
                    f.write(f"{detect_object.bbox_data[i]} ")
                else:
                    f.write(f"{detect_object.bbox_data[i]}")

            f.write("\n")


def clear_dataset_from_outliers(
    dataset_folder: str,
    new_dataset_path: str,
    outlier_files: list,
    train_dir: str,
    validation_dir: str,
    test_dir: Optional[str] = None,
) -> None:
    if not os.path.exists(new_dataset_path):
        os.mkdir(new_dataset_path)

    clear_dir_from_outliers(
        os.path.join(dataset_folder, train_dir),
        os.path.join(new_dataset_path, train_dir),
        outlier_files,
    )
    clear_dir_from_outliers(
        os.path.join(dataset_folder, validation_dir),
        os.path.join(new_dataset_path, validation_dir),
        outlier_files,
    )
    if test_dir is not None:
        clear_dir_from_outliers(
            os.path.join(dataset_folder, test_dir),
            os.path.join(new_dataset_path, test_dir),
            outlier_files,
        )


def clear_dir_from_outliers(dir: str, new_dir: str, outliers: list):
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)

    for filename in os.listdir(dir):
        if filename.split(".")[0] not in outliers:
            shutil.copyfile(
                os.path.join(dir, filename), os.path.join(new_dir, filename)
            )
            print(f"Copying {os.path.join(dir, filename)}")
        else:
            print(f"Ignoring {os.path.join(dir, filename)}")
