from detection.model_wrapper import YoloWrapper
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--obb", action="store_true",
        help="Flag to use OBB (Oriented Bounding Box) model."
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=300,
        help="Number of epochs to train. Defaults to 300"
    )
    parser.add_argument(
        "--img_size", type=int, default=608,
        help="Image size. Defaults to 608"
    )
    parser.add_argument(
        "--data_file", required=True, type=str,
        help="Path to data yaml file to train a model."
    )
    parser.add_argument(
        "--angle_aug", type=int, default=0,
        help="Angle augmentation value. Defaults to 0"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.obb:
        model = YoloWrapper("yolov8m-obb.pt", "cuda:0")
    else:
        model = YoloWrapper("yolov8m.pt", "cuda:0")

    model.train(
        args.data_file,
        epochs=args.epochs,
        img_size=args.img_size,
        angle_aug=args.angle_aug
    )
