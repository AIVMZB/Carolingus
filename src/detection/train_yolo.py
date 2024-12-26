from argparse import ArgumentParser

from detection.model_wrapper import YoloWrapper


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data-file", type=str, required=True, 
                        help="path to .yaml file to train a YOLO model")
    parser.add_argument("--model", type=str, default="yolov8m.pt", 
                        help="Name of model to train. Default value is yolov8m.pt")
    parser.add_argument("-e", "--epochs", type=int, default=300,
                        help="Number of epochs to train a model")
    parser.add_argument("--image-size", type=int, default=768,
                        help="Image size")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = YoloWrapper(args.model, "cuda:0")
    model.train(
        args.data_file, 
        args.epochs, 
        args.image_size
    )
