import cv2
import os
from detection.bounding_boxes.shapes import read_shapes, to_bbox, to_obb
from detection.dataset_utils.samples_creator import process_lines_and_terms, assign_terms_to_words


if __name__ == "__main__":
    lines = read_shapes(r"E:\Dyploma\Latina\LatinaProject\datasets\lines-obb-clean\train\AUR_891_III_9-101 (text).txt",
                        transform_func=to_obb, class_nums="0")
    words = read_shapes(r"E:\Dyploma\Latina\LatinaProject\datasets\seven-classes\train\AUR_891_III_9-101 (text).txt",
                            transform_func=to_bbox, class_nums=["1", "2", "3", "6"])
    image_path = r"E:\Dyploma\Latina\LatinaProject\datasets\seven-classes\train\AUR_891_III_9-101 (text).jpg"
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    file_path = 'terms/AUR_891_III_9-101.txt'

    file_name = os.path.splitext(os.path.basename(image_path))[0]
    output_folder = os.path.join("word_dataset", file_name)
    os.makedirs(output_folder, exist_ok=True)

    sorted_line_to_words, terms = process_lines_and_terms(words, lines, image, file_path)

    assign_terms_to_words(sorted_line_to_words, words, terms, image, output_folder)
