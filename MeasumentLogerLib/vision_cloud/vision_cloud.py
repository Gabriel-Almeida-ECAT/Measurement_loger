#source codes: https://github.com/GoogleCloudPlatform/python-docs-samples/blob/HEAD/vision/snippets/detect/detect.py

import os

from MeasumentLogerLib import data_handler as dh
from google.cloud import vision

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'teste-text-ocr-0d8011183118.json'

def detect_text(path):
    client = vision.ImageAnnotatorClient()

    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )

    '''text = response.text_annotations
    print("Texts:")

    for text in texts:
        print(f'\n"{text.description}"')'''

    return response.full_text_annotation.text


def detect_labels(path):
    client = vision.ImageAnnotatorClient()

    # [START vision_python_migration_label_detection]
    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.label_detection(image=image)
    labels = response.label_annotations
    print("Labels:")

    for label in labels:
        print(label.description)

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )


if __name__ == '__main__':
    imgs_path = dh.get_frames_path(6)
    for img_path in imgs_path:
        print("-------------------------------------------------------------------------------")
        print(f"# FILE {img_path}: ")
        print(f"Detected text: {detect_text(img_path)}")
    #print(f"Detected text: {detect_text('teste.png')}")