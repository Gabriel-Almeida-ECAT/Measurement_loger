# as dicovered now vision cloud is incapable of reliably detecting the number, search how to use the 'Vertex AI' API

import os

from MeasumentLogerLib import data_handler as dh
from google.cloud import vision

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'teste-text-ocr-0d8011183118.json'

def detect_numbers(path):
    client = vision.ImageAnnotatorClient()

    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    print("Texts:")

    for text in texts:
        print(f'\n"{text.description}"')

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )


if __name__ == '__main__':
    imgs_path = dh.get_frames_path(6)
    for img_path in imgs_path:
        print("===============================================================================")
        print(f"# FILE {img_path}: ")
        print(f"Detected text: {detect_numbers(img_path)}")