import requests
import imgcat
import io
from PIL import Image, ImageDraw
import argparse


def viz_bbox(image, outputs):
    for output in outputs:
        draw = ImageDraw.Draw(image)
        draw.rectangle(((output['bbox']['xmin'], output['bbox']['ymin']),
                       (output['bbox']['xmax'], output['bbox']['ymax'])),
                       outline='red')
        text = '{0}({1:.1f}%)'.format(
            output['class'], output['probability']*100)
        draw.text((output['bbox']['xmin'], output['bbox']['ymin'] - 8),
                  text, fill='black')
    return image


def arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1', type=str)
    parser.add_argument('--port', type=int, default=5000)
    return  parser.parse_args()


def main():
    args = arg()
    #REST_API_URL = "http://localhost:5000/predict"
    REST_API_URL = "http://{}:{}/predict".format(args.host, args.port)
    IMAGE_PATH = "dog.jpg"

    with open(IMAGE_PATH, "rb") as f:
        image = f.read()
    payload = {"image": image}

    r = requests.post(REST_API_URL, files=payload).json()

    if r["success"]:
        for (i, result) in enumerate(r["predictions"]):
            print("{}. {}: {:.4f}".format(i + 1, result["class"],
                result["probability"]))
        image = viz_bbox(Image.open(io.BytesIO(image)), r["predictions"])
        imgByteArr = io.BytesIO()
        image.save(imgByteArr, format='PNG')
        imgcat.imgcat(imgByteArr.getvalue())

    else:
        print("Request failed")


if __name__ == '__main__':
    main()
