import requests
import imgcat
import io
from PIL import Image, ImageDraw


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

def main():
    REST_API_URL = "http://localhost:5000/predict"
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
