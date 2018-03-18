import chainer
from chainercv.datasets import voc_bbox_label_names
from chainercv.links import SSD300
from chainercv import utils
from PIL import Image
import numpy as np
import flask
import io
from skimage import io as skio


app = flask.Flask(__name__)
model = None

def load_model():
    global model
    model = SSD300(
        n_fg_class=len(voc_bbox_label_names),
        pretrained_model='voc0712')


def prepare_image(image):
    img = image.transpose(2, 0, 1)
    return img


@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = skio.imread(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = prepare_image(image)

            # classify the input image and then initialize the list
            # of predictions to return to the client
            bboxes, labels, scores = model.predict([image])

            data["predictions"] = []

            # loop over the results and add them to the list of
            # returned predictions
            for index, bbox in enumerate(bboxes[0]):
                r = {"label": voc_bbox_label_names[int(labels[0][index])],
                     "bbox": {"ymin": int(bbox[0]),
                              "xmin": int(bbox[1]),
                              "ymax": int(bbox[2]),
                              "xmax": int(bbox[3])},
                     "probability": float(scores[0][index])
                     }
                data["predictions"].append(r)

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)

if __name__ == "__main__":
    print("Run server")
    load_model()
    app.run()
