import argparse
import chainer
from chainercv.datasets import voc_bbox_label_names
from chainercv.links import SSD300
from chainercv import utils
import os
import socket
from skimage import io as skio
import json
import base64
import io
import zlib

def arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--pretrained_model', default='voc0712')
    parser.add_argument('--img', default='img.jpg')
    parser.add_argument('--host', default="localhost", help='sergver address')
    parser.add_argument('--port', '-p', default=4000, type=int, help='port')
    return  parser.parse_args()


class Server(object):
    def __init__(self, host='localhost', port=3000, verbose=False):
        self.serversock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.serversock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.serversock.bind((host, port))
        self.serversock.listen(10)

    def predict(self, data):
        raise NotImplementedError

    def run(self):
        while True:
            print('waiting for connections...')
            clientsock, client_address = self.serversock.accept()
            rcvdata = b''
            i = 0
            max_iter = int(clientsock.recv(512).decode('utf-8'))
            print('Got max iteration number: {}'.format(max_iter))
            while max_iter >= i:
                chunk = clientsock.recv(4096)
                if not chunk:
                    break
                rcvdata += chunk
                i+=1
            print(len(rcvdata))

            json_str = rcvdata.decode('utf-8')
            print(len(json_str))
            json_data = json.loads(json_str)
            send_data = {}
            status_data = {}
            try:
                data = self.predict(json_data)
                status_data['code'] = str(200)
                status_data['details'] = ''

            except:
                import traceback
                error = traceback.format_exc()
                data = {}
                data['None'] = []
                status_data['code'] = str(400)
                status_data['details'] = error
                print(error)

            finally:
                send_data["status"] = status_data
                send_data["data"] = data
                converted_data = json.dumps(send_data, ensure_ascii=False)
                clientsock.sendall(converted_data.encode('utf-8'))
            clientsock.close()

class WeightServer(Server):
    def __init__(self, host, port, model):
        super().__init__(host, port)
        self.infmodel = model

    def predict(self, data):
        # data must be in json format.
        print(data['frame'])
        img = base64.b64decode(data['img'].encode('utf-8'))
        img = io.BytesIO(img)
        img = skio.imread(img)
        img = img.transpose(2, 0, 1)

        print('image shape: {}'.format(img.shape))
        bboxes, labels, scores = self.infmodel.predict([img])

        send_dict = {}
        result_data = {}
        object_list = []
        result_data['frame'] = data['frame']
        for num, bbox in enumerate(bboxes):
            object_dict = {}
            bndbox = {}
            object_dict['name'] = voc_bbox_label_names[int(labels[num])]
            object_dict['prob'] = str(float(scores[num]))
            bndbox['ymin'] = str(int(bbox[0][0]))
            bndbox['xmin'] = str(int(bbox[0][1]))
            bndbox['ymax'] = str(int(bbox[0][2]))
            bndbox['xmax'] = str(int(bbox[0][3]))
            object_dict['bndbox'] = bndbox
            object_list.append(object_dict)

        result_data['object'] = object_list


def main():
    args = arg()
    chainer.config.train = False

    model = SSD300(
        n_fg_class=len(voc_bbox_label_names),
        pretrained_model=args.pretrained_model
    )

    if args.gpu >= 0:
        model.to_gpu(args.gpu)
        chainer.cuda.get_device(args.gpu).use()

    ws =  WeightServer(host=args.host, port=args.port, model=model)
    ws.run()


if __name__ == '__main__':
    main()
