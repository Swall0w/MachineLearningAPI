import argparse
import socket
import json
import base64


def arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default="localhost", help='sergver address')
    parser.add_argument('--port', '-p', default=4000, type=int, help='port')
    parser.add_argument('--img', nargs=1, type=str, help='image files')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='debug mode if this flag is set (default: False)')
    return parser.parse_args()


def main():
    args = arg()
    senddict = {}
    imgfile = args.img[0]

    senddict['frame'] = str(5)
    with open(imgfile, 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read())

    senddict['img'] = encoded_string.decode('utf-8')

    if args.verbose:
        print(senddict)
    img_json = json.dumps(senddict)


    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((args.host, args.port))
    img_json = img_json.encode('utf-8')

    length_bytes = len(img_json)

    header = (str(length_bytes)+':').encode('utf-8')
    status = client.sendall(header+img_json)
    print('Send data... byte: {},  status: {}'.format(length_bytes, status))

    response = client.recv(4096).decode('utf-8')
    print(response)


if __name__ == '__main__':
    main()
