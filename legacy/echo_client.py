import argparse
import socket
import json

def arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default=4000, type=int, help='port')
    parser.add_argument('--host', default='localhost', help='host')
    return parser.parse_args()

def main():
    args = arg()
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((args.host, args.port))

    senddict = {}
    senddict['frame'] = str(5)
    senddict['img'] = 'hoge'
    sendjson = json.dumps(senddict)
    client.send(sendjson.encode('utf-8'))

    response = client.recv(4096).decode('utf-8')
    print('response {}'.format(response))

if __name__ == '__main__':
    main()
