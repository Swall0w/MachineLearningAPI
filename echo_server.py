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
    serversock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serversock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    serversock.bind((args.host, args.port))
    serversock.listen(10)

    while True:
        print('waiting for connections...')
        clientsock, client_address = serversock.accept()

        while True:
            rcvmsg = clientsock.recv(4096)
            if not rcvmsg:
                break

            try:
                json_data = rcvmsg.decode('utf-8')
                print(json_data)
                json_data = json.loads(json_data)
                ret = {}
                ret['frame'] = str(int(json_data['frame']) + 1)
                ret['img'] = 'fuga' * int(json_data['frame'])
            except:
                import traceback
                error = traceback.format_exc()
                print(error)
                ret = {}
                ret['frame'] = str(0)
                ret['img'] = error
            finally:
                ret = json.dumps(ret).encode('utf-8')
                clientsock.sendall(ret)
                print('done')
        clientsock.close()
        print('closed')

if __name__ == '__main__':
    main()
