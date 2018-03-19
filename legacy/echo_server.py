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

            ret = {}
            status = {}
            data = {}

            try:
                json_data = rcvmsg.decode('utf-8')
                print(json_data)
                json_data = json.loads(json_data)
#                ret['frame'] = str(int(json_data['frame']) + 1)
#                ret['img'] = 'fuga' * int(json_data['frame'])
                data['frame'] = json_data['frame']
                objct_all = []
                obj1 = {}
                obj1['name'] = 'dump'
                obj1['prob'] = str(0.8533421431)
                bndbox={"xmin": str(10), "xmax":str(100),
                        "ymin": str(15), "ymax":str(200)}
                obj1['bndbox'] = bndbox
                objct_all.append(obj1)
                obj2 = {}
                obj2['name'] = 'excavator'
                obj2['prob'] = str(0.2501)
                bndbox={"xmin": str(100), "xmax":str(150),
                        "ymin": str(30), "ymax":str(50)}
                obj2['bndbox'] = bndbox
                objct_all.append(obj2)
                data['object'] = objct_all

                status['code'] = str(200)
                status['details'] = ""
                
            except:
                import traceback
                error = traceback.format_exc()
                print(error)
                status['code'] = str(400)
                status['details'] = error

            finally:
                ret['status'] = status
                ret['data'] = data
                ret = json.dumps(ret).encode('utf-8')
                clientsock.sendall(ret)
                print('done')
        clientsock.close()
        print('closed')

if __name__ == '__main__':
    main()
