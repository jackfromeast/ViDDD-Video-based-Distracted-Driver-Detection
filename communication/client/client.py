import time
import argparse
import socketio
import threading
from queue import Queue
import cv2
import base64
import json

global sio, results_file
sio = socketio.Client()

results_file = None
write_lock = False

@sio.event
def connect():
    print('[INFO] Successfully connected to server.')


@sio.event
def connect_error():
    print('[INFO] Failed to connect to server.')


@sio.event
def disconnect():
    print('[INFO] Disconnected from server.')


@sio.on('results', namespace='/s2c')
def save_result(message):
    print('[INFO]Received a result from server. Start to save it.')

    global results_file, write_lock
    if results_file is None:
        results_file = open('./communication/client/results.txt', 'w')
    
    while True:
        if write_lock == False:
            break
    
    write_lock  = True
    results_file.write(message)
    write_lock  = False


class CVClient(object):
    def __init__(self, server_addr, stream_fps):
        self.server_addr = server_addr
        self.server_port = 5001
        self._stream_fps = stream_fps
        self._last_update_t = time.time()
        self._wait_t = (1/self._stream_fps)

        self.count = 0

    def setup(self):
        print('[INFO] Connecting to server http://{}:{}...'.format(
            self.server_addr, self.server_port))

        # Channel c2s for client sending video stream
        sio.connect(
                'http://{}:{}'.format(self.server_addr, self.server_port),
                namespaces=['/c2s', '/s2c'])   

        time.sleep(3)
        return self

    def _convert_image_to_jpeg(self, image):
        # Encode frame as jpeg
        frame = cv2.imencode('.jpg', image)[1].tobytes()
        # Encode frame in base64 representation and remove
        # utf-8 encoding
        frame = base64.b64encode(frame).decode('utf-8')
        return "data:image/jpeg;base64,{}".format(frame)

    def send_data(self, frame, text=None):
        cur_t = time.time()
        if cur_t - self._last_update_t > self._wait_t:
            self._last_update_t = cur_t
            if frame.shape != (1080, 720, 3):
                frame = cv2.resize(frame, (1080, 720))
            sio.emit(
                    'send_frame',
                    {
                        'image': self._convert_image_to_jpeg(frame),
                        'text': '<br>'.join(text)
                    },
                    namespace='/c2s')
            self.count += 1

    def check_exit(self):
        pass

    def close(self):
        sio.disconnect()


class Camera(object):

    def __init__(self, fps=30, video_source=0, streamer=None):
        print(f"[INFO]Initializing camera class with {fps} fps and video_source={video_source}")
        self.fps = fps
        self.video_source = video_source
        self.camera = cv2.VideoCapture(self.video_source)

        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # the buffer can only contain 1 frame to ensure that everytime you read from the camera, it is the lastest frame.
        self.camera.set(3, 1080) # set width
        self.camera.set(4, 720)  # set height
        
        self.frames_q = Queue()
        self.isrunning = False

        self.streamer = streamer

        self.capture_thread = None
        self.sending_thread = None
        
    def run(self):
        print("[INFO]Perparing threads")
    
        print("[INFO]Creating capture_thread")
        self.capture_thread = threading.Thread(target=self._capture_loop,daemon=True)
        print("[INFO]Creating sending_thread")
        self.sending_thread = threading.Thread(target=self.__send_frame,daemon=True)

        print("[INFO]Starting thread")
        self.isrunning = True

        self.capture_thread.start()
        self.sending_thread.start()
        print("[INFO]Threads started")

    def _capture_loop(self):
        dt = 1/self.fps
        print("[INFO]Observation started")
        while self.isrunning:
            v,frame = self.camera.read()
            if v:
                # self.frames.append(im)
                self.frames_q.put(frame)

        time.sleep(dt)
        print("[INFO]Thread stopped successfully")

    def __send_frame(self):
        while self.isrunning:
            time.sleep(0.001)
            if not self.frames_q.empty():
                frame = self.frames_q.get()
                self.streamer.send_data(frame, ['success'])

    def stop(self):
        print("[INFO]Stopping thread")
        self.isrunning = False
        self.release()
    
    def release(self):
        print("[INFO]Release camera")
        self.camera.release()


class Retriever(object):
    
    def __init__(self, time_interval=5):
        self.current_result = None
        self.time_interval = time_interval

        self.retrieve_thread = None
        self.isrunning = False

    def run(self):
        print("[INFO]Perparing threads")
    
        print("[INFO]Creating retrieve_thread")
        self.retrieve_thread = threading.Thread(target=self.__retrieve_results,daemon=True)
       
        print("[INFO]Starting thread")
        self.isrunning = True

        self.retrieve_thread.start()
        print("[INFO]Threads started")

    def __retrieve_results(self):
        while True:
            if self.isrunning == False:
                break
            time.sleep(self.time_interval)
            sio.emit('get_results', 'Give me the current results!', namespace='/s2c')
            print('[INFO]Retrieve results from the server.')

    
    def stop(self):
        print("[INFO]Stopping thread")
        self.isrunning = False

            

def main(camera, server_addr, stream_fps):
    streamer = CVClient(server_addr, stream_fps).setup()

    try:
        camera = Camera(stream_fps, camera, streamer)
        start = time.time()
        camera.run()

        retriever = Retriever()
        retriever.run()

        # Stunked and keep threads running
        camera.capture_thread.join()
        camera.sending_thread.join()
        retriever.retrieve_thread.join()

        print("[INFO]Threads stoped")

    except KeyboardInterrupt:
        print("[INFO]Exiting...")

    finally:
        end = time.time()
        if streamer is not None:
            streamer.close()

        camera.stop()
        retriever.stop()

        global results_file
        if results_file is not None:
            results_file.close()

        print("[INFO]elapsed time: {:.2f}".format(end-start))
        print("[INFO]approx. FPS: {:.2f}".format(streamer.count/(end-start)))

        print("[INFO]Program Ending")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='alwaysAI Video Streamer')
    parser.add_argument(
            '--camera', type=int, default='0',
            help='The camera index to stream from.')
    parser.add_argument(
            '--use-streamer',  action='store_true',
            help='Use the embedded streamer instead of connecting to the server.')
    parser.add_argument(
            '--server-addr',  type=str, default='127.0.0.1',
            help='The IP address or hostname of the SocketIO server.')
    parser.add_argument(
            '--stream-fps',  type=float, default=60.0,
            help="The rate to send frames to the server. Usually can't reach the rate")
    parser.add_argument(
            '--clips-len',  type=int, default=70)

    args = parser.parse_args()

    main(args.camera, args.server_addr, args.stream_fps)
