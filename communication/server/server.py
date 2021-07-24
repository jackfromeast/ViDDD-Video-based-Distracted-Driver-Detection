from flask_socketio import SocketIO
import cv2
from flask import Flask, render_template, request
import threading
import base64
import re
import numpy as np
from queue import Queue

app = Flask(__name__)
socketio = SocketIO(app)
global frames_count, clips_count, videoWriter, video_buffer, frames_buffer
frames_count = 1
clips_count = 1

video_buffer = Queue()
frames_buffer = []

save_path = './test/'
clip_file_name = 'test_' + str(clips_count) + '.mp4'
videoWriter = cv2.VideoWriter(save_path+clip_file_name, cv2.VideoWriter_fourcc(*'mp4v'),
                              30,(1080, 720))

@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')


@socketio.on('connect', namespace='/web')
def connect_web():
    print('[INFO] Web client connected: {}'.format(request.sid))


@socketio.on('disconnect', namespace='/web')
def disconnect_web():
    print('[INFO] Web client disconnected: {}'.format(request.sid))


@socketio.on('connect', namespace='/cv')
def connect_cv():
    print('[INFO] CV client connected: {}'.format(request.sid))


@socketio.on('disconnect', namespace='/cv')
def disconnect_cv():
    print('[INFO] CV client disconnected: {}'.format(request.sid))


@socketio.on('cv2server')
def handle_cv_message(message):
    print("[INFO]Receiving data from Client.")
    # socketio.emit('server2web', message, namespace='/web')
    
    global frames_count, clips_count, videoWriter, video_buffer, frames_buffer
    frames_count += 1
    frames_buffer.append(message['image'])

    print(frames_count)
    if frames_count % 70 == 0:
        video_buffer.put((frames_buffer, clips_count))

        print('[INFO]worker_save_video thread started.')
        worker_save_video = threading.Thread(target=save_video, daemon=True, args=(video_buffer.get()))
        worker_save_video.start()
        

        frames_buffer = []
        clips_count += 1


def save_video(frames, clips_count):
    save_path = './raw_dataset/test/'
    clip_file_name = 'test_' + str(clips_count) + '.mp4'
    print(save_path+clip_file_name)
    videoWriter = cv2.VideoWriter(save_path+clip_file_name, cv2.VideoWriter_fourcc(*'mp4v'), 30,(1080, 720))

    print('[INFO]Start to save videos.')
    for raw_frame in frames:
        frame = image_handler(raw_frame)
        videoWriter.write(frame)

    videoWriter.release()
    print("worker_save_video Done!")


def image_handler(raw_image):
    items = re.split('[;,]', raw_image)
    image_in_64 = ''.join(items[2:])

    img = base64.b64decode(image_in_64)

    return cv2.imdecode(np.array(bytearray(img), dtype='uint8'), cv2.IMREAD_UNCHANGED)

if __name__ == "__main__":
    server_ip = '127.0.0.1'
    port = 5001
    print('[INFO] Starting server at http://{}:{}'.format(server_ip, port))
    socketio.run(app=app, host=server_ip, port=port)
