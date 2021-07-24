from flask_socketio import SocketIO, join_room, leave_room
from flask_redis import FlaskRedis
import cv2
from flask import Flask, render_template, request
import threading
import base64
import re
import numpy as np
from queue import Queue

app = Flask(__name__)
app.config['REDIS_URL'] = 'redis://127.0.0.1:6379/0'
redis_client = FlaskRedis(app, decode_responses=True)
socketio = SocketIO(app)


global frames_count, clips_count, videoWriter, video_buffer, frames_buffer
frames_count = 1
clips_count = 1
video_buffer = Queue()
frames_buffer = []

MODEL = None


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


@socketio.on('connect', namespace='/c2s')
def connect_cv():
    # join_room('frame_senders', sid=request.sid)
    print('[INFO] CV client connected: {} on c2s namespace.'.format(request.sid))


@socketio.on('disconnect', namespace='/c2s')
def disconnect_cv():
    # leave_room('frame_senders', sid=request.sid)
    print('[INFO] CV client disconnected: {} on c2s namespace'.format(request.sid))

@socketio.on('connect', namespace='/s2c')
def connect_cv():
    # join_room('result_receivers', sid=request.sid)
    print('[INFO] CV client connected: {} on s2c namespace.'.format(request.sid))

@socketio.on('disconnect', namespace='/s2c')
def disconnect_cv():
    # leave_room('result_receivers', sid=request.sid)
    print('[INFO] CV client disconnected: {} on s2c namespace'.format(request.sid))


@socketio.on('get_results', namespace='/s2c')
def send_results(message):
    results = redis_client.get('results')

    print("[Results Replier]Sending results to the client.")
    socketio.emit('results', results, namespace='/s2c')


@socketio.on('send_frame', namespace='/c2s')
def handle_cv_message(message):
    # print("[INFO]Receiving data from Client.")
    # socketio.emit('server2web', message, namespace='/web')
    
    global frames_count, clips_count, videoWriter, video_buffer, frames_buffer
    frames_count += 1
    frames_buffer.append(message['image'])

    if frames_count % 70 == 0:
        video_buffer.put((frames_buffer, clips_count))

        print('[Clips Dumper] worker_save_video thread started.')
        worker_save_video = threading.Thread(target=save_video, daemon=True, args=(video_buffer.get()))
        worker_save_video.start()
        

        frames_buffer = []
        clips_count += 1


def save_video(frames, clips_count):
    save_path = './raw_dataset/test/'
    clip_file_name = 'test_' + str(clips_count) + '.mp4'
    
    videoWriter = cv2.VideoWriter(save_path+clip_file_name, cv2.VideoWriter_fourcc(*'mp4v'), 30,(1080, 720))

    # print('[INFO]Start to save videos.')
    for raw_frame in frames:
        frame = image_handler(raw_frame)
        videoWriter.write(frame)

    videoWriter.release()
    print("[Clips Dumper] the %s video clip has saved." % save_path+clip_file_name)

    model_predict(save_path+clip_file_name)


def image_handler(raw_image):
    items = re.split('[;,]', raw_image)
    image_in_64 = ''.join(items[2:])

    img = base64.b64decode(image_in_64)

    return cv2.imdecode(np.array(bytearray(img), dtype='uint8'), cv2.IMREAD_UNCHANGED)


def model_setup():
    global MODEL
    if MODEL is None:
        # TODO: MODEL LOAD
        print('[Predictor]The MODEL has already loaded.')


def model_predict(video_path):
    global MODEL
    if MODEL is None:
        model_setup()
    
    # TODO: result = data_load_predict(video_path)
    result = ('safe_drive', 67.32)

    clip_index = re.split('[_./]', video_path)[-2]

    data = clip_index + '-' + result[0] +  '-' + str(result[1]) + ';'

    save_result(data)


def save_result(new_results):
    previous_results = redis_client.get('results')

    if previous_results is None:
        previous_results = new_results
    else:

        new_results = previous_results + new_results

    redis_client.set('results', new_results)
    print('[Predictor]The result has updated.')



if __name__ == "__main__":
    server_ip = '127.0.0.1'
    port = 5001
    print('[INFO] Starting server at http://{}:{}'.format(server_ip, port))
    socketio.run(app=app, host=server_ip, port=port)
