# -*- coding: UTF-8 -*-

import time
import os
import pyttsx3
import shutil
from tqdm import tqdm
import subprocess
import argparse
import cv2
import audio
import torch
import face_detection
from models import Wav2Lip
import numpy as np
from idcontrol import IDManager
import threading
from flask import jsonify, send_file, Flask, request, make_response, render_template,redirect,url_for
from ttskit import sdk_api


app = Flask(__name__)
IDM = IDManager()
Lock = threading.Lock()

# ---------------------------------------------------------------------------------------

# todo myqr

parser = argparse.ArgumentParser(
    description='Inference code to lip-sync videos in the wild using Wav2Lip models')

# todo 改参数
parser.add_argument('--checkpoint_path', type=str,
                    help='Name of saved checkpoint to load weights from',
                    default='checkpoints/wav2lip.pth')
parser.add_argument('--face', type=str,
                    help='Filepath of video/image that contains faces to use',
                    default='results/{}/input_video.mp4')  # todo results/taskid/input_video.mp4
parser.add_argument('--audio', type=str,
                    help='Filepath of video/audio file to use as raw audio source',
                    default='results/{}/result.wav')  # todo results/audioid/result.wav
parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.',
                    default='results/{}/output_video.mp4')  # todo results/taskid/output_video.mp4

parser.add_argument('--static', type=bool,
                    help='If True, then use only first video frame for inference', default=False)
parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)',
                    default=25., required=False)

parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0],
                    help='Padding (top, bottom, left, right). Please adjust to include chin at least')

parser.add_argument('--face_det_batch_size', type=int,
                    help='Batch size for face detection', default=16)
parser.add_argument('--wav2lip_batch_size', type=int,
                    help='Batch size for Wav2Lip model(s)', default=128)

parser.add_argument('--resize_factor', default=1, type=int,
                    help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1],
                    help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. '
                    'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')

parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1],
                    help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
                    'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')

parser.add_argument('--rotate', default=False, action='store_true',
                    help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
                    'Use if you get a flipped result, despite feeding a normal looking video')

parser.add_argument('--nosmooth', default=False, action='store_true',
                    help='Prevent smoothing face detections over a short temporal window')

args = parser.parse_args()
args.img_size = 96

mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))


def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i: i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes


def face_detect(images):
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
                                            flip_input=False, device=device)

    batch_size = args.face_det_batch_size

    while 1:
        predictions = []
        try:
            face_bar = [0, int(np.ceil(len(images)/batch_size))]
            for i in tqdm(range(0, len(images), batch_size)):
                face_bar[0] += 1
                Lock.acquire()
                IDM.wait_list[0]['process'] = face_bar[0] / face_bar[1] * 90
                Lock.release()

                predictions.extend(detector.get_detections_for_batch(
                    np.array(images[i:i + batch_size])))
        except RuntimeError:
            if batch_size == 1:
                raise RuntimeError(
                    'Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = args.pads
    for rect, image in zip(predictions, images):
        if rect is None:
            # check this frame where the face was not detected.
            cv2.imwrite('temp/faulty_frame.jpg', image)
            raise ValueError(
                'Face not detected! Ensure the video contains a face in all the frames.')

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)

        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    if not args.nosmooth:
        boxes = get_smoothened_boxes(boxes, T=5)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)]
               for image, (x1, y1, x2, y2) in zip(images, boxes)]

    del detector
    return results


def datagen(frames, mels):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if args.box[0] == -1:
        if not args.static:
            # BGR2RGB for CNN face detection
            face_det_results = face_detect(frames)
        else:
            face_det_results = face_detect([frames[0]])
    else:
        print('Using the specified bounding box instead of face detection...')
        y1, y2, x1, x2 = args.box
        face_det_results = [
            [f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

    for i, m in enumerate(mels):
        idx = 0 if args.static else i % len(frames)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx].copy()

        face = cv2.resize(face, (args.img_size, args.img_size))

        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= args.wav2lip_batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, args.img_size//2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(
                mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, args.img_size//2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(
            mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

        yield img_batch, mel_batch, frame_batch, coords_batch


def _load(checkpoint_path):
    if device == 'cpu':
        checkpoint = torch.load(checkpoint_path,map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_model(path):
    model = Wav2Lip()
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()


# ---------------------------------------------------------------------------------------

def Wav2LipTask(taskid, audioid):
    args.face = 'results/{}/input_video.mp4'.format(taskid)
    args.audio = 'results/{}/result.wav'.format(audioid)
    args.outfile = 'results/{}/output_video.mp4'.format(taskid)

    if not os.path.isfile(args.face):
        raise ValueError(
            '--face argument must be a valid path to video/image file')

    elif args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
        full_frames = [cv2.imread(args.face)]
        fps = args.fps

    else:
        video_stream = cv2.VideoCapture(args.face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)

        print('Reading video frames...')

        full_frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            if args.resize_factor > 1:
                frame = cv2.resize(
                    frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor))

            if args.rotate:
                frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

            y1, y2, x1, x2 = args.crop
            if x2 == -1:
                x2 = frame.shape[1]
            if y2 == -1:
                y2 = frame.shape[0]

            frame = frame[y1:y2, x1:x2]

            full_frames.append(frame)

    print("Number of frames available for inference: "+str(len(full_frames)))

    if not args.audio.endswith('.wav'):
        print('Extracting raw audio...')
        command = 'ffmpeg -y -i {} -strict -2 {}'.format(
            args.audio, 'results/'+audioid+'/temp.wav')
        args.audio = 'results/'+audioid+'/temp.wav'

        os.system(command)

    wav = audio.load_wav(args.audio, 16000)
    mel = audio.melspectrogram(wav)
    # print(mel.shape)

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError(
            'Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

    mel_chunks = []
    mel_idx_multiplier = 80./fps
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
        i += 1

    print("Length of mel chunks: {}".format(len(mel_chunks)))

    full_frames = full_frames[:len(mel_chunks)]

    batch_size = args.wav2lip_batch_size
    gen = datagen(full_frames.copy(), mel_chunks)

    pbar = tqdm(gen, total=int(np.ceil(float(len(mel_chunks))/batch_size)))
    inference_bar = [0, int(np.ceil(float(len(mel_chunks))/batch_size))]

    for i, (img_batch, mel_batch, frames, coords) in enumerate(pbar):
        if i == 0:
            frame_h, frame_w = full_frames[0].shape[:-1]
            out = cv2.VideoWriter('results/'+taskid+'/result.avi',
                                  cv2.VideoWriter_fourcc(*'MJPG'), fps, (frame_w, frame_h))

        inference_bar[0] += 1
        Lock.acquire()
        IDM.wait_list[0]['process'] = inference_bar[0] / inference_bar[1] * 10 + 90
        Lock.release()

        img_batch = torch.FloatTensor(
            np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(
            np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

        with torch.no_grad():
            pred = model(mel_batch, img_batch)

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

            f[y1:y2, x1:x2] = p
            out.write(f)

    out.release()

    command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 -y {}'.format(
        args.audio, 'results/'+taskid+'/result.avi', args.outfile)
    os.system(command)

# ---------------------------------------------------------------------------------------


def tts_task(taskid):
    '''
    TTS-后端实现
    '''

    # engine = pyttsx3.init()
    # # 设置新的语音速率，默认200
    # engine.setProperty('rate', 180)
    # # 设置新的语音音量，音量最小为 0，最大为 1，默认1
    # engine.setProperty('volume', 1.0)
    # # 获取当前语音声音的详细信息，这个是获取你电脑上语音识别的语音列表
    # voices = engine.getProperty('voices')
    # # 设置当前语音声音，根据自己的语音列表设置
    # engine.setProperty('voice', voices[0].id)
    # # 语音文本
    out_file = 'results/'+taskid+'/temp.wav'
    final_file = 'results/'+taskid+'/result.wav'
    # 将文本转换成音频，并输出到out_file
    with open('results/'+taskid+'/text.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    with open('results/'+taskid+'/voiceid.txt', 'r', encoding='utf-8') as f:
        voiceid = int(f.read())
    _speaker_dict = {
        1: 'Aibao', 2: 'Aicheng', 3: 'Aida', 4: 'Aijia', 5: 'Aijing',
        6: 'Aimei', 7: 'Aina', 8: 'Aiqi', 9: 'Aitong', 10: 'Aiwei',
        11: 'Aixia', 12: 'Aiya', 13: 'Aiyu', 14: 'Aiyue', 15: 'Siyue',
        16: 'Xiaobei', 17: 'Xiaogang', 18: 'Xiaomei', 19: 'Xiaomeng', 20: 'Xiaowei',
        21: 'Xiaoxue', 22: 'Xiaoyun', 23: 'Yina', 24: 'biaobei'
    }

    print(text, voiceid, _speaker_dict[voiceid])
    wav = sdk_api.tts_sdk(text, speaker=_speaker_dict[voiceid])
    with open(out_file, 'wb') as f:
        f.write(wav)

    # engine.save_to_file(text, out_file)
    # engine.runAndWait()
    # engine.stop()
    index = IDM.search(taskid)
    Lock.acquire()
    IDM.wait_list[index]['process'] = 50.
    Lock.release()

    command = "ffmpeg -i {} -ar 48000 -ac 2  -acodec  pcm_f32le -y {}".format(
        out_file, final_file)
    os.system(command)
    Lock.acquire()
    IDM.wait_list[index]['process'] = 100.
    Lock.release()

# ---------------------------------------------------------------------------------------

"""
@app.route('/uploadtext', methods=['GET', 'POST'])
def uploadtext():
    '''
    上传文字转语音的文字内容
    '''
    # values 对GET和POST通用
    text = request.values.get('text')
    voiceid = request.values.get('voiceid')
    #开始上传，渲染模板
    if request.method == "GET" and not text:
        return make_response(render_template('uploadtext.html'))
    if not text:
        return make_response('Upload Error: No text upload')
    if not voiceid:
        voiceid = '1'

    # 生成唯一id
    taskid = IDM.generate_id('ts')
    print(text, voiceid)

    os.mkdir('results/'+taskid)
    with open('results/'+taskid+'/text.txt', 'w', encoding='utf-8') as f:
        f.write(text)
    with open('results/'+taskid+'/voiceid.txt', 'w', encoding='utf-8') as f:
        f.write(voiceid)

    #! 自动启动任务
    # IDM.add({'task': 'TTS', 'taskid': taskid, 'process': 0})

    return taskid
"""
@app.route('/aivoicestart',methods=['GET', 'POST'])
def aivoicestart():
    '''
    开始任务
    '''

    taskid = request.values.get('taskid')
    # if not os.path.exists('results/'+taskid+'/result.wav'):
    #     with open('results/'+taskid+'/text.txt', 'r') as f:
    #         text = f.read()

    if IDM.search(taskid) == -1 and not IDM.isdone(taskid):
        Lock.acquire()
        IDM.add({'task': 'TTS', 'taskid': taskid, 'process': 0})
        Lock.release()
    # print(IDM.wait_list)

    return make_response(jsonify(dict(taskid=taskid, done=True)))


@app.route('/aivoiceprocess')
def aivoiceprocess():
    '''
    查看进度
    '''

    taskid = request.values.get('taskid')
    # if os.path.exists('results/'+taskid+'/result.wav'):
    #     response = dict(taskid=taskid, progress=100., done=True)
    # elif os.path.exists('results/'+taskid+'/temp.wav'):
    #     response = dict(taskid=taskid, progress=50., done=False)
    # else:
    #     response = dict(taskid=taskid, progress=0., done=False)

    if IDM.isdone(taskid):
        response = dict(taskid=taskid, progress=100., done=True)
    else:
        Lock.acquire()
        index = IDM.search(taskid)
        if index == -1:
            response = dict(taskid=taskid, progress=0., done=False)
        else:
            process = IDM.wait_list[index]['process']
            response = dict(taskid=taskid, progress=process,
                            done=(process == 100.))
        Lock.release()

    return make_response(jsonify(response))


@app.route('/aivoiceresult')
def aivoiceresult():
    '''
    获取结果
    '''

    taskid = request.values.get('taskid')
    if not os.path.exists('results/'+taskid+'/result.wav'):
        return make_response(jsonify(dict(taskid=taskid, done=False)))

    return send_file('results/'+taskid+'/result.wav')


@app.route('/aivoicecancel')
def aivoicecancel():
    '''
    删除任务
    '''

    taskid = request.values.get('taskid')
    Lock.acquire()
    IDM.pop(taskid)
    Lock.release()
    # 此方法会强制删除所有文件
    if os.path.exists('results/'+taskid):
        shutil.rmtree('results/'+taskid)

    return make_response(jsonify(True))

# ---------------------------------------------------------------------------------------


@app.route('/uploadVideo', methods=['GET', 'POST'])
def uploadVideo():
    if request.method=="GET":
        res=make_response(render_template('index.html'))
        return res
    if 'input_video' in request.files:
        objFile = request.files.get('input_video')
    else:
        return ('Upload Error: No such file')

    audioid = request.values.get('audioid')

    # 生成唯一id
    taskid = IDM.generate_id('am')

    os.mkdir('results/'+taskid)
    objFile.save('results/'+taskid+'/input_video.mp4')
    with open('results/'+taskid+'/audioid.txt', 'w', encoding='utf-8') as f:
        f.write(audioid)
    return taskid


@app.route('/aimanstart')
def aimanstart():
    '''
      开始任务
      '''
    taskid = request.values.get('taskid')
    with open('results/'+taskid+'/audioid.txt', 'r', encoding='utf-8') as f:
        audioid = f.read()

    if IDM.search(taskid) == -1 and not IDM.isdone(taskid):
        Lock.acquire()
        IDM.add({'task': 'Wav2Lip', 'taskid': taskid,
                'audioid': audioid, 'process': 0})
        Lock.release()

    return make_response(jsonify(dict(taskid=taskid, done=True)))


@app.route('/aimanprocess')
def aimanprocess():
    '''
    查看进度
    '''
    taskid = request.values.get('taskid')
    # if os.path.exists('results/'+taskid+'/result.wav'):
    #     response = dict(taskid=taskid, progress=100., done=True)
    # elif os.path.exists('results/'+taskid+'/temp.wav'):
    #     response = dict(taskid=taskid, progress=50., done=False)
    # else:
    #     response = dict(taskid=taskid, progress=0., done=False)

    if IDM.isdone(taskid):
        response = dict(taskid=taskid, progress=100., done=True)
    else:
        index = IDM.search(taskid)
        if index == -1:
            response = dict(taskid=taskid, progress=0., done=False)
        else:
            process = IDM.wait_list[index]['process']
            response = dict(taskid=taskid, progress=process,
                            done=(process == 100.))

    return make_response(jsonify(response))


@app.route('/aimanresult')
def aimanresult():
    '''
    获取结果
    '''
    taskid = request.values.get('taskid')
    if not os.path.exists('results/' + taskid + '/output_video.mp4'):
        return make_response(jsonify(dict(taskid=taskid, done=False)))

    return send_file('results/' + taskid + '/output_video.mp4')


@app.route('/aimancancel')
def aimancancel():
    '''
    删除任务
    '''
    taskid = request.values.get('taskid')
    IDM.pop(taskid)
    # 此方法会强制删除所有文件
    if os.path.exists('results/' + taskid):
        shutil.rmtree('results/' + taskid)

    return make_response(jsonify(True))


@app.route('/mainpage')
def mainpage():
    return render_template('mainpage.html')


#附加内容：主页面，用于整合八个接口
@app.route('/',methods=("GET", "POST"))
def menupage():
    taskid = request.values.get('taskid')
    # values 对GET和POST通用
    text = request.values.get('text')
    voiceid = request.values.get('voiceid')
    if request.method == 'POST' and not text and not voiceid:
       # print(url_for('uploadtext',taskid=taskid))
        return render_template('main.html',taskid=taskid)

    '''
    上传文字转语音的文字内容
    '''
    #开始上传，渲染模板
    if request.method == "GET" and not text:
        return make_response(render_template('main.html',taskid=taskid))
    if not text:
        return make_response('Upload Error: No text upload')
    if not voiceid:
        voiceid = '1'

    # 生成唯一id
    taskid = IDM.generate_id('ts')
    print(text, voiceid)

    os.mkdir('results/'+taskid)
    with open('results/'+taskid+'/text.txt', 'w', encoding='utf-8') as f:
        f.write(text)
    with open('results/'+taskid+'/voiceid.txt', 'w', encoding='utf-8') as f:
        f.write(voiceid)

    #! 自动启动任务
    # IDM.add({'task': 'TTS', 'taskid': taskid, 'process': 0})

   # return taskid

    return render_template('main.html',taskid=taskid)




# ---------------------------------------------------------------------------------------


def check_task():
    '''
    后台持续运行, 管理任务分配
    '''
    while True:
        if IDM.wait_list:
            currentTask = IDM.wait_list[0]
            print(currentTask)
            if os.path.exists('results/'+currentTask['taskid']):
                if currentTask['task'] == 'TTS':
                    print('开始执行TTS任务, ', currentTask['taskid'])
                    tts_task(currentTask['taskid'])
                    Lock.acquire()

                elif currentTask['task'] == 'Wav2Lip':
                    print('开始执行Wav2Lip任务, ', currentTask['taskid'])
                    Wav2LipTask(currentTask['taskid'], currentTask['audioid'])
                    Lock.acquire()

            IDM.pop(currentTask['taskid'])
            IDM.result_list.append(currentTask['taskid'])
            IDM.save()
            Lock.release()

        else:
            time.sleep(1)


model = load_model(args.checkpoint_path)
print("Model loaded")

thread = threading.Thread(target=check_task)
thread.start()

app.run(host='127.0.0.1', port=8080, debug=True)

