import wave
import librosa
import numpy as np
import pyaudio
import argparse
import functools

from macls.predict import MAClsPredictor
from macls.utils.record import RecordAudio
from macls.utils.utils import add_arguments, print_arguments

# 录音参数
CHUNK = 1024
FORMAT = pyaudio.paInt24
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 3
WAVE_OUTPUT_FILENAME = "./infer_audio.wav"

# 模型参数
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,    'configs/ecapa_tdnn.yml',   '配置文件')
add_arg('use_gpu',          bool,   True,                       '是否使用GPU预测')
add_arg('model_path',       str,    'models/EcapaTdnn_MelSpectrogram_2/best_model/', '导出的预测模型文件路径')
args = parser.parse_args()
print_arguments(args=args)

# 打开录音
p = pyaudio.PyAudio()
info = p.get_host_api_info_by_index(0)
devices = info.get('deviceCount')
for i in range(0, devices):
    if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
        print("Input Device id", i, "-", p.get_device_info_by_host_api_device_index(0, i))
way = input("Select a mic:\n")
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                input_device_index=int(way)
                )


# 读取音频数据
def load_data(data_path):
    wav, sr = librosa.load(data_path, sr=44100)
    intervals = librosa.effects.split(wav, top_db=20)
    wav_output = []
    for sliced in intervals:
        wav_output.extend(wav[sliced[0]:sliced[1]])
    if len(wav_output) < 22050:
        raise Exception("有效音频小于0.5s")
    wav_output = np.array(wav_output)
    ps = librosa.feature.melspectrogram(y=wav_output, sr=sr, hop_length=256).astype(np.float32)
    ps = ps[np.newaxis, ..., np.newaxis]
    return ps


# 获取录音数据
def record_audio():
    print("开始录音......")
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)

    frames = []
    for i in range(0, int(RATE * RECORD_SECONDS / CHUNK)):
        data = stream.read(CHUNK)
        wf.append(data)

    print("录音已结束!")
    # wf.writeframes(b''.join(frames))
    wf.close()
    return WAVE_OUTPUT_FILENAME


if __name__ == '__main__':
    # 获取识别器
    predictor = MAClsPredictor(configs=args.configs,
                               model_path=args.model_path,
                               use_gpu=args.use_gpu)
    try:
        while True:
            # 加载数据
            audio_data = record_audio()
            label, s = predictor.predict(audio_data, sample_rate=RATE)
            print(f'预测的标签为：{label}，得分：{s}')

    except Exception as e:
        print(e)
        stream.stop_stream()
        stream.close()
        p.terminate()
