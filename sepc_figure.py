"""Python绘制语谱图"""

# 导入相应的包
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import wave
import os
import librosa

filepath = r'C:\Users\WYang\Desktop\paper about speech spearate\ONN\data'  # 添加路径

files = os.listdir(filepath)
for i in files:
    f = librosa.read(files+'//'+ i,)  # 调用wave模块中的open函数，打开语音文件。
    params = f.getparams()  # 得到语音参数
    nchannels, sampwidth, framerate, nframes = params[:4]  # nchannels:音频通道数，sampwidth:每个音频样本的字节数，framerate:采样率，nframes:音频采样点数
    strData = f.readframes(nframes)  # 读取音频，字符串格式
    wavaData = np.fromstring(strData, dtype=np.int16)  # 得到的数据是字符串，将字符串转为int型
    wavaData = wavaData * 1.0/max(abs(wavaData))  # wave幅值归一化
    wavaData = np.reshape(wavaData, [nframes, nchannels]).T  # .T 表示转置
    f.close()

    # 绘制语谱图
    spec, freqs, t = mlab.specgram(x=wavaData[0], Fs=framerate, scale_by_freq=True, mode='psd', sides='default', NFFT=320)
    spec = 10. * np.log10(spec)
    spec = np.flipud(spec)
    plt.imsave(i.replace('.wav','.jpg'), spec)