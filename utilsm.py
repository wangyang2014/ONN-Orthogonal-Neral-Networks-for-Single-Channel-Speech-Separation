# -*- coding: utf-8 -*
import os
import librosa
import scipy.io as sio
from matplotlib import pyplot
from pystoi.stoi import stoi
import numpy as np
import sys
from confing import SAMPLE_RATE, FFT_LEN, CONSTANT, SPEECH_RANK
import pickle

def alignment(signalA,signalB):
    L1 = signalA.shape[0]
    L2 = signalB.shape[0]
    L = min(L1,L2)
     
    return signalA[0:L] + signalB[0:L],L
    
def synthesis(path,methon):
    allFilepath = []
    saveMixFilePath = []
    resultPath = []
    filelist = os.listdir(path)
    for i in range(0,len(filelist)):
        for j in range(i+1,len(filelist)):
            ii = filelist[i]
            jj = filelist[j]
            signal_Address_one = path + "\\" + ii + "\\" + ii + ".wav"
            signal_Address_tow = path + "\\" + jj + "\\" + jj + ".wav"
            filepath = [signal_Address_one,signal_Address_tow]
            allFilepath.append(filepath)
            savePath = path + "\\" + ii + jj
            saveMixFilePath.append(savePath)
            result_Address_one = savePath + "\\" + methon + ii +str(SPEECH_RANK)+ ".wav"
            result_Address_tow = savePath + "\\" + methon + jj +str(SPEECH_RANK)+ ".wav"
            resultPath.append([result_Address_one,result_Address_tow])
            if not os.path.exists(savePath.replace('TRAIN','RESULT')):
                os.makedirs(savePath.replace('TRAIN','RESULT'))

    return allFilepath,saveMixFilePath,resultPath

def Vad(signal,size,threshold):
    Template = []
    for i in range(0,size):
        Template.append(signal[i])    
    for i in range(size,signal.shape[0]- size):
        data = signal[i-size:i+size]
        value = np.sum(np.abs(data))
        if value >= threshold:
            Template.append(signal[i])
    
    for i in range(signal.shape[0]- size,signal.shape[0]):
        Template.append(signal[i])

    return np.asarray(Template)

def readSignal(fliepath,sr = SAMPLE_RATE):
    signal, _ = librosa.load(fliepath,sr=sr,mono=True)
    return signal

def wirteSignal(signal,filename):
    write_wav(signal,filename,sr=SAMPLE_RATE)

def dumpValue(vlaue,filename):
     pickle.dump(vlaue,open(filename, 'wb'))

def loadValue(filename):
    return  pickle.load(open(filename,'rb'))

#将MP3文件转为wav文件
def mp2Wav(inFlieName,outFileName):
    command = 'ffmpeg  -i %s -f wav %s' %(inFlieName,outFileName)
    os.system(command)

def decode(inFlieName,outFileName,sr=SAMPLE_RATE):
    signal, _ = librosa.load(inFlieName,sr=sr,mono=True)
    write_wav(signal,outFileName,sr=sr)

#语音波形图形
def plot_spec(signal, sr=SAMPLE_RATE, title=''):
    """
    to polt a spec quickly, wrap it

    Example:
    >>> sig, _ = librosa.load('speech/test/fjcs0_sx409.wav', sr=SAMPLE_RATE)
    >>> plot_spec(sig)

    >>> plot_spec(sig, SAMPLE_RATE)

    >>> plot_spec(sig, SAMPLE_RATE, 'test_title')

    :param signal:
    :param sr:
    :param title:
    :return:
    """
    pyplot.figure()
    pyplot.title(title)
    pyplot.specgram(signal, Fs=sr, scale_by_freq=True, sides='default')
    pyplot.show()
    
#用降噪后的幅度谱和带噪语音的文件，生成输出语音
def spec2sig_from_file(spec, mix_file, sr=SAMPLE_RATE):
    """
    use phase of mix spec to restruct the signal

    Example:
    >>> sig, _ = librosa.load('speech/test/fjcs0_sx319.wav', sr=SAMPLE_RATE)
    >>> spec = get_spec(sig)
    >>> new_sig = spec2sig(spec, 'mix/fjcs0_sx319.wav_0.wav')
    >>> print(len(sig)==len(new_sig))
    True

    :param spec:
    :param mix_file:
    :param sr:
    :return:
    """
    file_sig, _ = librosa.load(mix_file, sr=SAMPLE_RATE)
    N = FFT_LEN
    mix = librosa.stft(file_sig, n_fft=N, hop_length=N // 2)
    mask = spec / np.abs(mix)
    # 这里使用了维纳滤波的合成公式
    signal_spec = np.array(mix) * np.array(mask)
    sig = librosa.istft(signal_spec, hop_length=N // 2)
    return sig


def spec2sig_from_sig(spec, mix_sig=None, sr=SAMPLE_RATE, mix=None):
    N = FFT_LEN
    if mix is None:
        mix = librosa.stft(mix_sig, n_fft=N, hop_length=N // 2)
    mask = spec / np.abs(mix)
    # 这里使用了维纳滤波的合成公式
    signal_spec = np.array(mix) * np.array(mask)
    sig = librosa.istft(signal_spec, hop_length=N // 2)
    return sig


def write_wav(signal, output, sr=SAMPLE_RATE):
    """
    wrapped librosa write_wav to use the same SAMPLE_RATE
    resample may cause diff between ori_sig and sig in wav

    Example:
    >>> write_wav(np.ndarray([1, 2]), 'doc_test/test.wav')

    >>> print(os.path.exists('doc_test'))
    True

    >>> print(os.path.exists('doc_test/test.wav'))
    True

    >>> write_wav([1,2,3], 'doc_test/test_err.wav')
    Traceback (most recent call last):
    ...
    librosa.util.exceptions.ParameterError: data must be of type numpy.ndarray

    >>> sig, fs = librosa.load('speech/test/fjcs0_sx319.wav', sr=None)
    >>> write_wav(sig, 'doc_test/same_test.wav', sr=fs)
    >>> new_sig, _ = librosa.load('doc_test/same_test.wav', sr=fs)

    :param signal:
    :param output:
    :return:
    """
    librosa.output.write_wav(output, signal, sr, norm=True)

#计算pesq
def cal_pesq(enhance, clean, sr=SAMPLE_RATE):
    """
    Use bin file to calculate PESQ of enhance speech

    Example:
    >>> cal_pseq('test', 'test')
    Traceback (most recent call last):
    ...
    Exception: you use the PESQ program with wrong command

    >>> cal_pseq('mix/fjcs0_sx319.wav_-1.wav', 'speech/test/fjcs0_sx319.wav')
    1.12

    :param enhance:
    :param clean:
    :return:
    """
    lines = os.popen("./tools/pesq +%s %s %s" % (sr, clean, enhance)).readlines()
    try:
        res = float(lines[-1].split('=')[-1])
    except ValueError:
        raise Exception('you use the PESQ program with wrong command')
    return res

#计算stoi
def cal_stoi_from_file(enhance, clean, sr=SAMPLE_RATE):
    """
    A wrapped pystoi func to get stoi

    Example:
    >>> cal_stoi('test', 'test')
    Traceback (most recent call last):
    ...
    Exception: can not find files, please check path

    >>> cal_stoi('mix/fjcs0_sx319.wav_-1.wav', 'speech/test/fjcs0_sx319.wav')
    0.76689639728190695

    :param enhance:
    :param clean:
    :return:
    """
    try:
        clean_signal, _ = librosa.load(clean, sr=sr)
        # plot_wav(clean_signal, title=clean)
        enhance_signal, _ = librosa.load(enhance, sr=sr)
        size = min(enhance_signal.shape[0],clean_signal.shape[0])
        #clean_signal = clean_signal[:len(enhance_signal)] # 有时信号不等长，所以这里对齐
    except FileNotFoundError:
        raise Exception('can not find files, please check path')
    except IsADirectoryError:
        raise Exception('can not find files, please check path')

    res = stoi(clean_signal[0:size], enhance_signal[0:size], SAMPLE_RATE, extended=False)
    return res

def cal_stoi(enhance, clean, sr=SAMPLE_RATE):
    clean_signal = clean[:len(enhance)] # 有时信号不等长，所以这里对齐
    return stoi(clean_signal, enhance, SAMPLE_RATE, extended=False)


def plot_box(arr, title='', range=[-.5, 4.5]):
    """
    plot a box

    Example:
    >>> plot_box([1,2,3,4,5,6,7], title='Title')
    >>> pyplot.close()

    >>> plot_box(123)
    Traceback (most recent call last):
    ...
    Exception: please use a data array to plot, but get <class 'int'>

    :param arr:
    :param title:
    :return:
    """
    if type(arr) not in [list, np.array]:
        raise Exception('please use a data array to plot, but get %s'%type(arr))
    pyplot.ylim(range)
    pyplot.boxplot(arr, labels=[title])
    pyplot.show()


def plot_wav(signal, sr=SAMPLE_RATE, title=''):
    """
    plot waves of a wav file

    Example:
    >>> sig, sr = librosa.load('speech/test/fjcs0_sx319.wav')
    >>> plot_wav(sig)
    >>> pyplot.close()

    >>> plot_wav(sig, sr=sr)
    >>> pyplot.close()

    >>> plot_wav(sig, sr=sr, title='Title')
    >>> pyplot.close()

    >>> plot_wav(123)
    Traceback (most recent call last):
    ...
    Exception: please use a signal array to plot, but get <class 'int'>

    :param signal:
    :param sr:
    :param title:
    :return:
    """
    if type(signal) not in [list, np.array, np.ndarray]:
        raise Exception('please use a signal array to plot, but get %s'%type(signal))
    n_frames = len(signal)
    time = np.arange(0, n_frames) * (1.0 / sr)
    pyplot.figure()
    pyplot.title(title)
    pyplot.plot(time, signal, c="g")
    pyplot.xlabel("time (seconds)")
    pyplot.show()

def get_stats(data):
    """

    :param data:
    :return:
    """
    mean = np.mean(data)
    std = np.std(data)
    return mean, std


def get_spec(sig):
    N = FFT_LEN
    return librosa.stft(sig, n_fft=N, hop_length=N // 2)

def norm_signal(sig):
    """
    避免音量对指标和处理流程的影响，这里对整体的能量进行一个归一化

    :param sig:待处理信号
    :return :音量归一化后的信号
    """
    rate = np.sqrt(CONSTANT * len(sig) / np.sum(sig ** 2))
    return sig * rate


def mix_noise(signal, noise_wav, snr, random=False):
    """
    对信号与噪声按指定的SNR进行混合

    Example:

    >>> import librosa
    >>> sig, _ = librosa.load('speech/test/fjcs0_sx319.wav', sr=SAMPLE_RATE)
    >>> noise, _ = librosa.load('noise/f16.wav', sr=SAMPLE_RATE)
    >>> mix = mix_noise(sig, noise, 0)

    >>> print(len(mix)==len(sig))
    True

    :param signal:纯净信号
    :param noise_wav:噪声信号
    :param snr:信噪比
    :return:混合信号
    
    # 对齐噪声
    num_padd = len(signal) // len(noise_wav) + 1
    noise = np.repeat(noise_wav, num_padd)
    cut_end = len(noise) - len(signal)
    # 随机切割点
    cut = 0
    if cut_end > 0:
        cut = np.random.randint(0, cut_end+1)
    noise = np.array(noise[cut:cut+len(signal)])
    # 计算噪声调整比例
    e_noise = np.mean(np.abs(noise))
    e_signal = np.mean(np.abs(signal))
    x = (e_signal / e_noise) / np.sqrt(np.power(10, snr/10))
    return norm_signal(x * noise + signal)"""




if __name__ == '__main__':
    import doctest
    import shutil
    if os.path.exists('doc_test'):
        shutil.rmtree('doc_test')
    doctest.testmod(verbose=False)
    shutil.rmtree('doc_test')
