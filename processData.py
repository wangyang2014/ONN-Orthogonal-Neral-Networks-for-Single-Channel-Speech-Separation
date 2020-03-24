import os
import random
import sys

import time
import librosa
import numpy as np
from scipy.linalg import solve

from sklearn.preprocessing import normalize

from confing import CONSTANT, FFT_LEN, SAMPLE_RATE, SPEECH_RANK

from utilsm import get_spec,readSignal,alignment

class Information_Struct():
    def __init__(self,train_Filelist=None,test_Filelist=None):
        self.__train_Filelist = train_Filelist
        self.__test_Filelist = test_Filelist
        

    def __gettrainSignal(self):
        self.__signal_S1_Train = readSignal(self.__train_Filelist[0])
        self.__signal_S2_Train = readSignal(self.__train_Filelist[1])
        self.__signal_Mix_Train = readSignal(self.__train_Filelist[2])
    
    def __getTestSignal(self):
        self.__signal_S1_Test = readSignal(self.__test_Filelist[0])
        self.__signal_S2_Test = readSignal(self.__test_Filelist[1])
        self.__signal_Mix_Test,_ = alignment(self.__signal_S1_Test,self.__signal_S2_Test)

    def __getTrainSpectrum(self):
        self.__spectrum_S1_Train = get_spec(self.__signal_S1_Train)
        self.__spectrum_S2_Train = get_spec(self.__signal_S2_Train)
        self.__spectrum_Mix_Train = get_spec(self.__signal_Mix_Train)

        train_Data = np.column_stack((self.__spectrum_S1_Train,self.__spectrum_S2_Train))
        train_Data = np.column_stack((train_Data,self.__spectrum_Mix_Train))

        return np.abs(train_Data)

    
    def __getTestSpectrum(self):
        self.__spectrum_Mix_Test = get_spec(self.__signal_Mix_Test)

    def __gettarget(self):
        _,s1_len = self.__spectrum_S1_Train.shape
        _,s2_len = self.__spectrum_S2_Train.shape
        _,mix_len = self.__spectrum_Mix_Train.shape

        target = np.ones([s1_len+s2_len+mix_len,2])
        target[0:s1_len,1] = 0
        target[s1_len:s2_len+s1_len,0] = 0

        return target

    def getTrainData(self):
        self.__gettrainSignal()
        trianData = self.__getTrainSpectrum()

        target = self.__gettarget()
        #self.__getTrainSpectrum()
        
        return target,trianData

    def getTestData(self):
        #self.__getTestSignal()
        #self.__getTestSpectrum()
        self.__spectrum_Mix_Test = get_spec(readSignal(self.__test_Filelist[0]))
        testData = np.abs(self.__spectrum_Mix_Test)
        
        return testData,self.__spectrum_Mix_Test
if __name__ == "__main__":
    trainData = Information_Struct(train_Filelist=[r'D:\paper code\data\Track3_train.wav',r'D:\paper code\data\Track2_train.wav',r'D:\paper code\data\mix_train.wav'])
    testData = Information_Struct(test_Filelist=[r'D:\paper code\data\mix_test.wav'])
    trainData.getTrainData()
    testData.getTestData()