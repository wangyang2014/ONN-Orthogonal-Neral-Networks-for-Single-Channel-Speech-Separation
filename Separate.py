import os
import numpy as np
import librosa
import sys
from confing import SAMPLE_RATE, FFT_LEN, CONSTANT,SPEECH_RANK
from spesyn import SpeechSynthesis ,Across 
from decompose import decompose, decompose_with_dict
from cs_utils import CommonSpace,is_in_dict
from utilsm import *
from getVectorAndPoint import  MixVectorAndIndividVector,CalCommonSpacePoint, IndividualitySpace
from spesyn import Across
import sys
import sys,os

class Separate():
    def __init__(self,signalMix,dictMixVector,mixVector,individVector,sigIdividLen,sigMixLen):
        self.__signalMix = signalMix
        self.__dictMixVector = dictMixVector
        self.__mixVector = mixVector
        self.__individVector = individVector
        self.__sigIdividLen = sigIdividLen
        self.__sigMixLen = sigMixLen
        self.__signalMixSTFT = get_spec(self.__signalMix) 
        self.__commonSpacePoint = None
        self.__label = None
    
    def __getWight(self):
        if os.access('Mixwight.txt',os.F_OK):
            wight = loadValue('Mixwight.txt')
        else:
            wight = decompose_with_dict(np.abs(self.__signalMixSTFT),self.__dictMixVector)
            dumpValue(wight,'Mixwight.txt')
        return wight

    def __getRata(self):
        if os.access('CommonRata.txt',os.F_OK):
            rate = loadValue('CommonRata.txt')
        else:
            value = None
            m = self.__mixVector.shape[1] 
            k = self.__dictMixVector.shape[1]
            for i in range(0,m):
                V = np.transpose(self.__mixVector[:,i]).reshape(-1, 1)
                dic = self.__dictMixVector[:,k-m-2:k]
                wigth = decompose_with_dict(V,dic)
                if value is None:
                    value = wigth
                else:
                    value = np.column_stack([value,wigth])
            COUNT = 10**(-10)    
            rate = (np.sum(value[:,0:self.__sigMixLen],axis=1) + COUNT)/(np.sum(value,axis=1)+2*COUNT)
            dumpValue(rate,'CommonRata.txt')
        return rate

    def getLabel(self,sigAdict,sigBdict):
        _,m = self.__signalMixSTFT.shape
        if os.access('Label.txt',os.F_OK):
            self.__label = loadValue('Label.txt')
        else:
            self.__label = np.zeros([m])
            '''for i in range(0,m):
                if is_in_dict(np.abs(self.__signalMixSTFT[:,i]),sigAdict):
                    self.__label[i] = 1
                elif is_in_dict(np.abs(self.__signalMixSTFT[:,i]),sigBdict):
                    self.__label[i] = 2
            #self.__commonSpacePoint = None'''
            for i in range(0,m):
                if is_in_dict(np.abs(self.__signalMixSTFT[:,i]),self.__mixVector):
                    if self.__commonSpacePoint is None:
                        self.__commonSpacePoint = self.__signalMixSTFT[:,i]
                    else:
                        self.__commonSpacePoint = np.column_stack([self.__commonSpacePoint,self.__signalMixSTFT[:,i]])
                else:
                    if is_in_dict(np.abs(self.__signalMixSTFT[:,i]),sigAdict):
                        self.__label[i] = 1
                    elif is_in_dict(np.abs(self.__signalMixSTFT[:,i]),sigBdict):
                        self.__label[i] = 2
                    else:
                        if self.__commonSpacePoint is None:
                            self.__commonSpacePoint = self.__signalMixSTFT[:,i]
                        else:
                            self.__commonSpacePoint = np.column_stack([self.__commonSpacePoint,self.__signalMixSTFT[:,i]])
                        self.__label[i] = 0
                if i%100 == 0 :
                    print(m,i)


            dumpValue(self.__label,'Label.txt')
                    


    def process(self):
        wight = self.__getWight()
        m = self.__individVector.shape[1] 
        k = self.__dictMixVector.shape[1]

        rata = self.__getRata()

        partA = np.dot(self.__individVector[:,0:self.__sigIdividLen],wight[0:self.__sigIdividLen,:]) + \
            np.dot(self.__dictMixVector[:,m:k] * rata,wight[m:k,:])#np.multiply(np.dot(self.__dictMixVector[:,m:k],wight[m:k,:]),rata)
        
        #commomSignal = np.dot(self.__dictMixVector[:,m:k] ,wight[m:k,:])

        partB = np.dot(self.__individVector[:,self.__sigIdividLen:m],wight[self.__sigIdividLen:m,:]) + \
            np.dot(self.__dictMixVector[:,m:k] * (1-rata),wight[m:k,:])

        m = self.__label.shape[0]

        signalA = np.zeros(partA.shape) 
        signalB = np.zeros(partA.shape)
        TpsigA =  np.zeros(partA.shape)
        TpsigB =  np.zeros(partA.shape)
        TpCom = np.zeros(partA.shape)
        
        for i in range(0,m): 
            if self.__label[i] == 0:
                TpCom[:,i] = 1
            elif self.__label[i] == 1:
                TpsigA[:,1]  = 1
            else:
                TpsigB[:,1]  = 1
            
            if  i % 100 == 0:
                print(i)
        
        signalA = self.__signalMixSTFT * TpsigA + partA * TpCom
        signalB = self.__signalMixSTFT * TpsigB + partB * TpCom

        return signalA,signalB

def Initialization():
    #print('please input signalPath A and B')
    #flielist = input().split(' ')
    flielist = [r'dataspeech\track1.wav',r'dataspeech\track10.wav',r'dataspeech\across.wav']
    signalA = readSignal(flielist[0])
    signalB = readSignal(flielist[1])
    mixSignal = readSignal(flielist[2])

    #mixSignal = Across(signalA=signalA,signalB=signalB).process()
    #N = mixSignal.shape[0]
    #wirteSignal(mixSignal,flielist[2])
    #wirteSignal(signalA[0:N],flielist[0])
    #wirteSignal(signalB[0:N],flielist[1])
    sigIdividLen,sigMixLen,mixVector,IndividVector,W_a,W_b = MixVectorAndIndividVector(signalA,signalB).process()

    #CommonVector,_ ,sigAdict,sigBdict= CalCommonSpacePoint(signalA,signalB).getgetCommonSpace(k=mixVector.shape[1] + 2)

    return sigIdividLen,sigMixLen,mixSignal,mixVector,IndividVector,CommonVector,sigAdict,sigBdict
 
def calEandRata(partA,partB,commomSignal):
    n = partA.shape[0]
    #for i in range(0,n):
    sigPartAE =  np.sum(partA ** 2)
    sigPartBE =  np.sum(partB ** 2)
    sigcomE  =  np.sum(commomSignal ** 2)
    SUME = sigPartAE + sigPartBE + sigcomE

    return sigPartAE/SUME,sigPartBE/SUME,sigcomE/SUME

def VadProcess():
    inputflie = [r'dataspeech\track1.wav',r'dataspeech\track10.wav',r'dataspeech\across.wav']
    outputflie = [r'dataspeech\track1Vad.wav',r'dataspeech\track10Vad.wav',r'dataspeech\acrossVad.wav']
    for i in range(0,len(outputflie)):
        signal = readSignal(inputflie[i])
        Template = Vad(signal,64,0.25)
        wirteSignal(Template,outputflie[i])
        print(signal.shape[0],Template.shape[0])


def preTreatment():
    mp3list = [r'dataspeech\track1.mp3',r'dataspeech\track10.mp3',r'dataspeech\across.mp3']
    wavlist = [r'dataspeech\track1.wav',r'dataspeech\track10.wav',r'dataspeech\across.wav']
    for i in range(0,len(mp3list)):
        mp2Wav(mp3list[i],wavlist[i])

    VadProcess()

if __name__ == "__main__":
    #mp2Wav(r'data\track1.mp3',r'data\track1.wav')
    #preTreatment()
    if os.access('valueinformation.txt',os.F_OK):
        sigIdividLen,sigMixLen,mixSignal,mixVector,IndividVector,CommonVector,sigAdict,sigBdict = loadValue('valueinformation.txt')    
    else:
        sigIdividLen,sigMixLen,mixSignal,mixVector,IndividVector,CommonVector,sigAdict,sigBdict = Initialization()
        dumpValue([sigIdividLen,sigMixLen,mixSignal,mixVector,IndividVector,CommonVector,sigAdict,sigBdict],'valueinformation.txt')
    
    #mixSignal = readSignal(r'data\track1.mp3')
    
    dictMixVector = np.column_stack([IndividVector,CommonVector])
    process = Separate(mixSignal,dictMixVector,mixVector,IndividVector,sigIdividLen,sigMixLen)
    process.getLabel(sigAdict,sigBdict)
    partA,partB = process.process()
    write_wav(spec2sig_from_sig(partA,mixSignal),'track1New.wav')
    write_wav(spec2sig_from_sig(partB,mixSignal),'track10New.wav')
    #getwav()
    #AR,BR,CR = calEandRata(partA,partB,commomSignal)
