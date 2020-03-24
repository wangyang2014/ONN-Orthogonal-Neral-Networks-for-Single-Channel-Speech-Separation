import numpy as np 
from utilsm import *
def alignment(filepath1,filepath2,newName):
    signal_1 = readSignal(filepath1)
    signal_2 = readSignal(filepath2)

    size = min(signal_1.shape[0],signal_2.shape[0])
    wirteSignal(signal_1[0:size],newName)
    wirteSignal(signal_2[0:size],filepath2)

if __name__ == "__main__":
	alignment('man1.wav','NBNMFM1best.wav','man1best.wav')
	alignment('man2.wav','NBNMFM2best.wav','man2best.wav')
	alignment('woman1.wav','NBNMFW1best.wav','woman1best.wav')
	alignment('woman2.wav','NBNMFW2best.wav','woman2best.wav')