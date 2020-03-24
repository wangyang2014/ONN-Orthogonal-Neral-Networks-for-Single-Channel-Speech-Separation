from mir_eval.separation import bss_eval_sources
from utilsm import * 
#Methon = 'CNMF'
#allFilepath,saveMixFilePath,resultPath = synthesis('D:\\paper code\\data\\TIMIT\\TRAIN\\DR1',Methon)
print("*********")
''''
for i in range(0,300): 
    savaPath = saveMixFilePath[i].replace('TRAIN','RESULT')+'test'
    mixSignalpath = savaPath + "\\" + 'mix.wav'
    mixed_wav = readSignal(mixSignalpath)

    src1_wav = readSignal(allFilepath[i][0].replace('TRAIN','TEST'))
    src2_wav = readSignal(allFilepath[i][1].replace('TRAIN','TEST'))

    pred_src1_wav = readSignal(resultPath[i][0].replace('TRAIN','RESULT'))
    pred_src2_wav = readSignal(resultPath[i][1].replace('TRAIN','RESULT'))

    length = min([src1_wav.shape[0],src2_wav.shape[0],pred_src1_wav.shape[0],pred_src2_wav.shape[0]])
    
    sdr, sir, sar, _ = bss_eval_sources(np.array([src1_wav, src2_wav]),
                                            np.array([pred_src1_wav, pred_src2_wav]), False)
    
    sdr_mixed, _, _, _ = bss_eval_sources(np.array([src1_wav, src1_wav]),
                                              np.array([mixed_wav, mixed_wav]), False)
    result = "sdr {}, sir {}, sar {}, nsdr {}".format(sdr,sir,sar,sdr - sdr_mixed)
    print(result)'''
mixed_wav = readSignal(r'D:\paper code\data\mix_test.wav')
src1_wav = readSignal(r'D:\paper code\data\Track3_test.wav')
src2_wav = readSignal(r'D:\paper code\data\Track2_test.wav')

pred_src1_wav = readSignal(r'C:\Users\WYang\Desktop\ONN\1.wav')
pred_src2_wav = readSignal(r'C:\Users\WYang\Desktop\ONN\2.wav')
length = min([src1_wav.shape[0],src2_wav.shape[0],pred_src1_wav.shape[0],pred_src2_wav.shape[0]])

length = int(length/100)
src1_wav = src1_wav[0:length]
src2_wav = src2_wav[0:length]
mixed_wav = mixed_wav[0:length]
pred_src1_wav = pred_src1_wav[0:length]
pred_src2_wav = pred_src2_wav[0:length]

sdr, sir, sar, _ = bss_eval_sources(np.array([src1_wav, src2_wav]),
                                    np.array([pred_src1_wav, pred_src2_wav]), False)
                        
sdr_mixed, _, _, _ = bss_eval_sources(np.array([src1_wav, src2_wav]),
                                              np.array([mixed_wav, mixed_wav]), False)

result = "sdr {}, sir {}, sar {}, nsdr {}".format(sdr,sir,sar,sdr - sdr_mixed)

print(result)
print(sdr_mixed)

def cal_SDRi(src_ref, src_est, mix):
    """Calculate Source-to-Distortion Ratio improvement (SDRi).
    NOTE: bss_eval_sources is very very slow.
    Args:
        src_ref: numpy.ndarray, [C, T]
        src_est: numpy.ndarray, [C, T], reordered by best PIT permutation
        mix: numpy.ndarray, [T]
    Returns:
        average_SDRi
    """
    ''''mask = np.ones_like(mix)
    mask[np.where((np.abs(src_ref[0]) - np.abs(src_ref[1])) > 0) ] = 0
    src_est[0] = (1- mask ) * mix
    src_est[1] = mask * mix'''

    src_anchor = np.stack([mix, mix], axis=0)
    sdr, sir, sar, popt = bss_eval_sources(src_ref, src_est)
    sdr0, sir0, sar0, popt0 = bss_eval_sources(src_ref, src_anchor)
    avg_SDRi = ((sdr[0]-sdr0[0]) + (sdr[1]-sdr0[1])) / 2
    # print("SDRi1: {0:.2f}, SDRi2: {1:.2f}".format(sdr[0]-sdr0[0], sdr[1]-sdr0[1]))
    return avg_SDRi

def cal_SISNRi(src_ref, src_est, mix):
    """Calculate Scale-Invariant Source-to-Noise Ratio improvement (SI-SNRi)
    Args:
        src_ref: numpy.ndarray, [C, T]
        src_est: numpy.ndarray, [C, T], reordered by best PIT permutation
        mix: numpy.ndarray, [T]
    Returns:
        average_SISNRi
    """
    #mask = np.ones_like(mix)
    #mask[np.where((np.abs(src_ref[0]) - np.abs(src_ref[1])) > 0) ] = 0
   
    #src_est[0] = (1- mask ) * mix
    #src_est[1] = mask * mix

    sisnr1 = cal_SISNR(src_ref[0], src_est[0])
    sisnr2 = cal_SISNR(src_ref[1], src_est[1])
    sisnr1b = cal_SISNR(src_ref[0], mix)
    sisnr2b = cal_SISNR(src_ref[1], mix)
    # print("SISNR base1 {0:.2f} SISNR base2 {1:.2f}, avg {2:.2f}".format(
    #     sisnr1b, sisnr2b, (sisnr1b+sisnr2b)/2))
    # print("SISNRi1: {0:.2f}, SISNRi2: {1:.2f}".format(sisnr1, sisnr2))
    avg_SISNRi = ((sisnr1 - sisnr1b) + (sisnr2 - sisnr2b)) / 2
    return avg_SISNRi


def cal_SISNR(ref_sig, out_sig, eps=1e-8):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        ref_sig: numpy.ndarray, [T]
        out_sig: numpy.ndarray, [T]
    Returns:
        SISNR
    """
    assert len(ref_sig) == len(out_sig)
    ref_sig = ref_sig - np.mean(ref_sig)
    out_sig = out_sig - np.mean(out_sig)
    ref_energy = np.sum(ref_sig ** 2) + eps
    proj = np.sum(ref_sig * out_sig) * ref_sig / ref_energy
    noise = out_sig - proj
    ratio = np.sum(proj ** 2) / (np.sum(noise ** 2) + eps)
    sisnr = 10 * np.log(ratio + eps) / np.log(10.0)
    return sisnr


print(cal_SISNRi(np.array([src1_wav, src2_wav]),np.array([pred_src1_wav, pred_src2_wav]),mixed_wav))
print(cal_SDRi(np.array([src1_wav, src2_wav]),np.array([pred_src1_wav, pred_src2_wav]),mixed_wav))

