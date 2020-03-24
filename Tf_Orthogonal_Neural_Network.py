import tensorflow.compat.v1 as tf
import os 
import  numpy as np
import pickle
from confing import SAMPLE_RATE, FFT_LEN, CONSTANT, SPEECH_RANK
from sklearn import preprocessing 
from processData import Information_Struct
import librosa
from utilsm import get_spec,readSignal,alignment
tf.disable_v2_behavior()
training_iters = 5000000
learning_rata = 0.0005

n_input = 64
n_otput = n_input
batch_size = 20
code_size = 60
#keep_prob = tf.placeholder(tf.float32) dropout

indata = tf.placeholder(tf.float32,[None,n_input])
otdata = tf.placeholder(tf.float32,[None,n_otput])
target = tf.placeholder(tf.float32,[None,2])

weights = {
    'wd1': tf.Variable(tf.random_normal([n_input, n_input]))*0.01,
    'wd2': tf.Variable(tf.random_normal([n_input, 2*n_input]))*0.1,
    'wd_1_ST': tf.Variable(tf.random_normal([2*n_input])),
    #'wd_2_ST': tf.Variable(tf.random_normal([2*n_input]))*0.01,
    'codelayer_1': tf.Variable(tf.random_normal([2*n_input, 2*n_input]))*0.1,
    'codelayer_2': tf.Variable(tf.random_normal([2*n_input, 2*n_input]))*0.1,
    'wd3': tf.Variable(tf.random_normal([2*n_input, n_input]))*0.1,
    'out': tf.Variable(tf.random_normal([n_input, n_input]))*0.1,
}


biases = {
    'bc1': tf.Variable(tf.random_normal([n_input])),
    'bc2': tf.Variable(tf.random_normal([2*n_input])),
    'codelayer_b1': tf.Variable(tf.random_normal([2*n_input])),
    'codelayer_b2': tf.Variable(tf.random_normal([2*n_input])),
    #'bc3': tf.Variable(tf.random_normal([2*n_input])),
    #'out': tf.Variable(tf.random_normal([n_input])),
}

def fullConnect(inputData,weight,biases,actiation=None):
    if actiation is None:
        return tf.matmul(inputData,weight) + biases
    return actiation(tf.matmul(inputData,weight) + biases)

def ONNModel(indata):
    #encode layer
    layer_1_out = fullConnect(indata,weights['wd1'],biases['bc1'],tf.nn.relu)
    layer_2_out = fullConnect(layer_1_out,weights['wd2'],biases['bc2'],tf.nn.sigmoid)

    #code 
    oNNSpeaker_1_Transformation = tf.multiply(layer_2_out,weights['wd_1_ST'])
    wd_2_ST = 1 - weights['wd_1_ST']
    oNNSpeaker_2_Transformation = tf.multiply(layer_2_out,wd_2_ST)
    
    #Orthogonal
    speaker_1_Code = fullConnect(oNNSpeaker_1_Transformation,weights['codelayer_1'],biases['codelayer_b1'],tf.nn.relu)
    speaker_2_Code = fullConnect(oNNSpeaker_2_Transformation,weights['codelayer_2'],biases['codelayer_b2'],tf.nn.relu)
    
    out_code_1 = tf.transpose(tf.multiply(tf.transpose(speaker_1_Code), target[:,0])) 
    out_code_2 = tf.transpose(tf.multiply(tf.transpose(speaker_2_Code), target[:,1]))
    #out_code = out_code_1 + out_code_2 
    #decode layer
    layer_3_out_1 = fullConnect(out_code_1,weights['wd3'],0,tf.nn.relu)
    layer_4_out_1 = fullConnect(layer_3_out_1,weights['out'],0,tf.nn.sigmoid) - 0.5
    layer_4_out_1 = layer_4_out_1  * 100

    layer_3_out_2 = fullConnect(out_code_2,weights['wd3'],0,tf.nn.relu)
    layer_4_out_2 = fullConnect(layer_3_out_2,weights['out'],0,tf.nn.sigmoid) - 0.5
    layer_4_out_2 = layer_4_out_2 * 100

    #return layer_4_out_1,out_code_1,out_code_2
    return layer_4_out_1,layer_4_out_2,speaker_1_Code,speaker_2_Code

layer_4_out_1,layer_4_out_2,out_code_1,out_code_2 = ONNModel(indata)
out = layer_4_out_1+layer_4_out_2

s_1_w =  tf.nn.l2_normalize(weights['codelayer_1'],dim = 0)
s_2_w =  tf.nn.l2_normalize(weights['codelayer_2'],dim = 0)
orthogonal_err = tf.reduce_sum(tf.abs(tf.matmul(tf.transpose(s_2_w),s_1_w)))
code_1 = tf.abs(out_code_1)
code_2 = tf.abs(out_code_2)
rata = tf.reduce_sum(code_1*code_2/(code_1+ 10**-12 + code_2))
#loss_add = tf.reduce_sum(tf.square(layer_4_out_1-speaker_1)) + tf.reduce_sum(tf.square(layer_4_out_2-speaker_2))
#set_loss = tf.reduce_sum(tf.square(otdata-speaker_1-speaker_2))

cost = tf.reduce_sum(tf.square(otdata - out)) + 100 * (orthogonal_err + rata) #+ loss_add + set_loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rata).minimize(cost)

init = tf.global_variables_initializer()

def preprocessingData(trainData):
    standard_scaler = preprocessing.StandardScaler()
    x_train_standard = standard_scaler.fit_transform(trainData)
    return standard_scaler,x_train_standard

def setTrainData(train_filelist):
    signal_S1_Train = readSignal(train_filelist[0])
    signal_S2_Train = readSignal(train_filelist[1])

    signal_Mix_Train = readSignal(train_filelist[2])
    signal_Mix_Train = signal_Mix_Train * 50
    signal_S1_Train = signal_S1_Train * 50
    signal_S2_Train = signal_S2_Train * 50
    L = signal_S2_Train.shape[0]
    L = L // n_input
    length = L * n_input

    trainData = np.zeros([3*L,n_input])
    trainData[0:L,:] = signal_S1_Train[0:length].reshape([L,n_input])
    trainData[L:2*L,:] = signal_S2_Train[0:length].reshape([L,n_input])
    trainData[2*L:3*L,:]= signal_Mix_Train[0:length].reshape([L,n_input])
    
    target = np.ones([2,3*L])
    target[1,0:L] = 0
    target[0,L:2*L] = 0

    #standard_scaler,x_train_standard = preprocessingData(trainData)
    return trainData,target,None

def getTrainData(trainData,target,start,allindex=None):
    maxleen = len(trainData)

    if start + batch_size > maxleen or allindex is None:
        start = 0
        allindex= [i for i in range(0,maxleen)]
        allindex = np.random.permutation(allindex)
    
    index = allindex[start:start+batch_size]
    _,m = trainData.shape
    feature,label =  np.zeros([batch_size,m]),np.zeros([batch_size,2])
    feature = trainData[index,:]
    label = np.transpose(target[:,index])

    return feature,label,start+batch_size,allindex

def setTestData(standard_scaler,test_Filelist):
    signal_Mix_Test = readSignal(test_Filelist)
    signal_Mix_Test = signal_Mix_Test * 50
    L = signal_Mix_Test.shape[0]
    L = L // n_input
    length = L * n_input

    testData = signal_Mix_Test[0:length].reshape([L,n_input])
    #x_test_standard = standard_scaler.fit_transform(testData)
    return testData

def getTestData(index,testData):
    size = testData.shape[0]
    if index+10 < size:
        return testData[index:index+10,:]
    else:
        return testData[index:size,:]

def spec2sig(spec, mix,spec_12,sr=SAMPLE_RATE):
    #bmask = spec_12 / abs(mix)
    mask = spec / abs(spec_12)
    n,m = mask.shape
    mask[np.where(mask >= 0.5)] = 1
    mask[np.where(mask < 0.5)] = 0
    signal_spec = np.array(mix) * np.array(mask)
    sig = librosa.istft(signal_spec, hop_length=FFT_LEN // 2)
    return sig

with tf.Session() as sess:
    step = 1
    trainData,target_s,standard_scaler = setTrainData([r'D:\paper code\data\Track3_train.wav',r'D:\paper code\data\Track2_train.wav',r'D:\paper code\data\mix_train.wav'])
    testData = setTestData(standard_scaler,r'D:\paper code\data\mix_test.wav')
    start,allindex = 0,None
    ckpt_dir = "./Tf_Miackpt_dirt"
    saver = tf.train.Saver()
    TRAIN = True

    if TRAIN:
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        sess.run(init)
        while step * batch_size < 100 * trainData.shape[0]:
            feature,label,start,allindex = getTrainData(trainData,target_s,start,allindex)
            sess.run(optimizer, feed_dict={indata: feature, otdata: feature,target: label})

            step += 1
            if step % 1000 == 0:
                loss,err,rata_,c_1 = sess.run([cost,orthogonal_err,rata,out_code_1],feed_dict={indata: feature, otdata: feature,target:label})
                A,B = sess.run([speaker_1,speaker_2], feed_dict={indata: feature})
                print(loss,err,rata_)
                print(np.sum((A+B-feature)* (A+B-feature)))
        
        saver.save(sess,ckpt_dir+'/Tf_MiaModel.ckpt')
    else:
        model_file=tf.train.latest_checkpoint('Tf_Miackpt_dirt/')
        saver.restore(sess,model_file)
        n,m = testData.shape
        spec1,spec2 = np.zeros_like(testData),np.zeros_like(testData)
        m = n // 10
        for i in range(0,m):
            start = i * 10
            endl = start + 10
            A,B = sess.run([speaker_1,speaker_2], feed_dict={indata: testData[start:endl,:]})
            spec1[start:endl,:],spec2[start:endl,:] = A,B
        
        A,B = sess.run([speaker_1,speaker_2], feed_dict={indata: testData[endl:n,:]})
        spec1[endl:n,:],spec2[endl:n,:] = A,B
        
        #librosa.output.write_wav('11.wav',librosa.istft(np.transpose(spec1), hop_length=FFT_LEN // 2),16000)
        #librosa.output.write_wav('22.wav',librosa.istft(np.transpose(spec2), hop_length=FFT_LEN // 2),16000)
        def mask(sig_a,sig_b,mix_sig,hop_length=64):
            
            mix_sig = mix_sig[0]
            size = sig_a.shape[0]
            #maK =  np.abs(sig_a) / (np.abs(sig_a)+np.abs(sig_b))
            #maK[np.where((np.abs(sig_a)-np.abs(sig_b))>0)] = 1 
            #sig_a = mix_sig* maK
            #sig_b = mix_sig* (1-maK)

            sig_a_m = np.mean(np.abs(sig_a))
            sig_b_m = np.mean(np.abs(sig_b))

            gate_a = hop_length * sig_a_m * 0.5
            gate_b = hop_length * sig_b_m * 0.5

            for i in range(0,(size-hop_length)//10):
                start = i * 10
                end = start + hop_length
                a = np.sum(np.abs(sig_a[start:end]))
                b = np.sum(np.abs(sig_b[start:end]))
                if gate_a > a:
                    sig_a[start:start+10] = 0 #mix_sig[start:start+10] #- sig_b[i]
                    #sig_b[i] = 0#mix_sig[i] - sig_a[i]
                else:
                    pass
                    #sig_b[start:start+10] = mix_sig[start:start+10] #- sig_a[i]
                    #sig_a[i] = 0#mix_sig[i] - sig_b[i]
                if gate_b > b:
                    sig_b[start:end] = 0
            #hop_length = 128 
            '''
            for i in range(0,size//hop_length):
                start = i * hop_length
                end = (i+1) * hop_length
                if gate_a > np.sum(np.abs(sig_a[start:end])):
                    sig_a[start:end] = 0
                #Mask = np.abs(sig_a[start:end]**2)/(np.abs(sig_a[start:end])**2 + np.abs(sig_b[start:end])**2)
                #Mask[np.where(Mask > 0.9)] = 1
                #Mask[np.where(Mask < 0.1)] = 0
        
                #sig_a[start:end] = Mask *  mix_sig[start:end]
                
                #sig_b[start:end] = (1-Mask) *  mix_sig[start:end]
                if gate_b > np.sum(np.abs(sig_b[start:end])):
                    sig_b[start:end] = 0'''
            
            return sig_a,sig_b

        sig_a = spec1.reshape(1,-1)
        sig_b = spec2.reshape(1,-1)
        sig_a.dtype = np.float32
        sig_b.dtype = np.float32
        sig_a,sig_b = sig_a[0],sig_b[0]
        #sig_a,sig_b = mask(sig_a,sig_b,testData.reshape(1,-1))
        
        librosa.output.write_wav('tf1.wav',sig_a/25,16000)
        librosa.output.write_wav('tf2.wav',sig_b/25,16000)
        #spec_12 = np.transpose(spec1 + spec2 + 10**-12)

        #librosa.output.write_wav('1.wav',spec2sig(np.transpose(spec1),spec,spec_12),16000)
        #librosa.output.write_wav('2.wav',spec2sig(np.transpose(spec2),spec,spec_12),16000)
        