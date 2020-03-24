import tensorflow as tf 
import os 
import  numpy as np
import pickle
from confing import SAMPLE_RATE, FFT_LEN, CONSTANT, SPEECH_RANK
from sklearn import preprocessing 
from processData import Information_Struct
import librosa

training_iters = 5000000
learning_rata = 0.005

n_input = 129
n_otput = n_input
batch_size = 20
code_size = 60
#keep_prob = tf.placeholder(tf.float32) dropout

indata = tf.placeholder(tf.float32,[None,n_input])
otdata = tf.placeholder(tf.float32,[None,n_otput])
#otdata = tf.placeholder(tf.float32,[None,n_otput])
target = tf.placeholder(tf.float32,[None,2])

weights = {
    'wd1': tf.Variable(tf.random_normal([n_input, n_input])),
    'wd2': tf.Variable(tf.random_normal([n_input,n_input])),
    'wd_1_ST': tf.Variable(tf.random_normal([n_input])),
    'codelayer_1': tf.Variable(tf.random_normal([n_input, 64])) * 0.01,
    'codelayer_2': tf.Variable(tf.random_normal([n_input, 64])) * 0.01,
    'wd3': tf.Variable(tf.random_normal([64, 64])),
    'out': tf.Variable(tf.random_normal([64, n_input],mean=0, stddev=0.25)),
}
biases = {
    'bc1': tf.Variable(tf.random_normal([n_input])),
    'bc2': tf.Variable(tf.random_normal([n_input])),
    'codelayer_b1': tf.Variable(tf.random_normal([64]))* 0.01,
    'codelayer_b2': tf.Variable(tf.random_normal([64]))* 0.01,
    'bc3': tf.Variable(tf.random_normal([64])),
    'out': tf.Variable(tf.random_normal([n_input])),
}

def fullConnect(inputData,weight,biases,actiation):
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
    
    #speaker_1_Code = tf.layers.batch_normalization(speaker_1_Code, training=True)
    #类似dropout  out_code_1 out_code_2
    out_code_1 = tf.transpose(tf.multiply(tf.transpose(speaker_1_Code), target[:,0])) 
    out_code_2 = tf.transpose(tf.multiply(tf.transpose(speaker_2_Code), target[:,1]))
    #out_code = out_code_1 + out_code_2 
    #decode layer
    layer_3_out_1 = fullConnect(out_code_1,weights['wd3'],0,tf.nn.tanh)
    layer_4_out_1 = fullConnect(layer_3_out_1,weights['out'],0,tf.nn.relu)

    layer_3_out_2 = fullConnect(out_code_2,weights['wd3'],0,tf.nn.tanh)
    layer_4_out_2 = fullConnect(layer_3_out_2,weights['out'],0,tf.nn.relu)


    return layer_4_out_1 , layer_4_out_2, layer_3_out_1, layer_3_out_2,speaker_1_Code,speaker_2_Code,out_code_1,out_code_2
    #return layer_4_out_1 + layer_4_out_2,speaker_1_Code,speaker_2_Code

layer_4_out_1 , layer_4_out_2,out_code_1,out_code_2,speaker_1_Code,speaker_2_Code,code_1,code_2 = ONNModel(indata)
out = layer_4_out_1 + layer_4_out_2
#正交约束
s_1_w =  tf.nn.l2_normalize(weights['codelayer_1'],dim = 0)
s_2_w =  tf.nn.l2_normalize(weights['codelayer_2'],dim = 0)
orthogonal_err = tf.reduce_mean(tf.abs(tf.matmul(tf.transpose(s_2_w),s_1_w)))

#稀疏化约束
code_1 = tf.abs(out_code_1) 
code_2 = tf.abs(out_code_2) 
rata = tf.reduce_sum((code_1*code_2 )/(code_1+ 10**-12 + code_2))

#loss_add = tf.reduce_sum(tf.square(speaker_1_Code-code_1)) + tf.reduce_sum(tf.square(speaker_2_Code-code_2))

MES = tf.reduce_sum(tf.square(out - otdata))
cross_entropy = tf.reduce_mean(10 ** -11/ (out + 10 ** -12)) #loss_add
cost =cross_entropy + orthogonal_err + rata + MES #+ loss_add


optimizer = tf.train.AdamOptimizer(learning_rate=learning_rata).minimize(cost)

init = tf.global_variables_initializer()

def preprocessingData(trainData):
    standard_scaler = preprocessing.StandardScaler()
    x_train_standard = standard_scaler.fit_transform(trainData)
    return standard_scaler,x_train_standard

def setTrainData(train_filelist):
    #train_filelist = ['','']
    target,trainData = Information_Struct(train_Filelist = train_filelist).getTrainData()
    trainData = np.transpose(trainData)
    #trainData = preprocessing.normalize(trainData,norm = 'l2',axis= 1)
    standard_scaler,x_train_standard = preprocessingData(trainData)
    return trainData,target,standard_scaler

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
    label = target[index,:]

    return feature,label,start+batch_size,allindex

def setTestData(standard_scaler,test_Filelist):
    #test_Filelist = ['','']
    testData,spec = Information_Struct(test_Filelist = test_Filelist).getTestData()
    testData = np.transpose(testData)
    #x_test_standard = standard_scaler.fit_transform(testData)
    return testData,spec

def getTestData(index,testData):
    size = testData.shape[0]
    if index+10 < size:
        return testData[index:index+10,:]
    else:
        return testData[index:size,:]

def spec2sig(spec, mix,spec2,sr=SAMPLE_RATE):
    eps = 10 ** -20
    #bmask = spec_12 / abs(mix)
    mask = (spec) / np.abs(spec2+spec+eps)
    #n,m = spec.shape

            
    #print(count/n/m)
    mask[np.where(mask >= 0.5)] = 1
    mask[np.where(mask < 0.5)] = 0
    signal_spec = np.array(mix) * np.array(mask)
    sig = librosa.istft(signal_spec, hop_length=FFT_LEN // 2)
    return sig

with tf.Session() as sess:
    step = 1 
    trainData,target_s,standard_scaler = setTrainData([r'D:\paper code\data\Track3_train.wav',r'D:\paper code\data\Track2_train.wav',r'D:\paper code\data\mix_train.wav'])
    testData,spec = setTestData(standard_scaler,[r'D:\paper code\data\mix_test.wav'])
    start,allindex = 0,None

    ckpt_dir = "./Miackpt_dir"
    saver = tf.train.Saver()
    TRAIN = False

    if TRAIN:
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        sess.run(init)
        while step * batch_size < 40 * trainData.shape[0]:
            feature,label,start,allindex = getTrainData(trainData,target_s,start,allindex)
            sess.run(optimizer, feed_dict={indata: feature, otdata: feature,target: label})

            step += 1
            if step % 1000 == 0:
                loss,err,ce,rata_ ,code_1= sess.run([cost,orthogonal_err,cross_entropy,rata ,out_code_1],feed_dict={indata: feature, otdata: feature,target:label})
                A,B = sess.run([layer_4_out_1 , layer_4_out_2], feed_dict={indata: feature,target: np.ones_like(label)})
                C = A + B
                print(loss,err,ce,rata_)
                print(np.sum((C-feature)* (C-feature)))
        
        saver.save(sess,ckpt_dir+'/MiaModel.ckpt')
    else:
        model_file=tf.train.latest_checkpoint('Miackpt_dir/')
        saver.restore(sess,model_file)
        n,m = testData.shape
        spec1,spec2 = np.zeros_like(testData),np.zeros_like(testData)
        m = n // 10
        label = np.ones([n,2])
        for i in range(0,m):
            start = i * 10
            endl = start + 10
            A,B = sess.run([layer_4_out_1 , layer_4_out_2], feed_dict={indata: testData[start:endl,:],target: label[start:endl,:]})
            spec1[start:endl,:],spec2[start:endl,:] = A,B
        
        A,B = sess.run([layer_4_out_1 , layer_4_out_2], feed_dict={indata: testData[endl:n,:],target: label[endl:n,:]})
        spec1[endl:n,:],spec2[endl:n,:] = A,B

        spec1,spec2 = np.transpose(spec1),np.transpose(spec2)
        sig_a = spec2sig(spec1,spec,spec2)
        sig_b = spec2sig(spec2,spec,spec1)
        #mask = np.abs(sig_a) - np.abs(sig_b) 
        #mask[np.where(mask >= 0)] = 1
        #mask[np.where(mask < 0)] = 0
        #sig_a = mask * sig_a
        #sig_b = (1-mask) * sig_b

        librosa.output.write_wav('1.wav',sig_a,16000)
        librosa.output.write_wav('2.wav',sig_b,16000)

        #librosa.output.write_wav('11.wav',librosa.istft(np.transpose(IBM)*spec, hop_length=FFT_LEN // 2),16000)
        #librosa.output.write_wav('22.wav',librosa.istft(np.transpose(1- IBM)* spec, hop_length=FFT_LEN // 2),16000)
        
        '''spec_12 = np.transpose(spec1 + spec2)
        sig_a = spec2sig(np.transpose(spec1),spec,spec_12)
        sig_b = spec2sig(np.transpose(spec2),spec,spec_12)

        def mask(sig_a,sig_b,mix_sig,hop_length=64):
            size = sig_a.shape[0]
            for i in range(0,size//hop_length):
                start = i * hop_length
                end = (i+1) * hop_length
                if np.sum(sig_a[start:end]*sig_a[start:end]) > np.sum(sig_b[start:end]*sig_b[start:end]):
                    #sig_a[start:end] = mix_sig[start:end]
                    sig_b[start:end] = 0
                else:
                    #sig_b[start:end] = mix_sig[start:end]
                    sig_a[start:end] = 0
            return sig_a,sig_b

        mix_sig = librosa.istft(spec, hop_length=FFT_LEN // 2)
        #sig_a,sig_b = mask(sig_a,sig_b,mix_sig)

        #sig_a,sig_b = spec2sig
        librosa.output.write_wav('mix.wav',mix_sig,16000)
        librosa.output.write_wav('1.wav',sig_a,16000)
        librosa.output.write_wav('2.wav',sig_b,16000)'''
        