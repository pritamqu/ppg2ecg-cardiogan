import os
dirname, filename = os.path.split(os.path.abspath(__file__))
print("running: {}".format(filename) )
os.chdir(dirname)
script_name = filename
current_folder = dirname
import logging
logging.getLogger('tensorflow').disabled = True
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
logging.getLogger('tensorflow').disabled = True

import socket
import time
import numpy as np
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
import cv2
import sklearn.preprocessing as skp

import tflib
import module 
import preprocessing

tf.keras.backend.set_floatx('float64')
tf.autograph.set_verbosity(0)


def connect(deviceID
            , serverAddress = '127.0.0.1'
            , serverPort = 28000
            , bufferSize = 4096 
            ):
    
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(3)

    print("Connecting to server")
    s.connect((serverAddress, serverPort))
    print("Connected to server\n")

    print("Devices available:")
    s.send("device_list\r\n".encode())
    response = s.recv(bufferSize)
    print(response.decode("utf-8"))

    print("Connecting to device")
    s.send(("device_connect " + deviceID + "\r\n").encode())
    response = s.recv(bufferSize)
    print(response.decode("utf-8"))

    print("Pausing data receiving")
    s.send("pause ON\r\n".encode())
    response = s.recv(bufferSize)
    print(response.decode("utf-8"))
    
    return s

def suscribe_to_data(s
                     , acc=False
                     , bvp=True
                     , gsr=False
                     , tmp=False
                     , bufferSize = 4096
                     ):
    if acc:
        print("Suscribing to ACC")
        s.send(("device_subscribe " + 'acc' + " ON\r\n").encode())
        response = s.recv(bufferSize)
        print(response.decode("utf-8"))
    if bvp:
        print("Suscribing to BVP")
        s.send(("device_subscribe " + 'bvp' + " ON\r\n").encode())
        response = s.recv(bufferSize)
        print(response.decode("utf-8"))
    if gsr:
        print("Suscribing to GSR")
        s.send(("device_subscribe " + 'gsr' + " ON\r\n").encode())
        response = s.recv(bufferSize)
        print(response.decode("utf-8"))
    if tmp:
        print("Suscribing to Temp")
        s.send(("device_subscribe " + 'tmp' + " ON\r\n").encode())
        response = s.recv(bufferSize)
        print(response.decode("utf-8"))

    print("Resuming data receiving")
    s.send("pause OFF\r\n".encode())
    response = s.recv(bufferSize)
    print(response.decode("utf-8"))
    
    return s

@tf.function
def sample_P2E(P, model):
    fake_ecg = model(P, training=False)
    return fake_ecg

########### params ###########
ecg_sampling_freq = 128
ppg_sampling_freq = 128
window_size = 4
ecg_segment_size = ecg_sampling_freq*window_size
ppg_segment_size = ppg_sampling_freq*window_size
model_dir = 'path'
output_dir = 'path'
os.makedirs(output_dir)
ep = open(os.path.join(output_dir, "ecg_ppg_recordings.txt"), "w+")
tb_step = 0

bufferSize = 10240  
deviceID = 'AAAAAA'      
e4 = connect(deviceID)
e4 = suscribe_to_data(e4)
response = e4.recv(bufferSize).decode("utf-8")

train_summary_writer = tf.summary.create_file_writer(os.path.join(output_dir, 'summary'))

""" model """
Gen_PPG2ECG = module.generator_attention()
""" resotre """
tflib.Checkpoint(dict(Gen_PPG2ECG=Gen_PPG2ECG), model_dir).restore()

data_list   = []
PPG         = []
ECG         = []
with train_summary_writer.as_default():
    try:
        while True:    
            try:
                ## process 1
                response = e4.recv(bufferSize).decode("utf-8")
                samples = response.split("\n")
                print('receiving ppg: ', datetime.now())
                for i in range(1, len(samples)-1): ## removing 1st and last point.. may be broken
                
                    # stream_type = samples[i].split()[0]
                    # timestamp = float(samples[i].split()[1].replace(',','.'))
                    data = float(samples[i].split()[2].replace(',','.'))
                    data_list.append(data)
                
                ## process 2
                if len(data_list)>=256:
                    print('input PPG: ', datetime.now())
                    x_ppg = np.array(data_list[:256])
                    data_list   = data_list[256:]                    
                    x_ppg = cv2.resize(x_ppg, (1,ppg_segment_size), interpolation = cv2.INTER_LINEAR)
                    x_ppg = x_ppg.reshape(1, -1)
                    x_ppg = preprocessing.filter_ppg(x_ppg, 128)
                    x_ppg = skp.minmax_scale(x_ppg, (-1, 1), axis=1)
                    
                    x_ecg = sample_P2E(x_ppg, Gen_PPG2ECG)
                    
                    x_ecg = x_ecg.numpy()
                    x_ecg = preprocessing.filter_ecg(x_ecg, 128)
                    x_ppg = x_ppg.reshape(-1)
                    x_ecg = x_ecg.reshape(-1)
                    print('output ECG: ', datetime.now())
                    
                    for plot_points in range(x_ecg.shape[0]):
                        
                        tf.summary.scalar('FECG', x_ecg[plot_points], step=tb_step)
                        tf.summary.scalar('RPPG', x_ppg[plot_points], step=tb_step)
                        tb_step+=1
                        
                                                        
                    ECG.append(x_ecg)
                    PPG.append(x_ppg)
        
                    np.savetxt(ep, np.c_[x_ecg,x_ppg], fmt='%1.8f %1.8f')
                                   
    
            except:
                e4 = connect(deviceID)
                e4 = suscribe_to_data(e4)            
    except KeyboardInterrupt:
        print("Disconnecting from device")          
        e4.send("device_disconnect\r\n".encode())
        e4.close()
        ep.close()


# tensorboard --logdir=./output/realtime --host localhost --port 8088 --samples_per_plugin scalars=0 --reload_interval 1
