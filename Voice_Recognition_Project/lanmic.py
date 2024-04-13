import socket
import numpy as np
import struct
import time
from scipy.io.wavfile import write 

import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import csv
from statistics import mean
from sklearn.model_selection import train_test_split
from sklearn import datasets, svm, metrics
#LanMIC programming
def record(host_ip, samplerate, AqTime): #host ip adress from lanmic , #samplerate from lanmic, #Aquisition time in seconds 
    
    print ('The begin (lanmic)...')

    #wavFile = input('Wav filename: ')
    wavFile = 'predict_sample.wav'
    #AqTime = int(input('Aquisition time (s): '))
    #settings
    host = host_ip #ip address of the mobile

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        #host = "192.168.1.155"
        port = 8080
        
        print('Opening socket')
        s.connect((host, port))


        chunk_size = 1024 # 512
        #audio_format = pyaudio.paInt16
        channels = 1
        #samplerate = 22050 #(LANmic sample rate /2)
        samplerate = int(samplerate/2)    #(LANmic sample rate /2)

        print('connected to server\n')
        
        print('Sound being acquired ...')
        
        #wait for data to be aquired before start
        time.sleep(chunk_size/samplerate)
        data=s.recv(chunk_size)
        t0=time.time()
        #aquire for a period of time
        while time.time()-t0 < AqTime:
            time.sleep(chunk_size/samplerate/4)
            data+=s.recv(chunk_size)
        # to flush the buffer
        time.sleep(0.2)
        data+=s.recv(chunk_size)
        
        print('... finished')

        def get_max(data):
            l = len(data)
            for i in range(l):
                if (l - i - 180)%4 == 0:
                    max = l - i
                    break
                i += 1
            return max

        l=len(data)
        
        print('Length of data (3)= ', l)
       
        #Convert to numpy array
        npdata=np.frombuffer(data[180:get_max(data)], dtype=np.int32)
        
        #save data   
        write(wavFile, samplerate, npdata);     
        print('wav file written\n')
        
        #close socket
        s.close()   
        print('socket closed') 
    print ('... end')
    
    return wavFile