def extract(wavFile):
    #Power as a rejection class 
    average = []
    data, sampling_rate = librosa.load(wavFile)
    average = mean(librosa.feature.rms( y=data, frame_length=552, hop_length=221, center=True, pad_mode='constant')[0])

    #Feature Extraction
    if average > 0.02:
        mfcc_avg_list =[]
        average = []
        print(wavFile)
        
        data, sampling_rate = librosa.load(wavFile)
        n = int(sampling_rate*3/(4*1000)+1) #n of coeficients for mfcc 
        
        data_trim, index = librosa.effects.trim(data, top_db=20, frame_length = 552, hop_length = 221 )#trims ends of wave when they are below 20 db
        data_clean = librosa.effects.split(data_trim, top_db =20, frame_length = 552, hop_length = 221) #returns wave intervals above 20db
        data_trim_clean = []

        for i in range(0, len(data_clean)):
            data_trim_clean = data_trim_clean + data_trim[data_clean[i][0]: data_clean[i][1]].tolist()  
        data_trim_clean = np.array(data_trim_clean)

        mfcc_2= mean(librosa.feature.mfcc(y=data_trim_clean, sr=sampling_rate, n_mfcc= n+1, n_fft = 552, hop_length = 221)[1])
        mfcc_6= mean(librosa.feature.mfcc(y=data_trim_clean, sr=sampling_rate, n_mfcc= n+1, n_fft = 552, hop_length = 221)[5])
        mfcc_11= mean(librosa.feature.mfcc(y=data_trim_clean, sr=sampling_rate, n_mfcc= n+1, n_fft = 552, hop_length = 221)[10])
        
        print(mfcc_2)
        print(mfcc_6)
        print(mfcc_11)
       
        test_mfcc =[[mfcc_2,mfcc_6,mfcc_11]]
        return clf.predict(test_mfcc)
    else:
        return ["Silence"]