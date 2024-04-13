import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
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
from flask import Flask, request, jsonify, render_template
from lanmic import record
from feature_extract import extract



class_labels = []
mfcc = []
with open('total_final_dataset.csv','r') as f:
    csvreader = csv.reader(f)
    for row in csvreader:
        class_labels.append(row[0])
        mfcc.append([row[2], row[6], row[11]])


class_labels.pop(0)
mfcc.pop(0)

#training
clf = svm.SVC()
clf.fit(mfcc, class_labels)


pickle.dump(clf, open('model.pkl','wb')) #saving the model





#############Running App#####



app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    settings = [x for x in request.form.values()]
    wav_file_name = record(settings[0], int(settings[1]), int(settings[2]))
    output = extract(wav_file_name, clf)
    
    if output == ["Silence"]:
       return render_template('index.html', prediction_text='There is no speaker') 
    else:
       return render_template('index.html', prediction_text='The speaker is {}'.format(output[0]))

#@app.route('/results',methods=['POST'])
#def results():

    #data = request.get_json(force=True)
    #prediction = model.predict([np.array(list(data.values()))])

    #output = prediction
    #return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)










#from lanmic import record

#record('192.168.1.133' ,44100, 10)



