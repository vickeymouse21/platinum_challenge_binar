from flask import Flask, jsonify
from flask import request
import flask_ngrok
from flasgger import Swagger, LazyJSONEncoder, LazyString, swag_from
import re
import pandas as pd
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nlp_id.lemmatizer import Lemmatizer
from sklearn.neural_network import MLPClassifier
import pickle

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import defaultdict
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, SimpleRNN, Activation
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.layers import Flatten
from tensorflow.keras import backend as K
from keras.models import load_model

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
stemmer = StemmerFactory().create_stemmer()


app = Flask(__name__)
app.json_encoder = LazyJSONEncoder
swagger_template = dict(
info = {
    'title': LazyString(lambda:'API for Predicting Sentiment'),
    'version': LazyString(lambda:'1.0.0'),
    'description': LazyString(lambda:'API untuk Prediksi Sentimen'),
    },
    host = LazyString(lambda:request.host)
)
swagger_config = {
    'headers':[],
    'specs':[
        {
            'endpoint': 'docs',
            'route': '/docs.json',
        }
    ],
    'static_url_path':'/flasgger_static',
    'swagger_ui': 'True',
    'specs_route': '/'
}
swagger = Swagger(app,template=swagger_template,config=swagger_config)

#CLEANSING FOR NEURAL NETWORK
def bersihkan(text):
    text = text.lower()
    text = re.sub('\w+[0-9]\w+',' ',text)
    text = re.sub('__\w+__',' ',text)
    text = re.sub(r'[^A-Za-z0-9]',' ',text)
    text = re.sub('( ){2,10}',' ',text)
    return text

#LEMATISASI
def lemas(text):
    lema = Lemmatizer()
    text = lema.lemmatize(text)
    return text
#CALL FITUR UNTUK NEURAL NETWORK
def call_fitur():
    return pickle.load(open('fitur_fix.p','rb'))
#CALL MODEL UNTUK NEURAL NETWORK
def call_model():
    return pickle.load(open('model_fix.p','rb'))


#bagian untuk memanggil isi tabel kata alay
conn = sqlite3.connect('database_hate.db')
call_alay = pd.read_sql_query('SELECT * FROM kamus_alay',conn)

alay = dict(zip(call_alay['kata_alay'],call_alay['kata_normal']))

#fungsi untuk mengganti kata alay ke normal
def normalize(text):
    hasil = []
    splitting = text.split(' ')
    for kata in splitting:
        if kata in alay:
            hasil.append(alay[kata])
        else:
            hasil.append(kata)
    
    return ' '.join(hasil)

#LABEL UNTUK LSTM MODEL
sentiment = ['negative', 'neutral', 'positive']

#LOAD SIMPENAN TOKENIZER LSTM
file = open("tokenizer_2.pickle",'rb')
tokenizer = pickle.load(file)
file.close()

#FUNGSI UNTUK LOAD MODEL LSTM
def lstm_model():
    model = load_model('model_plat_sastrawi2.h5')
    return model

#LOAD SIMPENAN PAD SEQUENCES LSTM
file = open("x_pad_sequences_2.pickle",'rb')
X = pickle.load(file)
file.close()

#CLEANSING UNTUK LSTM
def cleansing(text):
    text = re.sub('[^A-Za-z0-9]',' ',text)
    text = re.sub('( ){2,13}',' ',text)
    return text

def stemming(text):
    text = stemmer.stem(text)
    return text


#NEURAL NETWORK
@swag_from('docs/text_processing_nn.yml', methods=['POST'])
@app.route('/1text-processing-nn',methods=['POST'])
def text_processing_nn():

    text_ori = request.form.get('text')
    text = bersihkan(text_ori)
    text = normalize(text)
    text = lemas(text)
    vect = call_fitur()

    vectorizing = vect.transform([text])
    
    model = call_model()
    
    result = model.predict(vectorizing)[0]
        

    json_response = {
        'text asli': text_ori,
        'Jenis Sentimen': result,
        'cleaned text' : text
    }

    response = jsonify(json_response)
    return response

@swag_from('docs/text_processing_file_nn.yml', methods=['POST'])
@app.route('/2text-processing-file-nn',methods=['POST'])
def text_processing_file_nn():

    file = request.files.getlist('file')[0]

    original_text_list = []
    cleaned_text_list = []
    sentimen_list = []
    for text in file:
        text = bytes.decode(text, 'latin-1')

        original_text_list.append(text)

        cleaned = bersihkan(text)
        text_bersih = normalize(cleaned)
        text = lemas(text_bersih)

        vect = call_fitur()
        vectorizing = vect.transform([text])
        model = call_model()
        result = model.predict(vectorizing)[0]

        cleaned_text_list.append(text)
        sentimen_list.append(result)
        

    json_response = {
        'text asli': original_text_list,
        'Jenis Sentimen': sentimen_list,
        'cleaned text' : cleaned_text_list
    }

    response = jsonify(json_response)
    return response

#LSTM
@swag_from('docs/text_processing_lstm.yml', methods=['POST'])
@app.route('/3text-processing-lstm',methods=['POST'])
def text_processing():

    text_ori = request.form.get('text')
    
    #CLEANSING
    text = [cleansing(text_ori)]
    text = [stemming(text[0])]

    #PREDICTING
    predicted = tokenizer.texts_to_sequences(text)
    guess = pad_sequences(predicted, maxlen=X.shape[1])
    model = lstm_model()
    prediction = model.predict(guess)
    polarity = np.argmax(prediction[0])

    json_response = {
        'text asli': text_ori,
        'Jenis Sentimen': sentiment[polarity],
        'cleaned text' : text
    }

    response = jsonify(json_response)
    return response

@swag_from('docs/text_processing_file_lstm.yml', methods=['POST'])
@app.route('/4text-processing-file',methods=['POST'])
def text_processing_file():

    file = request.files.getlist('file')[0]

    original_text_list = []
    cleaned_text_list = []
    sentimen_list = []
    for text in file:
        #decoding
        text = bytes.decode(text, 'latin-1')
        
        #MENYIMPAN TEXT ORI
        original_text_list.append(text)
        
        #CLEANSING
        text = [cleansing(text)]
        text = [stemming(text[0])]

        #MENYIMPAN CLEANED TEXT
        cleaned_text_list.append(text)

        #PREDICTING
        predicted = tokenizer.texts_to_sequences(text)
        guess = pad_sequences(predicted, maxlen=X.shape[1])
        model = lstm_model()
        prediction = model.predict(guess)
        polarity = np.argmax(prediction[0])

        #MENYIMPAN HASIL PREDIKSI SENTIMEN
        sentimen_list.append(sentiment[polarity])
        

    json_response = {
        'text asli': original_text_list,
        'Jenis Sentimen': sentimen_list,
        'cleaned text': cleaned_text_list
    }

    response = jsonify(json_response)
    return response


#menjalankan flask    
if __name__ == '__main__':
    app.run(debug=True)
