from flask import Flask, render_template, send_file, make_response, url_for, Response, redirect, request   
import numpy as np
import moviepy.editor as mp
import networkx as nx
from nltk.tokenize.punkt import PunktSentenceTokenizer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import os
import pandas as pd
import re
import string
from ibm_watson import SpeechToTextV1
from ibm_watson.websocket import RecognizeCallback, AudioSource 
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import requests
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
import nltk
nltk.download('wordnet')

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

def clean_data(x):

    x = re.sub("%HESITATION", '', x)
    x = re.sub("%HESITATION", '', x)
    x = re.sub("&.+;", '', x)
    x = re.sub("\[.+\]", '', x)
    x = re.sub("[MUSIC]", '', x)
    x = re.sub("-+", '', x)
    
    _RE_COMBINE_WHITESPACE = re.compile(r"\s+")
    x = _RE_COMBINE_WHITESPACE.sub(" ", x).strip()
    return x

def getTranscripts(res):
    a=len(res['results'])
    text=""
    for i in range(0,a):
        text = text+ res['results'][i]['alternatives'][0]['transcript']
    return text

def generateTranscripts(filepath,stt):
    
    with open(filepath, 'rb') as f:
        res = stt.recognize(audio=f, content_type='audio/mp3', model='en-US_NarrowbandModel', continuous=True,inactivity_timeout=90).get_result()
    return res

def tokenize(document):
    # We are tokenizing using the PunktSentenceTokenizer
    # we call an instance of this class as sentence_tokenizer
    doc_tokenizer = PunktSentenceTokenizer()
    
    # tokenize() method: takes our document as input and returns a list of all the sentences in the document
    
    # sentences is a list containing each sentence of the document as an element
    sentences_list = doc_tokenizer.tokenize(document)
    return sentences_list

def lemmatize_stemming(text):
    stemmer = SnowballStemmer('english')
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

@app.route("/convertToTranscript/",methods=["POST"])
def getTranscript():
    mp4File = request.files['video_file']
    if mp4File.filename != '':
        mp4File.save(mp4File.filename)
    mp4File=mp4File.filename
    mp3File='temp.mp3'
    apikey = 'bINr1qlr1OcXS4qvVfb9E4fTORJCFAHfJOrI1upql4kr'
    url = 'https://api.eu-gb.speech-to-text.watson.cloud.ibm.com/instances/c510c2cf-a3b3-44d8-9892-225f63de839c'
    authenticator = IAMAuthenticator(apikey)
    stt = SpeechToTextV1(authenticator=authenticator)
    stt.set_service_url(url)
    print(authenticator)
    video = mp.VideoFileClip(mp4File) 
    video.audio.write_audiofile(mp3File)
    res=generateTranscripts(mp3File,stt)
    res=getTranscripts(res)
    res=clean_data(res)
    data = {
    'text': res
        }

    response = requests.post('http://bark.phon.ioc.ee/punctuator', data=data)
    res=response.text
   
    return render_template('transcript.html', data=res)
    
    
@app.route("/submitTranscript/",methods=["POST"])
def summarise():
    uploaded_file = request.files['txt_file']
    
    #making sure its not empty
    if uploaded_file.filename != '':
        #reading the file
        text = uploaded_file.read()
        #converting to a string.
        document = str(text)
        document = clean_data(document)
        '''data = {
        'text': document
            }
        response = requests.post('http://bark.phon.ioc.ee/punctuator', data=data)
        document=response.text'''
    sentences_list = tokenize(document)
    cv = CountVectorizer()
    cv_matrix = cv.fit_transform(sentences_list)
    cv_demo = CountVectorizer()
  
    
    normal_matrix = TfidfTransformer().fit_transform(cv_matrix)
    res_graph = normal_matrix * normal_matrix.T
    nx_graph = nx.from_scipy_sparse_matrix(res_graph)
    ranks = nx.pagerank(nx_graph)

    sentence_array = sorted(((ranks[i], s) for i, s in enumerate(sentences_list)), reverse=True)
    sentence_array = np.asarray(sentence_array)
    
    rank_max = float(sentence_array[0][0])
    rank_min = float(sentence_array[len(sentence_array) - 1][0])
    temp_array = []

    flag = 0
    if rank_max - rank_min == 0:
        temp_array.append(0)
        flag = 1
    
  
    if flag != 1:
        for i in range(0, len(sentence_array)):
            temp_array.append((float(sentence_array[i][0]) - rank_min) / (rank_max - rank_min))
    
    

    
    threshold = (sum(temp_array) / len(temp_array)) + 0.2
    sentence_list = []
    if len(temp_array) > 1:
        for i in range(0, len(temp_array)):
            if temp_array[i] > threshold:
                    sentence_list.append(sentence_array[i][1])
    else:
        sentence_list.append(sentence_array[0][1])
    
   
    summary = " ".join(str(x) for x in sentence_list)
    return render_template('summary.html', data=summary)



@app.route("/getTopic/",methods=["GET","POST"])
def getTopicLabel():
    text = request.form.get("inputFileTopic")
    

    document = str(text)
    lda_model=gensim.models.ldamodel.LdaModel.load("C:\\Users\\farhe\\Downloads\\capstone_website\\online_summarizer\\trainedTopic.model")
    dictionary=gensim.corpora.Dictionary.load('C:\\Users\\farhe\\Downloads\\capstone_website\\online_summarizer\\dictionaryFile.dic')
   
    bow_vector = dictionary.doc2bow(preprocess(document))
    sorted_scores = sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1])
    
    index = sorted_scores[0][0]
    score = sorted_scores[0][1]
    
    ideal_topic = lda_model.print_topic(index, 5)
    print(ideal_topic)
    li= [i.split('*')[1] for i in ideal_topic.split(' + ')]
    d={}
    
    for i in range(0,len(li)):
        if i not in d:
            s=li[i]
            s=s[1:len(s)-1]
            d[i]=s
    return render_template('topic.html',output=d)
   

@app.route("/goBack/",methods=["POST"])       
def goBack():    
    return render_template("index.html")
    
        
    
if __name__ == "__main__":
    app.run()
 