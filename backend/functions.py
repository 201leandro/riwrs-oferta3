from textblob import TextBlob
import robobrowser
import nltk
import re
import pandas as pd
import string
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
import time
import datetime
import sys
import jsonpickle
import os
import tweepy
import json

from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer

from all_nomes import nomes
from all_sinonimos import sinonimos
from all_otherwords import otherwords
from all_stopwords import stopwords

def get_tweet(product = 'riachuelo'):


    consumer_key = 'lp23SZhUq9pI0CiQqL6Re2WEt'
    consumer_secret = 'CTR3NT3gvPPPLPMVkiEHB1M0Q9F6c84xzlAeRQv6mau0rYa67h'
    access_token = '1030921121493868544-qPqiUFJa02bT59YePKx4FE6kehCKIR'
    access_token_secret = 'zd48eKEwEp6juiAPmj0KcFRtEC65vK5deCNa5fQrGcKr4'

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    api = tweepy.API(auth)

    searchQuery = product  
    maxTweets = 30000 
    tweetsPerQry = 100  
    fName = 'tweets_' + searchQuery + '.txt' 
    lang = 'pt'

    sinceId = None

    max_id = -1

    tweetCount = 0

    #print("Downloading max {0} tweets".format(maxTweets))
    with open('tweets_' + searchQuery + '.txt', 'w') as f:
        while tweetCount < maxTweets:
            try:
                if (max_id <= 0):
                    if (not sinceId):
                        new_tweets = api.search(q=searchQuery,lang=lang, count=tweetsPerQry)
                    else:
                        new_tweets = api.search(q=searchQuery,lang=lang, count=tweetsPerQry,
                                                since_id=sinceId)
                else:
                    if (not sinceId):
                        new_tweets = api.search(q=searchQuery,lang=lang, count=tweetsPerQry,
                                                max_id=str(max_id - 1))
                    else:
                        new_tweets = api.search(q=searchQuery,lang=lang, count=tweetsPerQry,
                                                max_id=str(max_id - 1),
                                                since_id=sinceId)
                if not new_tweets:

                    break
                for tweet in new_tweets:
                    f.write(jsonpickle.encode(tweet._json, unpicklable=False) +
                            '\n')
                tweetCount += len(new_tweets)
                #print("Downloaded {0} tweets".format(tweetCount))
                max_id = new_tweets[-1].id

            except tweepy.TweepError as e:

                break

    data_ = open('tweets_' + searchQuery +'.txt', 'r')
    data_r = data_.readlines()
    data = []
    for line in data_r:
        d = json.loads(line)
        data.append(d['text'])
    
    data = pd.DataFrame(data, columns = ['Text'])
        
    return data

def get_prod(product = 'playstation 2'):

    url = 'https://www.mercadolivre.com.br/'
    browser = robobrowser.RoboBrowser(history=True, parser='html.parser')
    browser.open(url)
    form = browser.get_form(action = 'https://www.mercadolivre.com.br/jm/search')
    browser.open(url)
    form['as_word'].value = product
    browser.submit_form(form)
    sense = []
    comments = []
    pags_un = browser.find_all(class_ = 'pagination__page')[2:]
    i = 0
    while i < len(pags_un):
        #print('Página:', i+1)
        item_t1 = browser.find_all('a', class_ = 'item__info-title')
        if len(item_t1) == 0:
            itens = browser.find_all('a', class_ = 'item-link item__js-link')
            t1 = False
        else:
            itens = item_t1
            t1 = True
        for item in itens:
            if t1:
                item_link = str(item)[34:].split('"')[0]
            else:
                item_link = str(item)[41:].split('"')[0]
            browser.open(item_link)
            for com in browser.find_all(class_="review-tooltip expanded"):
                n_stars = len(com.find_all(class_ = "ch-icon-star star-icon star-icon-full"))
                text = str(com.find(class_="review-tooltip-full-body"))[38:].split('<')[0]
                if n_stars > 3:
                    sense.append(1)
                    comments.append(text)
                if n_stars < 3:
                    sense.append(-1)
                    comments.append(text)
                if n_stars == 3:
                    sense.append(0)
                    comments.append(text)
            #print('Produto avaliado')
        browser.open(str(pags_un[i])[39:].split('"')[0])
        i += 1
    x = sense.count(-1)/len(sense)
    #print('Proporção de negativos:', x)
    #print("Total:", len(sense))

    product = product.replace(' ', '_')
    data = []
    for i in range(len(comments)):
        data.append([comments[i], sense[i]])
    data = pd.DataFrame(data, columns = ["Text", "Classificacao"])
    #data.to_csv("data_" + product + ".csv", index = None)
    return data

#Geraçao de dataset
def get_politics():

    sense = []
    text = []
    url_ = 'http://www.votenaweb.com.br'
    url_all= ['http://www.votenaweb.com.br/politicos?apenas=Senador&page=1',
           'http://www.votenaweb.com.br/politicos?apenas=Deputado&page=1',
           'http://www.votenaweb.com.br/politicos?apenas=Presidente&page=1'
          ]
    browser = robobrowser.RoboBrowser(history=True, parser='html.parser')
    
    for url in url_all:
    
        browser.open(url)
        politicos = browser.find_all(class_ = 'politician')
    
        for politico in politicos:
            
            link_pol = url_ + str(politico.find('a'))[9:].split('"')[0]
            browser.open(link_pol)
            projects = browser.find_all(class_ = 'bill')
            
            for project in projects:
                
                link_proj = url_ + str(project.find(class_ = 'comments_count'))[32:].split('"')[0]
                browser.open(link_proj)
                comments = browser.find_all(class_ = 'comment comment-depth-0')
                
                for comment in comments:
                    
                    nota_ = comment.find('span')
                    
                    if not nota_:
                        
                        sense.append(0)
                    else:
                        
                        nota = str(nota_).split('<')[1].split('>')[1]
                        mensagem = str(comment.find_all('p')[1]).split('>')[1].split('<')[0]
                        
                        if nota == 'Sim':
                            sense.append(1)
                        
                        if nota == 'Não':
                            sense.append(-1)
                    
                    text.append(mensagem)
    data = []
    
    for i in range(len(text)):
        data.append([text[i], sense[i]])
    data = pd.DataFrame(data, columns = ["Text", "Classificacao"])
    data["Classificacao"] = data["Classificacao"].map({-1: "Negativo", 0: "Neutro", 1: "Positivo",})

    return data


def LoadFile(filename):
    return pd.read_csv(filename,sep='\t',encoding='utf-8')[['Text','Classificacao']]

def MakeLower(text):
    if type(text)==str:
        return text.lower()
    else:
        return text

def CreateWordsInDF(df,col='Text'):
    words_in_df = []
    for text in df['Text']:
        if type(text)==str:
            text = text.split()
            words_in_df += text
    return words_in_df

def ReplaceEmptyString(x):
    if str(x).strip()=="":
        return np.nan
    else:
        return x

def RemovePunctuation(x):
    if type(x) == str:
        # x = re.sub(r'[^\x00-\x7f]',r' ',x)
        x = ' '.join(re.sub("(@[A-Za-z0-9]+)"," ",x).split())
        return re.sub("["+string.punctuation+"]", r" ", x)
    else:
        return x

def RemoveNumbers(x):
    if type(x) == str:
        res = re.sub(r'[0-9]+',r' ',x)
        return res
    else:
        return x

def getIntersectionNames(df, words_in_df, col="Texto"):
    n = set(nomes)
    w = set(words_in_df)
    nomes_intersection = n.intersection(w)
    return nomes_intersection

def RemoveNames(frase, names_in_df):
    if type(frase) == str:
        non_nomes = [i for i in frase.split() if i.lower() not in names_in_df]
        return (" ".join(non_nomes))
    else:
        return frase

def findLessFrequent(counter, Num_Less_Frequent=10):
    return set([w for w in counter if counter[w] < 10])

def RemoveLessFrequent(frase, less_frequent_words):
    if type(frase) == str:
        tokens = set(frase.lower().split())
        inter = tokens.intersection(less_frequent_words)
        non_less = [i for i in frase.split() if i not in inter]
        return (" ".join(non_less))
    else:
        return frase

def RemoveSinonimos(frase, sinonimos):
    if type(frase) == str:
        frase = frase.lower()
        for key in sinonimos.keys():
            for word in sinonimos[key]:
                frase = frase.replace(word, key)
        return frase
    else:
        return frase

def RemoveOtherWords(frase, otherwords):
    if type(frase) == str:
        non_others = [i for i in frase.split() if i not in otherwords]
        return (" ".join(non_others))
    else:
        return frase

def RemoveStopWords(frase, stopwordsset):
    if type(frase) == str:
        non_stop = [i for i in frase.split() if i not in stopwordsset] 
        return (" ".join(non_stop))
    else:
        return frase

def cleanText(df, col='Text'):

    now = time.time()
    df.index = df.index.map(str)
    df.loc[:,'Text'] = df['Text'].map(str)
    df.loc[:,'Text'] = df['Text'].map(MakeLower)
    df.loc[:,'Text'] = df['Text'].map(RemovePunctuation)
    df.loc[:,'Text'] = df['Text'].map(RemoveNumbers)
    df.loc[:,'Text'] = df['Text'].map(ReplaceEmptyString)
    words_in_df = CreateWordsInDF(df)
    names_in_df = getIntersectionNames(df, words_in_df, col="Texto")
    df.loc[:,'Text'] = df['Text'].map(lambda x: RemoveNames(x, names_in_df))
    counter = Counter(words_in_df)
    less_frequent_words = findLessFrequent(counter)
    df.loc[:,'Text'] = df['Text'].map(lambda x: RemoveLessFrequent(x, less_frequent_words))
    df.loc[:,'Text'] = df['Text'].map(lambda x: RemoveSinonimos(x, sinonimos))
    df.loc[:,'Text'] = df['Text'].map(lambda x: RemoveOtherWords(x, otherwords))
    stopwordsset = set(stopwords)
    df.loc[:,'Text'] = df['Text'].map(lambda x: RemoveStopWords(x, stopwordsset))
    # df.loc[:,'Texto'] = df['Texto'].map(StemWords)
    df.loc[:,'Text'] = df['Text'].map(ReplaceEmptyString)

    return df

def GetVectorizing(df, NGRAM=(1,3), n_features=10, n_samples=8000, col='Text'):

    not_nan_indexes = df[col].notnull()
    data_samples = [
        text for text in df.loc[not_nan_indexes,col]
    ]
    tfidf_vectorizer = TfidfVectorizer(
        max_df=0.95,
        min_df=2,
        max_features=n_features,
        stop_words=stopwords,
        ngram_range = NGRAM
    )
    tfidf = tfidf_vectorizer.fit_transform(data_samples)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()

    return  tfidf, tfidf_feature_names

def fazTudo(df):

    n_samples = df.shape[0]
    df['Text'] = df.loc[:,"Text"]
    df_temp = cleanText(df)
    now = time.time()
    tfidf, tfidf_feature_names = GetVectorizing(df_temp)

    return df_temp.dropna()

def get_topics(df):

    tfidf, tfidf_feature_names = GetVectorizing(df)

    # get the model
    now = time.time()
    model = NMF(
        n_components=3,
        random_state=1,
        beta_loss='kullback-leibler',
        solver='mu',
        max_iter=1000,
        alpha=.1,
        l1_ratio=.5
    ).fit(tfidf)

    # get topics
    n_top_words = 4
    topics = []

    for topic_idx, topic in enumerate(model.components_):
        message = " ".join([tfidf_feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        topics.append(message)

    return topics

def create_model(df):

    clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf-svm', SGDClassifier(
            loss='hinge',
            penalty='l2',
            alpha=1e-3,
            max_iter=5,
            random_state=42))])
    clf.fit(df.Text, df.Classificacao)

    return clf, df

def get_percents(df_train, df_test):

    clf, _ = create_model(df_train)

    # create predictions
    preds = pd.DataFrame({
        'Text': df_test.Text,
        'Prediction': clf.predict(df_test.Text)
    })

    neutro = preds[preds['Prediction']=='Neutro']
    positivo = preds[preds['Prediction']=='Positivo']
    negativo = preds[preds['Prediction']=='Negativo']
    topicos_positivos = get_topics(positivo)
    topicos_negativos = get_topics(negativo)

    # data to donut plot
    total = df_test.shape[0]
    neutro_perc = round(100*neutro.shape[0]/total,2)
    positivo_perc = round(100*positivo.shape[0]/total,2)
    negativo_perc = round(100*negativo.shape[0]/total,2)

    # get topics
    topicos_positivos = get_topics(positivo)
    topicos_negativos = get_topics(negativo)

    # accuracy
    df_test = df_train.sample(frac=0.8) # here change if df_test != df_train
    accuracy = round(clf.score(df_test.Text,df_test.Classificacao)*100,2)

    return neutro_perc, positivo_perc, negativo_perc, total, accuracy, topicos_positivos, topicos_negativos

def tweetsReport(filename):

    if filename == "politicos.csv":
        df_train = fazTudo(get_politics())
    else:
        df_train = fazTudo(LoadFile(filename))

    neutro_perc, positivo_perc, negativo_perc, total, accuracy, topicos_positivos, topicos_negativos = get_percents(df_train, df_train)

    return {
        "name": filename.split('.')[0],
        "total": total,
        "positive": {
            "percent": positivo_perc,
            "tags": topicos_positivos[:3]
        },
        "negative": {
            "percent": negativo_perc,
            "tags": topicos_negativos[:3]
        },
        "neutro": {
            "percent": neutro_perc
        },
        "accuracy": accuracy
    }

