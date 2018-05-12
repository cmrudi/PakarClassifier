from __future__ import division
import pickle
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import average_precision_score
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


#Initialize stop words for Bahasa Indonesia
id_stop_words_file = pd.read_csv('id-stopwords.txt', header=None, names=['stopwords'])
id_stop_words_list = []
for i in range(len(id_stop_words_file)):
    id_stop_words_list.append(id_stop_words_file.values[i][0])

#Initialize Dataset
dataset = pd.read_csv('contoh-data.csv', header=0, sep=',')
input = dataset['judul_proposal'].str.lower().str.replace('[^a-zA-Z0-9 ]', '')
target = dataset['deskripsi'].str.lower()

#Initialize Stemmer, Stemmernya cukup lama jadi comment aja biar cepet
#factory = StemmerFactory()
#stemmer = factory.create_stemmer()
#for i in range(len(input)):
#    print i
#    input[i] = stemmer.stem(input[i])

#Initialize TFIDF Vectorizer
tvect = TfidfVectorizer(min_df=1,stop_words=id_stop_words_list)

#Split Test dan Data Train
x_train, x_test, y_train, y_test = train_test_split(input, target, test_size=0.2, random_state=4)

tvect1 = TfidfVectorizer(min_df=1,stop_words=id_stop_words_list)
x_traincv=tvect1.fit_transform(x_train)
x_testcv=tvect1.transform(x_test)
x_traincv=tvect1.fit_transform(x_train)
x_testcv=tvect1.transform(x_test)


filename = 'KlasifikasiPakar.sav'
loaded_model = pickle.load(open(filename, 'rb'))

y_train = y_train.astype('str')
loaded_model.fit(x_traincv,y_train)

print "Cross Validation"
predictions=loaded_model.predict(x_testcv)
count = 0
for i in range (len(predictions)):
  if predictions[i]==y_test.values[i]:
     count = count + 1
print count / len(y_test)


print "Same Data"
predictions=loaded_model.predict(x_traincv)
count = 0
for i in range (len(predictions)):
  if predictions[i]==y_train.values[i]:
     count = count + 1
print count / len(y_train)
