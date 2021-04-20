from nltk.tokenize import word_tokenize
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt 
from collections import Counter
import seaborn as sns
from nltk.corpus import stopwords
import re
import string
import requests

import pandas as pd
import numpy as np

import requests
from bs4 import BeautifulSoup

import html2text

import nltk
#nltk.download('punkt')
#nltk.download('stopwords')

from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

#mark_url = requests.get('https://www.cnnindonesia.com/nasional/20210319174728-20-619749/5-alasan-mui-izinkan-vaksin-astrazeneca-disuntik-ke-warga-ri')

##############################################################################
##############################################################################
###### bidang ekonomi  
mark_url = requests.get('https://www.cnnindonesia.com/ekonomi')
soup = BeautifulSoup(mark_url.content,'html.parser')



link_lst = []
for eachitem in soup.find_all('a', href=True):   
    link_lst.append(eachitem.get("href"))
    
pdLinkEkonomi = pd.DataFrame(link_lst)

pdLinkEkonomi = pdLinkEkonomi.loc[pdLinkEkonomi[0].str.contains("ekonomi/2021")][:10] 

pdLinkEkonomi = pdLinkEkonomi.reset_index(drop=True)
pdLinkEkonomi["label"] = 0
#################################################################
################################################################
######################## BIDANG TEKNOLOGI
mark_url = requests.get('https://www.cnnindonesia.com/teknologi')
soup = BeautifulSoup(mark_url.content,'html.parser')


link_lst = []
for eachitem in soup.find_all('a', href=True):   
    link_lst.append(eachitem.get("href"))
    
pdLinkTeknologi = pd.DataFrame(link_lst)

pdLinkTeknologi = pdLinkTeknologi.loc[pdLinkTeknologi[0].str.contains("teknologi/2021")][:10] 

pdLinkTeknologi = pdLinkTeknologi.reset_index(drop=True)
pdLinkTeknologi["label"] = 1

################################################################
######################## BIDANG OLAHRAGA
mark_url = requests.get('https://www.cnnindonesia.com/olahraga')
soup = BeautifulSoup(mark_url.content,'html.parser')


link_lst = []
for eachitem in soup.find_all('a', href=True):   
    link_lst.append(eachitem.get("href"))
    
pdLinkOlahraga = pd.DataFrame(link_lst)

pdLinkOlahraga = pdLinkOlahraga.loc[pdLinkOlahraga[0].str.contains("olahraga/2021")][:10] 

pdLinkOlahraga = pdLinkOlahraga.reset_index(drop=True)
pdLinkOlahraga["label"] = 2

# pdLink = pdLink.sort_index(ascending=False)

pdLink = pdLinkEkonomi.append(pdLinkTeknologi)
pdLink = pdLink.append(pdLinkOlahraga)

pdLink = pdLink.reset_index(drop=True)

#myFitur = pd.DataFrame(columns=['kontras','homogen','energy','korelasi'])
myListDF = pd.DataFrame(columns=['isi','label'])

for i in range (0, len(pdLink)):
    mark_url = requests.get(pdLink.iloc[i,0])
    soup = BeautifulSoup(mark_url.content,'html.parser')
    results = soup.find(id='detikdetailtext')
    h = html2text.HTML2Text()#ignoring all the links, tables and images in the blog
    h.ignore_links = True
    h.ignore_images = True
    h.ignore_tables = True
    main_content =""
    if hasattr(results, 'prettify'):
        main_content = h.handle(str(results.prettify()))
    else:
        pass
    main_content = main_content.translate(str.maketrans('', '', string.punctuation))
    main_content = main_content.replace("‘", '').replace("’", '').replace("'", '')
    main_content = main_content.lower().strip("b").strip()
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    katadasar = stemmer.stem(main_content)
    word_tokens = word_tokenize(katadasar)
    stop_words = stopwords.words('indonesian')
    main_content_tokens = [w for w in word_tokens if not w in stop_words]
    main_content_tokens = [word for word in main_content_tokens if word.isalpha()]
    semua_content = " " . join(main_content_tokens)
    myListDF.loc[i,'isi'] = semua_content
    myListDF.loc[i,'label'] = pdLink.iloc[i,1]


##################################################################################################
##################################################################################################
#################### CREATE DOCUMENT TERM MATRIX #################################################
##################################################################################################
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
X = vec.fit_transform(myListDF)
df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())


#################################################################################################
##################### CREATE DTM TERM FREQUENCY~INVERSE DOCUMENT FREQUENCY TF-IDF ###############
#################################################################################################
max_features =1000

dfListClean = myListDF.loc[myListDF["isi"] != '']

dfListClean = dfListClean.reset_index(drop=True)

dfCluster = dfListClean.T.copy()
#ngram_range = (1,1) one gram
#ngram_range = (2,2) dua gram
#ngram_range = (3,3) tiga gram

vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1,3), smooth_idf=False)
doc_vec = vectorizer.fit_transform(dfCluster.iloc[0])

dtm_df_idf = pd.DataFrame(doc_vec.toarray().transpose(),index=vectorizer.get_feature_names())

dtm_df_idf = dtm_df_idf.T
##############################################################################################
#################################### CLUSTERING MENGGUNAKAN K-MEANS
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

Nc = range(1, 10)
kmeans = [KMeans(n_clusters=i) for i in Nc]
#kmeans
score = [kmeans[i].fit(dtm_df_idf).score(dtm_df_idf) for i in range(len(kmeans))]

plt.plot(Nc,score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()
####################################################################################################

#KARENA HANYA TIGA BERITA MAKA SAYA HANYA BAGI MENJADI 3 CLUSTER
modelKmean = KMeans(n_clusters=3).fit(dtm_df_idf)

hasilCluster = dfListClean.copy()
hasilCluster = hasilCluster.reset_index(drop=True)
hasilCluster['hasilKmeanOneGram'] = pd.DataFrame(modelKmean.labels_)
#ngram_range = (2,2) dua gram
hasilCluster['hasilKmeanTwoGram'] = pd.DataFrame(modelKmean.labels_)
#ngram_range = (3,3) dua gram

hasilCluster['hasilKmeanThreeGram'] = pd.DataFrame(modelKmean.labels_)
