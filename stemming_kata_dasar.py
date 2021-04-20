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



mark_url = requests.get('https://www.cnnindonesia.com/olahraga')
soup = BeautifulSoup(mark_url.content,'html.parser')



link_lst = []
for eachitem in soup.find_all('a', href=True):   
    link_lst.append(eachitem.get("href"))
    
pdLink = pd.DataFrame(link_lst)

pdLink = pdLink.loc[pdLink[0].str.contains("olahraga/")]

mark_url = requests.get('https://www.cnnindonesia.com/nasional/20210319174728-20-619749/5-alasan-mui-izinkan-vaksin-astrazeneca-disuntik-ke-warga-ri')
soup = BeautifulSoup(mark_url.content,'html.parser')

results = soup.find(id='detikdetailtext')


print(soup.prettify())
###############################################################################
##REMOVE LINK, TANDA BACA, SPESIAL KARAKTER, TABEL, DAN IMAGES DARI ARTIKEL
###############################################################################
h = html2text.HTML2Text()#ignoring all the links, tables and images in the blog
h.ignore_links = True
h.ignore_images = True
h.ignore_tables = True

main_content = h.handle(str(results.prettify())) 
main_content = main_content.translate(str.maketrans('', '', string.punctuation))
main_content = main_content.replace("‘", '').replace("’", '').replace("'", '')

################################################################################
##################################### UBAH DALAM BENTUK HURUF KECIL
################################################################################
main_content = main_content.lower().strip("b").strip()
################################################################################


################################################################################
############### STEMMING DALAM BAHASA INDONESIA ################################
############### MENGGUNAKAN SASTRAWI ###########################################
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()

katadasar = stemmer.stem(main_content)
################################################################################

################################################################################
####################### TOKENIZATION
word_tokens = word_tokenize(katadasar)
################################################################################


################################################################################
##################################### REMOVE STOP WORDS
stop_words = stopwords.words('indonesian')
main_content_tokens = [w for w in word_tokens if not w in stop_words]

################################################################################
################################################################################
########################## VISUALKAN KATA YANG SERING MUNCUL ###################
################################################################################
semua_content = " " . join(main_content_tokens)


stopwords = set(STOPWORDS)
wordcloud = WordCloud(width = 800, height = 800, max_font_size=50,min_font_size = 10,
                stopwords = stopwords, background_color = "black", colormap="plasma").generate(semua_content) 
  
# plot the WordCloud image     
plt.figure(figsize = (12, 12)) 
plt.imshow(wordcloud, interpolation="bilinear") 
plt.axis("off") 
plt.tight_layout()# store to file
plt.savefig("av_wordcloud.png", dpi=150)  
plt.show()


##################################################################################
##################################################################################
################### PILIH 10 RANGKING TERTINGGI KATA YANG SERING MUNCUL
##################################################################################
freq = pd.Series(main_content_tokens).value_counts()[:10]
main_content_high_rank =[word for word in main_content_tokens if word in list(freq.index)]
cleaned_content = " " . join(main_content_high_rank)


##################################################################################
############################## VISUALKAN KATA
##################################################################################
stopwords = set(STOPWORDS)
wordcloud = WordCloud(width = 800, height = 800, max_font_size=50,min_font_size = 10,
                stopwords = stopwords, background_color = "black", colormap="plasma").generate(cleaned_content) 
  
# plot the WordCloud image     
plt.figure(figsize = (12, 12)) 
plt.imshow(wordcloud, interpolation="bilinear") 
plt.axis("off") 
plt.tight_layout()# store to file
plt.savefig("av_wordcloud.png", dpi=150)
  
plt.show()

###################################################################################
###################################################################################
#################################TAMPILKAN DALAM GRAFIK BAR RANGKING 10 TERTINGGI
counted_words = Counter(main_content_tokens)
most_common_df = pd.DataFrame(counted_words.most_common(10), columns=["words", "count"])#plot the most common words
sns.barplot(y = "words", x = "count", data = most_common_df, palette="viridis")
plt.xlabel("Frequency")
plt.ylabel("Words")
plt.title("Top 10 Most Occuring Words in the Corpus")
plt.show()
