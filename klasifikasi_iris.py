############################################################################################################
####################################################### KLASIFIKASI MENGGUNAKAN KNN
############################################################################################################
import numpy as np
import pandas as pd


df_iris = pd.read_csv("C:/Users/acer/Documents/latihan_iris/iris_dataset/iris.csv")

df_iris.groupby('Species').size()

feature_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm','PetalWidthCm']
X = df_iris[feature_columns].values
y = df_iris['Species'].values



from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# Instantiate learning model (k = 3)
classifier = KNeighborsClassifier(n_neighbors=3)

# Fitting the model
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)



cm = confusion_matrix(y_test, y_pred)
hasil_klasifikasi = classification_report(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)*100


daftar_k = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

daftar_k = list(range(1,20,2))



hasil_prediksi= []
for k in daftar_k:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)*100
    hasil_prediksi.append(accuracy)


import matplotlib.pyplot as plt

plt.figure()
plt.figure(figsize=(15,10))
plt.title('Mencari k terbaik', fontsize=20, fontweight='bold')
plt.xlabel('Jumlah K', fontsize=15)
plt.ylabel('Akurasi', fontsize=15)

plt.plot(daftar_k, hasil_prediksi)

plt.show()

#######################################################################################################
#######################################################################################################
############################################ KLASIFIKASI MENGGUNAKAN BAYES

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

df_iris = pd.read_csv("C:/Users/acer/Documents/latihan_iris/iris_dataset/iris.csv")


df_iris.groupby('Species').size()

feature_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm','PetalWidthCm']
X = df_iris[feature_columns].values
y = df_iris['Species'].values




le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)



cm = confusion_matrix(y_test, y_pred)
hasil_klasifikasi = classification_report(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)*100

