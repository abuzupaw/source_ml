from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
from six import StringIO 
from IPython.display import Image 
from pydot import graph_from_dot_data
from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

datapegawaipake = pd.read_csv('C://Users/acer/Documents/data_pegawai_pake_baru.csv',delimiter=',')

datapegawaipake['SEX'] = datapegawaipake.GENDER.astype('category').cat.codes

df = datapegawaipake[['SEX', 'DURASI', 'SENIOR', 'LEVEL_JOB', 'DEPT', 'RESIGN']] 




from sklearn.feature_selection import chi2

X = df.drop('RESIGN',axis=1)
y = df['RESIGN']

chi_scores = chi2(X,y)


nilai_chi = pd.Series(chi_scores[0],index = X.columns)
p_values = pd.Series(chi_scores[1],index = X.columns)
#p_values.sort_values(ascending = False , inplace = True)

nilai_chi.plot.bar()
p_values.plot.bar()

##################################################################################################
##############################PREDIKSI TANPA SELEKSI FITUR########################################
##################################################################################################
df = datapegawaipake[['SEX', 'DURASI', 'SENIOR', 'LEVEL_JOB', 'DEPT', 'RESIGN']] 
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,0:5], df.iloc[:,5], test_size=0.2)


#criterion="entropy", max_depth=3
#dt = DecisionTreeClassifier()

dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(X_train, y_train)

# dot_data = StringIO()
# export_graphviz(dt, out_file=dot_data, feature_names=datapegawaipake.iloc[:,2:5].columns)
# (graph, ) = graph_from_dot_data(dot_data.getvalue())
# Image(graph.create_png())

y_pred = dt.predict(X_test)


print("Accuracy Decision Tree:",metrics.accuracy_score(y_test, y_pred))



target_names = ['ACTIVE', 'RESIGN']

print(classification_report(y_test, y_pred, target_names=target_names))


hasil_prediksi = confusion_matrix(y_test, y_pred)
import seaborn as sns
sns.heatmap(hasil_prediksi, annot=True)





##########################################################################
#################### KLASIFIKASI MENGGUNAKAN NAIVE BAYES
##########################################################################
##########################################################################

#Create a Gaussian Classifier
gnb = GaussianNB()

#Train the model using the training sets
gnb.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = gnb.predict(X_test)

from sklearn import metrics

from sklearn.metrics import classification_report

hasil_prediksi = confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred, target_names=target_names))

# Model Accuracy, how often is the classifier correct?
print("Accuracy Bayes:",metrics.accuracy_score(y_test, y_pred))

import seaborn as sns
sns.heatmap(hasil_prediksi, annot=True)





##########################################################################
#################### KLASIFIKASI MENGGUNAKAN RANDOM FOREST
##########################################################################
##########################################################################
from sklearn.ensemble import RandomForestClassifier
# Instantiate model with 1000 decision trees
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

from sklearn import metrics

from sklearn.metrics import classification_report

print("Accuracy Random Forest:",metrics.accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=target_names))


hasil_prediksi = confusion_matrix(y_test, y_pred)
import seaborn as sns
sns.heatmap(hasil_prediksi, annot=True)









###########################################################################
######################## KLASIFIKASI SETELAH SELEKSI FITUR
###########################################################################
df = datapegawaipake[['DURASI', 'SENIOR', 'DEPT', 'RESIGN']] 
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,0:3], df.iloc[:,3], test_size=0.2)

dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(X_train, y_train)

# dot_data = StringIO()
# export_graphviz(dt, out_file=dot_data, feature_names=datapegawaipake.iloc[:,2:5].columns)
# (graph, ) = graph_from_dot_data(dot_data.getvalue())
# Image(graph.create_png())

y_pred = dt.predict(X_test)


print("Accuracy Decision Tree:",metrics.accuracy_score(y_test, y_pred))



target_names = ['ACTIVE', 'RESIGN']
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred, target_names=target_names))


hasil_prediksi = confusion_matrix(y_test, y_pred)
import seaborn as sns
sns.heatmap(hasil_prediksi, annot=True)





##########################################################################
#################### KLASIFIKASI MENGGUNAKAN NAIVE BAYES
##########################################################################
##########################################################################
from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier
gnb = GaussianNB()

#Train the model using the training sets
gnb.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = gnb.predict(X_test)

from sklearn import metrics

from sklearn.metrics import classification_report


# Model Accuracy, how often is the classifier correct?
print("Accuracy Bayes:",metrics.accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=target_names))



hasil_prediksi = confusion_matrix(y_test, y_pred)
import seaborn as sns
sns.heatmap(hasil_prediksi, annot=True)

##########################################################################
#################### KLASIFIKASI MENGGUNAKAN RANDOM FOREST
##########################################################################
##########################################################################
from sklearn.ensemble import RandomForestClassifier
# Instantiate model with 1000 decision trees
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

from sklearn import metrics

from sklearn.metrics import classification_report


# Model Accuracy, how often is the classifier correct?
print("Accuracy Random Forest:",metrics.accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=target_names))



hasil_prediksi = confusion_matrix(y_test, y_pred)
import seaborn as sns
sns.heatmap(hasil_prediksi, annot=True)