import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mlxtend.plotting import plot_decision_regions
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn import metrics
from imblearn.under_sampling import NearMiss
from trans import PCA, transform
from plot import knn_regions, svm_regions, plotScatter
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import roc_curve, auc

def cleanData(data):
  data = data.replace(['Yes'],1)
  data = data.replace(['Yes (during pregnancy)'],0.8)
  data = data.replace(['No'],0)
  data = data.replace(['Male'],1)
  data = data.replace(['Female'],0)
  data = data.replace(['Excellent'],1)
  data = data.replace(['Very good'],0.75)
  data = data.replace(['Good'],0.5)
  data = data.replace(['Fair'],0.25)
  data = data.replace(['Poor'],0)
  data = data.replace(['No, borderline diabetes'],0.7)
  data = data.replace(['18-24'],21)
  data = data.replace(['25-29'],27)
  data = data.replace(['30-34'],32)
  data = data.replace(['35-39'],37)
  data = data.replace(['40-44'],42)
  data = data.replace(['45-49'],47)
  data = data.replace(['50-54'],52)
  data = data.replace(['55-59'],57)
  data = data.replace(['60-64'],62)
  data = data.replace(['65-69'],67)
  data = data.replace(['70-74'],72)
  data = data.replace(['75-79'],77)
  data = data.replace(['80 or older'],82)
  data=(data-data.min())/(data.max()-data.min())
  return data

def main():
  predict= "HeartDisease"
  data=pd.read_csv("/Users/mariajosebernal/Documents/EAFIT/2022-2/Análisis numérico/Proyecto/Código/heart_2020_cleaned.csv")
  #data=data.sample(50000,random_state=0)
  data=data.drop(columns=["Race","AlcoholDrinking"])
  data=cleanData(data)
  y=data[predict]
  # y.value_counts().plot(kind='bar')
  # plt.show()
  X=data.drop(columns=[predict])
  #X=transform(X)
  #print(np.cov(np.transpose(X)))
  X=PCA(X, 2)
  X=pd.DataFrame(X)
  
  n=81
  
  x_train, x_test, y_train, y_test= train_test_split(X, y, test_size= 0.1, random_state=0)
  rus = RandomUnderSampler(random_state=42)

  x_train,y_train= rus.fit_resample(x_train,y_train)  
  
  classifier= KNeighborsClassifier(n_neighbors=n) 
  classifier.fit(x_train.values, y_train.values)
  y_pred= classifier.predict_proba(x_test)
  y_pred2=classifier.predict(x_test)
  
  print("k: " + str(n))
  print("Resultados")
  print("Matriz de confusión")
  print(confusion_matrix(y_test, y_pred2))
  print("AUC:", metrics.roc_auc_score(y_test, y_pred[:,1]))

  fpr, tpr, threshold = roc_curve(y_test, y_pred[:, 1])
  roc_auc = auc(fpr, tpr)
  plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
  plt.legend(loc = 'lower right')
  plt.plot([0, 1], [0, 1],'r--')
  plt.xlim([0, 1])
  plt.ylim([0, 1])
  plt.ylabel('Tasa de verdaderos positivos')
  plt.xlabel('Tasa de falsos positivos')
  plt.title('Curva ROC ')
  plt.show()

  knn_regions(x_test.head(100), y_test.head(100),classifier)
  
  
main()