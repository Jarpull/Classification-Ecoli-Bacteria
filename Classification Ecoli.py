# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.tree import export_text
from sklearn import metrics

direktori = "C:/Users/Lenovo/Documents/Penting/Kuliah/Data Mining/ecoli.csv"

#membaca dataset
data = pd.read_csv(direktori)

# menghilangkan fitur Sequence name
df = pd.DataFrame(data)
df_clean = data.drop('Sequence name',axis=1)

# Pemilihan atribut dataset
fitur = ['mcg','gvh','lip','chg','aac','alm1','alm2']
dataset = df_clean.values
x_data = dataset[:,:7]
y_data = dataset[:,7] 

# Pembagian data training dan data testing
x_train, x_test, y_train, y_test = train_test_split (x_data, y_data, test_size = 0.2, random_state = 0)

# Pemodelan algoritma K-Nearest Neighbour
neigh = KNeighborsClassifier(n_neighbors=5, weights='uniform')
neigh.fit(x_train,y_train)

print('Hasil klasifikasi menggunakan K-NN : ')
print(neigh.predict(x_test))
print()

print('Hasil klasifikasi data yang benar : ')
print(y_test)
print()

print('Nilai Probabilitas tiap kategori(class) : ')
print(neigh.predict_proba(x_test))
print()

y_pred = neigh.predict(x_test)

print('\nKLASIFIKASI DENGAN DECISION TREE\n')

# Pemodelan Algoritma Decision Tree
clf = DecisionTreeClassifier()
clf = clf.fit(x_train,y_train)

y_predict = clf.predict(x_test)

# Visualisasi hasil Decision Tree
tree.plot_tree(clf.fit(x_train,y_train))
r = export_text(clf, feature_names=fitur)
print(r)

# Melihat hasil akurasi dari hasil pemodelan
error = ((y_test != y_pred).sum()/len(y_pred))*100
print('error prediksi KNN = %.2f' %error, '%')

akurasi = 100-error
print("akurasi KNN = %.2f" %akurasi, "%")
print('\n')

y_predict = clf.predict(x_test)
print('prediksi untuk X Decision Tree = ',y_predict)

accDCT = metrics.accuracy_score(y_test, y_predict)*100
print("Accuracy Decision Tree : %.2f" %accDCT,'%')
