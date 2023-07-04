# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 22:56:04 2023

@author: serkan
"""

#1.MODEL
#Gerekli kütüphaneler
import keras
import matplotlib.pyplot as plt
#Veri setini yükle
fashion_mnist=keras.datasets.fashion_mnist

(train_images,train_labels),(test_images,test_labels)=fashion_mnist.load_data()

#Eğtim veri setini eğitim(%80) ve doğrulama(%20) olarak böl.
from sklearn.model_selection import train_test_split

train_images,validation_images,train_labels,validation_labels=train_test_split(train_images,train_labels,train_size=0.8)

#İlk 9 örneği gösterme
for i in range(9):
    plt.subplot(330+1+i)
    plt.imshow(validation_images[i],cmap=plt.get_cmap('gray'))
plt.show()

class_name=['Tisort','Pantalon','Kazak','Elbise','Ceket','Sandalet','Gömlek','Spor Ayakkabı','Canta','Cizme']

#Model Oluşturma

from keras.models import Sequential
from keras.layers import Dense, Conv2D,Flatten

model=Sequential()

model.add(Conv2D(32,kernel_size=4,activation='relu',input_shape=(28,28,1)))   
model.add(Conv2D(64,kernel_size=4,activation='relu'))
model.add(Conv2D(128,kernel_size=4,activation='relu'))
model.add(Flatten())
model.add(Dense(10,activation='softmax'))
#10 adet sınıfımız oluğu için    
#filtre sayısı olarak 32,64,128 ve 4*4 filtreler uygulandı modeli iyileştirmek için diğer modellerde değişim sağlandı           
model.summary()
#çoklu sınıflandırma için aktivasyon fonksiyonu olarak softmax kullanılır
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#kayıp fonksiyonu için çoklu sınıflandırmada oluğundan dolayı sparse_categorical_crossentropy kullandık

#Veriyi modelimizin istediği girdiye uydurduk.
train_images=train_images.reshape(-1,28,28,1)
validation_images=validation_images.reshape(-1,28,28,1)
test_images=test_images.reshape(-1,28,28,1)

#Özellik ölçeklendirme yapmadan eğitme
model.fit(train_images,train_labels,epochs=2,validation_data=(validation_images,validation_labels))

#Özellik ölçeklendirme
train_images=train_images/255.0
validation_images=validation_images/255.0
test_images=test_images/255.0

#Özellikölçeklendirme sonrası eğitim
model.fit(train_images,train_labels,epochs=5,validation_data=(validation_images,validation_labels))

#Modeli değerlendirme(test etme)
test_loss,test_acc=model.evaluate(test_images,test_labels)

#------------------------------------------------
#2.MODEL
#Gerekli kütüphaneler
import keras
import matplotlib.pyplot as plt
from keras.layers import MaxPooling2D
#Veri setini yükle
fashion_mnist=keras.datasets.fashion_mnist

(train_images,train_labels),(test_images,test_labels)=fashion_mnist.load_data()

#Eğtim veri setini eğitim(%80) ve doğrulama(%20) olarak böl.
from sklearn.model_selection import train_test_split

train_images,validation_images,train_labels,validation_labels=train_test_split(train_images,train_labels,train_size=0.8)

#İlk 9 örneği gösterme
for i in range(9):
    plt.subplot(330+1+i)
    plt.imshow(validation_images[i],cmap=plt.get_cmap('gray'))
plt.show()

class_name=['Tisort','Pantalon','Kazak','Elbise','Ceket','Sandalet','Gömlek','Spor Ayakkabı','Canta','Cizme']

#Model Oluşturma

from keras.models import Sequential
from keras.layers import Dense, Conv2D,Flatten

model=Sequential()

model.add(Conv2D(64,kernel_size=3,activation='relu',input_shape=(28,28,1))) 
  
model.add(Conv2D(128,kernel_size=3,activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(256,kernel_size=3,activation='relu'))
#burda maxpooling ile ortaya çıkan özelliklerin boyut düşürümü ile modelin özellikleri daha iyi öğrenmesi sağlandı 
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10,activation='softmax'))
#burda dense katmanında 256 adet nöron kullanıldı                
model.summary()

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#Veriyi modelimizin istediği girdiye uydurduk.
train_images=train_images.reshape(-1,28,28,1)
validation_images=validation_images.reshape(-1,28,28,1)
test_images=test_images.reshape(-1,28,28,1)

#Özellik ölçeklendirme yapmadan eğitme
model.fit(train_images,train_labels,epochs=2,validation_data=(validation_images,validation_labels))

#Özellik ölçeklendirme
train_images=train_images/255.0
validation_images=validation_images/255.0
test_images=test_images/255.0

#Özellikölçeklendirme sonrası eğitim
model.fit(train_images,train_labels,epochs=5,validation_data=(validation_images,validation_labels))

#Modeli değerlendirme(test etme)
test_loss,test_acc=model.evaluate(test_images,test_labels)
#-----------------------------------------------
#3.MODEL
#Gerekli kütüphaneler
import keras
import matplotlib.pyplot as plt
from keras.layers import MaxPooling2D
#Veri setini yükle
fashion_mnist=keras.datasets.fashion_mnist

(train_images,train_labels),(test_images,test_labels)=fashion_mnist.load_data()

#Eğtim veri setini eğitim(%80) ve doğrulama(%20) olarak böl.
from sklearn.model_selection import train_test_split

train_images,validation_images,train_labels,validation_labels=train_test_split(train_images,train_labels,train_size=0.8)

#İlk 9 örneği gösterme
for i in range(9):
    plt.subplot(330+1+i)
    plt.imshow(validation_images[i],cmap=plt.get_cmap('gray'))
plt.show()

class_name=['Tisort','Pantalon','Kazak','Elbise','Ceket','Sandalet','Gömlek','Spor Ayakkabı','Canta','Cizme']

#Model Oluşturma

from keras.models import Sequential
from keras.layers import Dense, Conv2D,Flatten

model=Sequential()

model.add(Conv2D(32,kernel_size=3,activation='relu',input_shape=(28,28,1)))   
model.add(Conv2D(64,kernel_size=3,activation='relu'))
model.add(Conv2D(128,kernel_size=3,activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(10,activation='softmax'))
#10 adet sınıfımız oluğu için               
#dense katmanı özellik haritasını düzleştirir. burda 512 adet nöron kullanıldı
model.summary()
#çoklu sınıflandırma için aktivasyon fonksiyonu olarak softmax kullanılır
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#kayıp fonksiyonu için çoklu sınıflandırmada oluğundan dolayı sparse_categorical_crossentropy kullandık

#Veriyi modelimizin istediği girdiye uydurduk.
train_images=train_images.reshape(-1,28,28,1)
validation_images=validation_images.reshape(-1,28,28,1)
test_images=test_images.reshape(-1,28,28,1)

#Özellik ölçeklendirme yapmadan eğitme
model.fit(train_images,train_labels,epochs=2,validation_data=(validation_images,validation_labels))

#Özellik ölçeklendirme
train_images=train_images/255.0
validation_images=validation_images/255.0
test_images=test_images/255.0

#Özellikölçeklendirme sonrası eğitim
model.fit(train_images,train_labels,epochs=5,validation_data=(validation_images,validation_labels))

#Modeli değerlendirme(test etme)
test_loss,test_acc=model.evaluate(test_images,test_labels)