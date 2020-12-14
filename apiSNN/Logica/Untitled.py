#!/usr/bin/env python
# coding: utf-8

# # REDES NEURONALES CONVOLUCIONALES (CNN) 

# In[56]:


import tarfile
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
import matplotlib.pyplot as plt
import cv2
print('importadas')


# In[57]:


import os


# In[58]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
cont = 0
for dirname, _, filenames in os.walk('input/'):
    for filename in filenames:
        os.path.join(dirname, filename)
        cont+=1
print(cont)


# In[99]:


dirpath = 'input/'
pixel = 100


# In[60]:


aretes = os.listdir(dirpath+'aretes')
print("Tamaño del subdirectorio aretes:",len(aretes))
backpack = os.listdir(dirpath+'backpack')
print("Tamaño del subdirectorio backpack:",len(backpack))
bufanda = os.listdir(dirpath+'bufanda')
print("Tamaño del subdirectorio bufanda:",len(bufanda))
camisetas = os.listdir(dirpath+'camisetas')
print("Tamaño del subdirectorio camisetas:",len(camisetas))
camisas = os.listdir(dirpath+'camisas')
print("Tamaño del subdirectorio camisas:",len(camisas))
cartera = os.listdir(dirpath+'cartera')
print("Tamaño del subdirectorio cartera:",len(cartera))
cinturon = os.listdir(dirpath+'cinturon')
print("Tamaño del subdirectorio cinturon:",len(cinturon))

corbata = os.listdir(dirpath+'corbata')
print("Tamaño del subdirectorio corbata:",len(corbata))
gafas = os.listdir(dirpath+'gafas')
print("Tamaño del subdirectorio gafas:",len(gafas))
gorro = os.listdir(dirpath+'gorro')
print("Tamaño del subdirectorio gorro:",len(gorro))
medias = os.listdir(dirpath+'medias')
print("Tamaño del subdirectorio medias:",len(medias))
pantalonCorto = os.listdir(dirpath+'pantalonCorto')
print("Tamaño del subdirectorio pantalonCorto:",len(pantalonCorto))
pantalonLargo = os.listdir(dirpath+'pantalonLargo')
print("Tamaño del subdirectorio pantalonLargo:",len(pantalonLargo))
reloj = os.listdir(dirpath+'reloj')
print("Tamaño del subdirectorio reloj:",len(reloj))

ropaInterior = os.listdir(dirpath+'ropaInterior')
print("Tamaño del subdirectorio ropaInterior:",len(ropaInterior))
roptaInteriorMujerC = os.listdir(dirpath+'roptaInteriorMujerC')
print("Tamaño del subdirectorio roptaInteriorMujerC:",len(roptaInteriorMujerC))
sandalias = os.listdir(dirpath+'sandalias')
print("Tamaño del subdirectorio sandalias:",len(sandalias))
sosten = os.listdir(dirpath+'sosten')
print("Tamaño del subdirectorio sosten:",len(sosten))


tacos = os.listdir(dirpath+'tacos')
print("Tamaño del subdirectorio tacos:",len(tacos))
zapatos = os.listdir(dirpath+'zapatos')
print("Tamaño del subdirectorio zapatos:",len(zapatos))


# In[61]:


import pandas as pd
dfCarga = pd.read_csv('styles.csv', low_memory=False, encoding= 'unicode_escape', sep=';', error_bad_lines=False)
dfCarga['image'] = dfCarga.apply(lambda row: str(row['id']) + ".jpg", axis=1)
dfCarga = dfCarga.reset_index(drop=True)
dfCarga


# In[62]:


dfCarga.isnull().sum()


# In[63]:


dfCarga.isin(['!']).sum()
dfCarga.isin(['@']).sum()
dfCarga.isin(['#']).sum()
dfCarga.isin(['$']).sum()
dfCarga.isin(['%']).sum()
dfCarga.isin(['^']).sum()
dfCarga.isin(['&']).sum()
dfCarga.isin(['*']).sum()
dfCarga.isin(['~']).sum()
dfCarga.isin(['|']).sum()
dfCarga.isin(['\'']).sum()


# In[64]:


dfCarga['baseColour'].fillna('desconocido',inplace = True)
dfCarga['season'].fillna('desconocido',inplace = True)
dfCarga['year'].fillna('desconocido',inplace = True)
dfCarga['usage'].fillna('desconocido',inplace = True)
dfCarga['productDisplayName'].fillna('desconocido',inplace = True)
dfCarga.isnull().sum()


# In[67]:


#is_grupo1_sub7 = dfCarga.loc[:, 'image'] == aretes[5]
#df.loc[i] = dfCarga.loc[is_grupo1_sub7]
#df = df.append(dfCarga.loc[is_grupo1_sub7],ignore_index=True)
#dfSub = pd.DataFrame([], columns = dfCarga.columns)
#df = pd.DataFrame([], columns = dfCarga.columns)


# In[65]:


def obtenerInformacionImagenes(directorio):
    df = pd.DataFrame([], columns = dfCarga.columns)
    for i in range(1,len(directorio)):
        #print("i: ",i, " dir: ",directorio[i])
        is_grupo1_sub7 = dfCarga.loc[:, 'image'] == directorio[i]
        #df.loc[i] = dfCarga.loc[is_grupo1_sub7]
        df = df.append(dfCarga.loc[is_grupo1_sub7],ignore_index=True)
    
    return df


# In[66]:


dfSub = pd.DataFrame([], columns = dfCarga.columns)
df = pd.DataFrame([], columns = dfCarga.columns)

df = obtenerInformacionImagenes(aretes)
dfSub = dfSub.append(df)

df =obtenerInformacionImagenes(backpack)
dfSub = dfSub.append(df)

df = obtenerInformacionImagenes(bufanda)
dfSub = dfSub.append(df)

df = obtenerInformacionImagenes(camisetas)
dfSub = dfSub.append(df)

df = obtenerInformacionImagenes(camisas)
dfSub = dfSub.append(df)

df = obtenerInformacionImagenes(cartera)
dfSub = dfSub.append(df)

df = obtenerInformacionImagenes(cinturon)
dfSub = dfSub.append(df)

df = obtenerInformacionImagenes(corbata)
dfSub = dfSub.append(df)

df = obtenerInformacionImagenes(gafas)
dfSub = dfSub.append(df)

df = obtenerInformacionImagenes(gorro)
dfSub = dfSub.append(df)

df = obtenerInformacionImagenes(medias)
dfSub = dfSub.append(df)

df = obtenerInformacionImagenes(pantalonCorto)
dfSub = dfSub.append(df)

df = obtenerInformacionImagenes(pantalonLargo)
dfSub = dfSub.append(df)

df = obtenerInformacionImagenes(reloj)
dfSub = dfSub.append(df)

df = obtenerInformacionImagenes(ropaInterior)
dfSub = dfSub.append(df)

df = obtenerInformacionImagenes(roptaInteriorMujerC)
dfSub = dfSub.append(df)

df = obtenerInformacionImagenes(sandalias)
dfSub = dfSub.append(df)

df = obtenerInformacionImagenes(sosten)
dfSub = dfSub.append(df)

df = obtenerInformacionImagenes(tacos)
dfSub = dfSub.append(df)

df = obtenerInformacionImagenes(zapatos)
dfSub = dfSub.append(df)


# In[67]:


print(dfSub.shape)
dfSub.to_csv('inventario.csv', index=False, sep=';')
dfSub.head(6)


# In[68]:


dfInventario = pd.read_csv('inventario.csv', low_memory=False, encoding= 'unicode_escape', sep=';', error_bad_lines=False)
dfInventario.head(6)


# In[69]:


def display_stats(sample_id, categoria,catImg):
    img_test = cv2.imread(dirpath+categoria+'/'+catImg[sample_id], 1 )
    img_test = cv2.cvtColor(img_test,cv2.COLOR_BGR2RGB)
    
    is_grupo1_sub7 = dfCarga.loc[:, 'image'] == camisetas[sample_id]
    df_grupo1_sub7 = dfCarga.loc[is_grupo1_sub7]
    
    print('Imagen - Valor Min: {} Valor Max: {}'.format(img_test.min(), img_test.max()))
    print('Imagen - Shape: {}'.format(img_test.shape))
    plt.imshow(img_test)
    plt.show()
    print(df_grupo1_sub7)
    


# In[70]:


for dirname, _, filenames in os.walk('input/'):
    print("Categorias Disponibles:", dirname)


# In[72]:


print("Ingrese el ID de la imagen que desea buscar: ")
idImg = int(input())
print("Ingrese la categoria")
categoria = input()
catImg = os.listdir(dirpath+categoria+"/")
#print(catImg)
display_stats(idImg, categoria ,catImg)


# ### Diseñar y optimizar la Arquitectura de la Red Neuronal Convolucional. Mostrar la arquitectura con summary(). 

# In[73]:


def obtenerInformacionImagenes(directorio, listImagenes, identificador):
    for i in tqdm(range(len(listImagenes))):
        path = dirpath+directorio+'/'+listImagenes[i]
        if 'jpg' in path:
            img = cv2.imread(path)
            img = img/255
            img = cv2.resize(img,(pixel,pixel))
            data.append([img,identificador])


# In[74]:


from tqdm import tqdm
data = []
obtenerInformacionImagenes("aretes", aretes,0)
obtenerInformacionImagenes("backpack",backpack,1)
obtenerInformacionImagenes("bufanda", bufanda,2)
obtenerInformacionImagenes("camisas", camisas,3)
obtenerInformacionImagenes("camisetas",camisetas,4)
obtenerInformacionImagenes("cartera", cartera,5)
obtenerInformacionImagenes("cinturon", cinturon,6)
obtenerInformacionImagenes("corbata", corbata,7)
obtenerInformacionImagenes("gafas", gafas,8)
obtenerInformacionImagenes("gorro", gorro,9)
obtenerInformacionImagenes("medias", medias,10)
obtenerInformacionImagenes("pantalonCorto", pantalonCorto,11)
obtenerInformacionImagenes("pantalonLargo", pantalonLargo,12)
obtenerInformacionImagenes("reloj", reloj,13)
obtenerInformacionImagenes("ropaInterior", ropaInterior,14)
obtenerInformacionImagenes("roptaInteriorMujerC", roptaInteriorMujerC,15)
obtenerInformacionImagenes("sandalias",sandalias,16)
obtenerInformacionImagenes("sosten", sosten,17)
obtenerInformacionImagenes("tacos", tacos,18)
obtenerInformacionImagenes("zapatos", zapatos,19)

dataTotal = np.array(data)


# In[75]:



dfDT = pd.DataFrame(dataTotal)
dfDT.to_csv("dataTotalSinIndex.csv", index=False, sep=';')
dfDT.to_csv("dataTotalIndex.csv", index=True, sep=';')


# In[76]:


dataTotal.shape


# In[77]:


for i in tqdm(range(5)):
    np.random.shuffle(dataTotal)


# In[78]:


X = []
Y = []
for i in tqdm(range(dataTotal.shape[0])):
    X.append(dataTotal[i][0])
    Y.append(dataTotal[i][1])

X = np.array(X)
Y = np.array(Y)
Y = np.reshape(Y,(dataTotal.shape[0],1))
    


# In[79]:


X.shape


# In[80]:


Y.shape


# In[81]:


from keras.utils import to_categorical
Y = to_categorical(Y)


# In[82]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)


# In[83]:


print(X_train.shape,'\t',Y_train.shape)
print(X_test.shape,'\t',Y_test.shape)


# In[87]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
#iniciamos el modelo
model = Sequential()
#Añadimos una capa convolucional con 32 filtros de 3píxeles de ancho x 3píxeles de altura. Vamos a aplicarlo a cada imagen. 
#Cada filtro sería de: 3x3x3. 
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape = X_train.shape[1:]))
#añadimos otra capa convolucional con 32 filtros de 3x3
model.add(Conv2D(32, (3, 3), activation='relu'))
#añadimos una capa de pooling de 2x2
model.add(MaxPooling2D(pool_size=(2, 2)))
#este dropout desactiva el 25% de las conexiones entre las neuronas, lo cual mejora los resultados
model.add(Dropout(0.25))

#repetimos todas las capas otra vez
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#repetimos todas las capas otra vez
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#añadimos una capa para aplanar la estructura y convertir en una matriz
model.add(Flatten())
#añadimos una capa con 512 neuronas
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
#en la última capa tiene que hacer tantas neuronas como clases haya, en este caso 10
model.add(Dense(20, activation='softmax'))

#compilamos el modelo
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[88]:


model.summary()


# In[89]:


size_batch = 256
epocas = 20
history = model.fit(X_train, Y_train, batch_size= size_batch, epochs=epocas, verbose=1)


# In[90]:


model.evaluate(X_test,Y_test)


# In[91]:


#Visualización de accuracy
print(history.history.keys())
# "Loss"
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# In[92]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
#plt.plot(history.history['val_accuracy'])
#plt.plot(history.history['val_loss'])
plt.xlabel('Epochs')
plt.ylabel('Values for Accuracy and Loss')
plt.legend(['Training Accuracy','Training Loss','Validation Accuracy','Validation Loss'])


# ## <span style="color:BLUE">ALMACENAMIENTO DE LOS MODELOS DE REDES  NEURONALES</span>

# In[93]:


#FUNCIONES PARA GuARDAR Y CARGAR CUALQUIER MODELO

#Guardar pesos y la arquitectura de la red en un archivo 

def guardarRNN(model,nombreArchivoModelo,nombreArchivoPesos):
    print("Guardando Red Neuronal en Archivo")  
    # serializar modelo a JSON

    # Guardar los Pesos (weights)
    model.save_weights(nombreArchivoPesos+'.h5')

    # Guardar la Arquitectura del modelo
    with open(nombreArchivoModelo+'.json', 'w') as f:
        f.write(model.to_json())

    print("Red Neuronal Grabada en Archivo")   
    


# In[94]:


#Guardar pesos y la arquitectura de la red en un archivo 

nombreArchivoModelo='arquitectura_optimizada'
nombreArchivoPesos='pesos'
guardarRNN(model,nombreArchivoModelo,nombreArchivoPesos)


# ## <span style="color:BLUE">PREDICCIÓN DEL MODELO</span>

# In[96]:




def predecir_imagen(img_url):
    datos = dict()
    dfPredicciones = pd.DataFrame([], columns = ['Prediccion' , 'Probabilidad'])
    tam_img = (100, 100)
    url_modelo = 'arquitectura_optimizada'
    url_pesos = 'pesos'
    modelo = cargar_rnn(url_modelo, url_pesos)

    img = load_img(img_url, target_size=tam_img)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255
    
    resultado = modelo.predict(img)
    
    
    probabilidad_aretes = resultado.tolist()[0][0]
    dfPredicciones.loc[0]=[ 'Aretes', probabilidad_aretes]
    probabilidad_backpack = resultado.tolist()[0][1]
    dfPredicciones.loc[1]=[ 'Backpack', probabilidad_backpack]
    probabilidad_bufanda = resultado.tolist()[0][2]
    dfPredicciones.loc[2]=[ 'Bufanda', probabilidad_bufanda]
    probabilidad_camisas = resultado.tolist()[0][3]
    dfPredicciones.loc[3]=[ 'Camisas', probabilidad_camisas]
    probabilidad_camisetas = resultado.tolist()[0][4]
    dfPredicciones.loc[4]=[ 'Camisetas', probabilidad_camisetas]
    probabilidad_cartera = resultado.tolist()[0][5]
    dfPredicciones.loc[5]=[ 'Cartera', probabilidad_cartera]
    probabilidad_cinturon = resultado.tolist()[0][6]
    dfPredicciones.loc[6]=[ 'Cinturon', probabilidad_cinturon]
    probabilidad_corbata = resultado.tolist()[0][7]
    dfPredicciones.loc[7]=[ 'Corbata', probabilidad_corbata]
    probabilidad_gafas = resultado.tolist()[0][8]
    dfPredicciones.loc[8]=[ 'Gafas', probabilidad_gafas]
    probabilidad_gorro = resultado.tolist()[0][9]
    dfPredicciones.loc[9]=[ 'Gorro', probabilidad_gorro]
    probabilidad_medias = resultado.tolist()[0][10]
    dfPredicciones.loc[10]=[ 'Medias', probabilidad_medias]
    probabilidad_pantalonCorto = resultado.tolist()[0][11]
    dfPredicciones.loc[11]=[ 'Pantalon Corto', probabilidad_pantalonCorto]
    probabilidad_pantalonLargo = resultado.tolist()[0][12]
    dfPredicciones.loc[12]=[ 'Pantalon Largo', probabilidad_pantalonLargo]
    probabilidad_reloj = resultado.tolist()[0][13]
    dfPredicciones.loc[13]=[ 'Reloj', probabilidad_reloj]
    probabilidad_ropaInterior = resultado.tolist()[0][14]
    dfPredicciones.loc[14]=[ 'Ropa Interior Hombre', probabilidad_ropaInterior]
    probabilidad_roptaInteriorMujerC = resultado.tolist()[0][15]
    dfPredicciones.loc[15]=[ 'Ropa Interior Mujer-Calzón', probabilidad_roptaInteriorMujerC]
    probabilidad_sandalias = resultado.tolist()[0][16]
    dfPredicciones.loc[16]=[ 'Sandalias', probabilidad_sandalias]
    probabilidad_sosten = resultado.tolist()[0][17]
    dfPredicciones.loc[17]=[ 'Sosten', probabilidad_sosten]
    probabilidad_tacos = resultado.tolist()[0][18]
    dfPredicciones.loc[18]=[ 'Tacos', probabilidad_tacos]
    probabilidad_zapatos = resultado.tolist()[0][19]
    dfPredicciones.loc[19]=[ 'Zapatos', probabilidad_zapatos]
    return dfPredicciones
    

def cargar_rnn(nombreArchivoModelo, nombreArchivoPesos):
    k.reset_uids()
    # Cargar la Arquitectura desde el archivo JSON
    with open(nombreArchivoModelo + '.json', 'r') as f:
        model = model_from_json(f.read())
    # Cargar Pesos (weights) en el nuevo modelo
    model.load_weights(nombreArchivoPesos + '.h5')
    return model


# In[105]:


from tensorflow.keras import backend as k
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing import image
import pandas as pd

dfpredic = predecir_imagen("input/camisas/12348.jpg")
#print(dfpredic)
#Obtencion del valor mas alto del dataset 
r  = dfpredic["Probabilidad"].max()
is_respuesta = dfpredic.loc[:, 'Probabilidad'] == r
df_respuesta = dfpredic.loc[is_respuesta]
print(df_respuesta)


# In[140]:


#dfpredicAlto=dfpredic["Probabilidad"].apply(np.max)
r  = dfpredic["Probabilidad"].max()
is_respuesta = dfpredic.loc[:, 'Probabilidad'] == r
df_respuesta = dfpredic.loc[is_respuesta]
print(df_respuesta)


# Predecir una imagen desde una URL

# In[102]:




def predecir_imagen_URL(img_url):
    datos = dict()
    dfPredicciones = pd.DataFrame([], columns = ['Prediccion' , 'Probabilidad'])
    res = request.urlopen(url).read()
    Sample_Image = Image.open(BytesIO(res)).resize((100,100))
    plt.imshow(Sample_Image)
    url_modelo = 'arquitectura_optimizada'
    url_pesos = 'pesos'
    modelo = cargar_rnn(url_modelo, url_pesos)

    img = image.img_to_array(Sample_Image)
    img = np.expand_dims(img, axis=0)
    img /= 255
    resultado = modelo.predict(img)

    probabilidad_aretes = resultado.tolist()[0][0]
    dfPredicciones.loc[0]=[ 'Aretes', probabilidad_aretes]
    probabilidad_backpack = resultado.tolist()[0][1]
    dfPredicciones.loc[1]=[ 'Backpack', probabilidad_backpack]
    probabilidad_bufanda = resultado.tolist()[0][2]
    dfPredicciones.loc[2]=[ 'Bufanda', probabilidad_bufanda]
    probabilidad_camisas = resultado.tolist()[0][3]
    dfPredicciones.loc[3]=[ 'Camisas', probabilidad_camisas]
    probabilidad_camisetas = resultado.tolist()[0][4]
    dfPredicciones.loc[4]=[ 'Camisetas', probabilidad_camisetas]
    probabilidad_cartera = resultado.tolist()[0][5]
    dfPredicciones.loc[5]=[ 'Cartera', probabilidad_cartera]
    probabilidad_cinturon = resultado.tolist()[0][6]
    dfPredicciones.loc[6]=[ 'Cinturon', probabilidad_cinturon]
    probabilidad_corbata = resultado.tolist()[0][7]
    dfPredicciones.loc[7]=[ 'Corbata', probabilidad_corbata]
    probabilidad_gafas = resultado.tolist()[0][8]
    dfPredicciones.loc[8]=[ 'Gafas', probabilidad_gafas]
    probabilidad_gorro = resultado.tolist()[0][9]
    dfPredicciones.loc[9]=[ 'Gorro', probabilidad_gorro]
    probabilidad_medias = resultado.tolist()[0][10]
    dfPredicciones.loc[10]=[ 'Medias', probabilidad_medias]
    probabilidad_pantalonCorto = resultado.tolist()[0][11]
    dfPredicciones.loc[11]=[ 'Pantalon Corto', probabilidad_pantalonCorto]
    probabilidad_pantalonLargo = resultado.tolist()[0][12]
    dfPredicciones.loc[12]=[ 'Pantalon Largo', probabilidad_pantalonLargo]
    probabilidad_reloj = resultado.tolist()[0][13]
    dfPredicciones.loc[13]=[ 'Reloj', probabilidad_reloj]
    probabilidad_ropaInterior = resultado.tolist()[0][14]
    dfPredicciones.loc[14]=[ 'Ropa Interior Hombre', probabilidad_ropaInterior]
    probabilidad_roptaInteriorMujerC = resultado.tolist()[0][15]
    dfPredicciones.loc[15]=[ 'Ropa Interior Mujer-Calzón', probabilidad_roptaInteriorMujerC]
    probabilidad_sandalias = resultado.tolist()[0][16]
    dfPredicciones.loc[16]=[ 'Sandalias', probabilidad_sandalias]
    probabilidad_sosten = resultado.tolist()[0][17]
    dfPredicciones.loc[17]=[ 'Sosten', probabilidad_sosten]
    probabilidad_tacos = resultado.tolist()[0][18]
    dfPredicciones.loc[18]=[ 'Tacos', probabilidad_tacos]
    probabilidad_zapatos = resultado.tolist()[0][19]
    dfPredicciones.loc[19]=[ 'Zapatos', probabilidad_zapatos]
    return dfPredicciones
    
    #if probabilidad_thanos > probabilidad_grimace:
     #   datos['pred'] = 'THANOS'
      #  datos['prob'] = probabilidad_thanos
       # return datos
    #else:
     #   datos['pred'] = 'GRIMACE'
      #  datos['prob'] = probabilidad_grimace
       # return datos

def cargar_rnn(nombreArchivoModelo, nombreArchivoPesos):
    k.reset_uids()
    # Cargar la Arquitectura desde el archivo JSON
    with open(nombreArchivoModelo + '.json', 'r') as f:
        model = model_from_json(f.read())
    # Cargar Pesos (weights) en el nuevo modelo
    model.load_weights(nombreArchivoPesos + '.h5')
    return model


# In[103]:


from PIL import Image
from urllib import request
from io import BytesIO
from tensorflow.keras import backend as k
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing import image
import pandas as pd


url = "https://images-na.ssl-images-amazon.com/images/I/81r8TV0JvOL._UY500_.jpg"
dfpredic = predecir_imagen_URL(url)
#print(dfpredic)
#Obtencion del valor mas alto del dataset 
r  = dfpredic["Probabilidad"].max()
is_respuesta = dfpredic.loc[:, 'Probabilidad'] == r
df_respuesta = dfpredic.loc[is_respuesta]
print(df_respuesta)

