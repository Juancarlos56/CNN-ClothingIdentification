from django.db import models
from django.urls import reverse
import pandas as pd
import numpy as np
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow.python.keras.models import load_model, model_from_json
from keras import backend as K
from apiSNN import models
import os
from tensorflow.python.keras.models import Sequential
import pathlib
from tensorflow.keras import backend as k
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing import image
import pandas as pd
from PIL import Image
from urllib import request
from io import BytesIO
from tensorflow.keras import backend as k
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing import image
import pandas as pd
import matplotlib.pyplot as plt




class modeloSNN():
    """Clase modelo SNN"""
    modelo = Sequential()

    def cargar_rnn(nombreArchivoModelo, nombreArchivoPesos):
        k.reset_uids()
        # Cargar la Arquitectura desde el archivo JSON
        with open(nombreArchivoModelo + '.json', 'r') as f:
            model = model_from_json(f.read())
        # Cargar Pesos (weights) en el nuevo modelo
        model.load_weights(nombreArchivoPesos + '.h5')
        print("Red Neuronal Cargada desde Archivo") 
        return model

    def predecir_imagen_URL(img_url):
        
        dfPredicciones = pd.DataFrame([], columns = ['Prediccion' , 'Probabilidad'])
        res = request.urlopen(img_url).read()
        Sample_Image = Image.open(BytesIO(res)).resize((100,100))
        plt.imshow(Sample_Image)
        
        print('MODELO OPTIMIZADO')
        nombreArchivoModelo=r'apiSNN/Logica/arquitectura_optimizada'
        nombreArchivoPesos=r'apiSNN/Logica/pesos_optimizados'
        
        self.modelo = self.cargar_rnn(nombreArchivoModelo, nombreArchivoPesos)
        print(self.modelo)
        print(self.modelo.summary())

        img = image.img_to_array(Sample_Image)
        img = np.expand_dims(img, axis=0)
        img /= 255

        resultado = self.modelo.predict(img)
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

       
    def predict(url):
        dfpredic = self.predecir_imagen_URL(url)
        #print(dfpredic)
        #Obtencion del valor mas alto del dataset 
        r  = dfpredic["Probabilidad"].max()
        is_respuesta = dfpredic.loc[:, 'Probabilidad'] == r
        df_respuesta = dfpredic.loc[is_respuesta]
        print(df_respuesta)
        return df_respuesta

