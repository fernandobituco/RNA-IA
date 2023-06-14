import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import vis_utils

# Carregar o arquivo CSV
data = pd.read_csv('Pokemon.csv')

# Selecionar os recursos e rótulos
features = ['Comprimento da Sépala', 'Largura da Sépala', 'Comprimento da Pétala', 'Largura da Pétala']
label = 'Espécie'

# Dividir os dados em conjuntos de treinamento e teste
train_data, test_data = train_test_split(data, test_size=0.2)

# Converter os rótulos usando o LabelEncoder
label_encoder = LabelEncoder()
train_data[label] = label_encoder.fit_transform(train_data[label])
test_data[label] = label_encoder.transform(test_data[label])

# Criar o modelo da rede neural
model = keras.Sequential([
    layers.Dense(10, activation='relu', input_shape=(len(features),)),
    layers.Dense(5, activation='softmax')
])

# Compilar o modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Treinar o modelo
history = model.fit(train_data[features], train_data[label], epochs=10, validation_data=(test_data[features], test_data[label]))

# Visualizar o modelo
vis_utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True, show_trainable=True)
