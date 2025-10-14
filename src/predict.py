import os
import sys
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image

PESOS_PATH = "models/pesos_mobilenetv2.weights.h5"

def criar_modelo_transfer_learning(input_shape=(224, 224, 3)):
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )

    for layer in base_model.layers[:-40]:
        layer.trainable = False
    for layer in base_model.layers[-40:]:
        layer.trainable = True

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    model = tf.keras.Sequential([
        layers.InputLayer(shape=input_shape),
        data_augmentation,
        layers.Lambda(preprocess_input),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

def carregar_modelo():
    print(f"🔹 Recriando arquitetura e carregando pesos: {PESOS_PATH}")
    model = criar_modelo_transfer_learning()
    model.load_weights(PESOS_PATH)
    print("✅ Pesos carregados com sucesso!")
    return model

def preprocessar_imagem(caminho):
    img = Image.open(caminho).convert("RGB").resize((224, 224))
    img_array = np.expand_dims(np.array(img), axis=0)
    return img_array

def prever(model, caminho_imagem):
    print(f"🔹 Analisando imagem: {caminho_imagem}")
    img = preprocessar_imagem(caminho_imagem)
    preds = model.predict(img)
    conf = preds[0][0]
    label = "PNEUMONIA" if conf > 0.5 else "NORMAL"
    print(f"✅ Diagnóstico previsto: {label} (confiança: {conf:.4f})")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Uso: python src/predict.py caminho/para/imagem.jpeg")
        sys.exit(1)

    caminho_imagem = sys.argv[1]
    model = carregar_modelo()
    prever(model, caminho_imagem)
