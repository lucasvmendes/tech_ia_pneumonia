import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from collections import Counter

# ============================
# CONFIGURAÇÕES INICIAIS
# ============================
DATA_DIR = 'data/train_data'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

# ============================
# FUNÇÃO DE CARREGAMENTO
# ============================
def carregar_datasets():
    print("🔹 Carregando datasets...")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "train"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary"
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "val"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary"
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "test"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary"
    )

    class_names = train_ds.class_names
    print(f"✅ Bases carregadas com sucesso! Classes: {class_names}")
    return train_ds, val_ds, test_ds, class_names

def preprocessar(train_ds, val_ds, test_ds):
    print("🔹 Aplicando pré-processamento...")

    # Removido o Rescaling, pois preprocess_input cuida disso

    train_ds = train_ds.map(lambda x, y: (x, y), num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (x, y), num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: (x, y), num_parallel_calls=AUTOTUNE)

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    print("✅ Pré-processamento concluído!")
    return train_ds, val_ds, test_ds

def visualizar_amostras(train_ds, class_names):
    print("🔹 Visualizando algumas imagens do dataset...")
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[int(labels[i])])
            plt.axis("off")
    plt.savefig("reports/exemplo_batch.png")
    print("✅ Imagem salva em reports/exemplo_batch.png")

# ============================
# CONSTRUÇÃO DO MODELO
# ============================
def criar_modelo_transfer_learning(input_shape=(224, 224, 3)):
    print("🔹 Construindo modelo com Transfer Learning (MobileNetV2 + Fine-tuning)...")

    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )

    # Congela as primeiras camadas
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
        tf.keras.layers.InputLayer(shape=input_shape),
        data_augmentation,
        tf.keras.layers.Lambda(preprocess_input),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    model.summary()
    return model

def calcular_pesos_classe(train_ds):
    print("🔹 Calculando pesos de classe...")
    all_labels = []
    for _, y in train_ds:
        all_labels.extend(y.numpy().flatten().astype(int).tolist())

    contador = Counter(all_labels)
    total = sum(contador.values())
    peso_normal = total / (2 * contador[0])
    peso_pneumonia = total / (2 * contador[1])
    class_weights = {0: peso_normal, 1: peso_pneumonia}
    print(f"✅ Pesos de classe: {class_weights}")
    return class_weights

def verificar_balanceamento(ds, nome="validação"):
    print(f"🔍 Verificando balanceamento do conjunto de {nome}...")
    all_labels = []
    for _, y in ds:
        all_labels.extend(y.numpy().flatten().astype(int).tolist())
    print(f"🔸 Distribuição de classes ({nome}): {Counter(all_labels)}")

# ============================
# TREINAMENTO
# ============================
def treinar_modelo(model, train_ds, val_ds, epochs=8):
    print("🔹 Iniciando treinamento...")
    class_weights = calcular_pesos_classe(train_ds)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights
    )
    model.save_weights('models/pesos_mobilenetv2.weights.h5')
    print("✅ Pesos salvos em 'models/pesos_mobilenetv2.weights.h5'")
    print("✅ Modelo salvo em 'models/modelo_mobilenetv2_finetuned.keras'")
    return history

# ============================
# AVALIAÇÃO
# ============================
def avaliar_modelo(model, test_ds):
    print("🔹 Avaliando modelo no conjunto de teste...")
    y_true = []
    y_pred = []

    for imgs, labels in test_ds:
        preds = model.predict(imgs)
        y_true.extend(labels.numpy().astype(int))
        y_pred.extend((preds > 0.5).astype(int).flatten())

    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['NORMAL', 'PNEUMONIA']))

    print("\n📉 Matriz de confusão:")
    print(confusion_matrix(y_true, y_pred))

# ============================
# MAIN
# ============================
def main():
    print("🚀 Iniciando pipeline de dados...")
    train_ds, val_ds, test_ds, class_names = carregar_datasets()
    train_ds, val_ds, test_ds = preprocessar(train_ds, val_ds, test_ds)

    verificar_balanceamento(val_ds, nome="validação")

    visualizar_amostras(train_ds, class_names)

    print("🧠 Criando e treinando modelo...")
    model = criar_modelo_transfer_learning(input_shape=(224,224,3))
    history = treinar_modelo(model, train_ds, val_ds, epochs=8)
    plotar_metricas(history)

    avaliar_modelo(model, test_ds)
    print("🏁 Treinamento e avaliação concluídos!")

def plotar_metricas(history):
    print("📈 Gerando gráficos de desempenho...")
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    precision = history.history.get('precision', [])
    recall = history.history.get('recall', [])
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Treino')
    plt.plot(epochs_range, val_acc, label='Validação')
    plt.legend(loc='lower right')
    plt.title('Acurácia')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Treino')
    plt.plot(epochs_range, val_loss, label='Validação')
    plt.legend(loc='upper right')
    plt.title('Perda')

    plt.savefig('reports/historico_treinamento.png')
    print("✅ Gráficos salvos em reports/historico_treinamento.png")

if __name__ == "__main__":
    main()
