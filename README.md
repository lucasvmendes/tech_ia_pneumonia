🩺 Detector de Pneumonia (MobileNetV2 + Fine-tuning)

Este projeto implementa uma rede neural convolucional com Transfer Learning (MobileNetV2) para detecção automática de pneumonia em radiografias de tórax.
Ele inclui scripts para treinamento, avaliação e inferência, e é totalmente executável via Docker ou ambiente Python local.

📂 Estrutura do Projeto

deteccao_pneumonia/
├── src/
│   ├── main.py          # Script de treinamento e avaliação
│   ├── predict.py       # Script de inferência (diagnóstico de uma imagem)
│   └── requirements.txt # Dependências do projeto
│
├── data/                # Diretório para o dataset (não incluído no repositório)
│   └── train_data/
│       ├── train/
│       │   ├── NORMAL/
│       │   └── PNEUMONIA/
│       ├── val/
│       │   ├── NORMAL/
│       │   └── PNEUMONIA/
│       └── test/
│           ├── NORMAL/
│           └── PNEUMONIA/
│
├── models/              # Modelos e pesos salvos
│   └── (gerados após o treinamento)
│
├── reports/             # Imagens e gráficos gerados durante o treinamento
│   └── exemplo_batch.png
│
├── Dockerfile
├── README.md
└── .gitignore

🧠 Tecnologias Utilizadas

Python 3.11

TensorFlow / Keras

Scikit-learn

Matplotlib

Docker

📦 Instalação Local (sem Docker)
1️⃣ Criar e ativar ambiente virtual
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
2️⃣ Instalar dependências
pip install -r src/requirements.txt
3️⃣ Treinar o modelo
python src/main.py
4️⃣ Rodar uma inferência
python src/predict.py data/train_data/test/NORMAL/IM-0001-0001.jpeg

🐳 Execução via Docker
1️⃣ Build da imagem

Na raiz do projeto:
sudo docker build -t pneumonia-detector:latest .

2️⃣ Treinamento (gera os pesos na pasta models/)
sudo docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/reports:/app/reports \
  pneumonia-detector:latest

Durante o treinamento, o modelo será salvo em:
models/pesos_mobilenetv2.weights.h5

3️⃣ Inferência (diagnóstico de uma imagem)

Após o treinamento, execute:
sudo docker run --rm \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  pneumonia-detector:latest \
  python src/predict.py data/train_data/test/NORMAL/IM-0001-0001.jpeg

🧩 Dataset Utilizado

📦 Kaggle Dataset:
Chest X-Ray Images (Pneumonia) https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

A estrutura esperada é:
data/train_data/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/

📈 Resultados Obtidos
| Métrica       | Valor aproximado |
| ------------- | ---------------- |
| **Acurácia**  | 0.78             |
| **Precision** | 0.83             |
| **Recall**    | 0.78             |
| **F1-score**  | 0.75             |

🧾 Saídas geradas

Após o treinamento e a inferência, são gerados:

models/pesos_mobilenetv2.weights.h5 → pesos da rede treinada

reports/exemplo_batch.png → amostras visuais do dataset

Logs detalhados de acurácia, loss, precision e recall

🧠 Autor

Lucas Mendes
Projeto acadêmico: "Detecção de Pneumonia com Deep Learning e Transfer Learning (MobileNetV2)"

⚙️ Licença

Este projeto é distribuído sob a licença MIT.
Sinta-se à vontade para utilizar, modificar e compartilhar com atribuição.