import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import random
from tensorflow.keras.callbacks import EarlyStopping

# Paramètres
INPUT_LENGTH = 16000
DATA_DIR = os.path.join(os.path.dirname(__file__), 'données')
MODEL_PATH = "siamese_ecg_model.keras"
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Lecture des fichiers .txt contenant les données
def read_signal(filename):
    path = os.path.join(DATA_DIR, filename + '.txt')
    with open(path, 'r') as f:
        data = f.read().replace('[','').replace(']','').split(',')
        return np.array([float(x.strip()) for x in data if x.strip()])

# Traitement de la longueur du signal ainsi que le phasage
def preprocess(signal, length=INPUT_LENGTH):
    signal = np.array(signal)
    if len(signal) > length:
        signal = signal[:length]
    else:
        signal = np.pad(signal, (0, length - len(signal)), 'constant')
    return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

# Distortion légère du signal pour la variation des données utilisé pour l'entrainement
def augment(signal):
    signal = signal + np.random.normal(0, 0.02, len(signal))
    shift = np.random.randint(-5, 5)
    return np.roll(signal, shift)

# Encodeur
def build_encoder(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Reshape((input_shape[0], 1))(inputs)
    x = layers.Conv1D(32, 5, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, 3, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    return models.Model(inputs, x, name="Encoder")

# Utilisation du système de comparaison constrative loss (TensorFlow ne l'a possède pas de base)
@tf.keras.utils.register_keras_serializable()
def contrastive_loss(y_true, y_pred):
    margin = 1.0
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.squeeze(y_pred, axis=-1)
    return tf.reduce_mean(y_true * tf.square(y_pred) + (1 - y_true) * tf.square(tf.maximum(margin - y_pred, 0)))

# Création du réseaux siamois
def build_siamese(input_shape):
    encoder = build_encoder(input_shape)
    input1 = layers.Input(shape=input_shape)
    input2 = layers.Input(shape=input_shape)
    encoded1 = encoder(input1)
    encoded2 = encoder(input2)
    diff = layers.Lambda(lambda x: tf.abs(x[0] - x[1]))([encoded1, encoded2])
    output = layers.Dense(1)(diff)  # Pas de sigmoid
    model = models.Model([input1, input2], output)
    model.compile(optimizer='adam', loss=contrastive_loss)
    return model

# Création du jeux de données
def create_dataset():
    personnes = {
        "LUCAS": [f"LUCAS{i}" for i in range(1, 6)],
        "KIKI": [f"KIKI{i}" for i in range(1, 8)],
        "MARC": [f"MARC{i}" for i in range(1, 10)],
        "CYPRIEN": [f"CYPRIEN{i}" for i in range(1, 10)],
        "JOJO": [f"JOJO{i}" for i in range(1, 3)]
    }

    pairs = []
    labels = []

    # Paires similaires
    for person, signals in personnes.items():
        for i in range(len(signals)):
            for j in range(i+1, len(signals)):
                s1 = preprocess(augment(read_signal(signals[i])))
                s2 = preprocess(augment(read_signal(signals[j])))
                pairs.append((s1, s2))
                labels.append(1)

    # Paires différentes
    all_people = list(personnes.keys())
    for i in range(len(all_people)):
        for j in range(i+1, len(all_people)):
            for s1_file in personnes[all_people[i]]:
                for s2_file in personnes[all_people[j]]:
                    s1 = preprocess(augment(read_signal(s1_file)))
                    s2 = preprocess(augment(read_signal(s2_file)))
                    pairs.append((s1, s2))
                    labels.append(0)

    # Mélange
    combined = list(zip(pairs, labels))
    random.shuffle(combined)
    pairs, labels = zip(*combined)
    X1 = np.array([p[0] for p in pairs])
    X2 = np.array([p[1] for p in pairs])
    y = np.array(labels)

    return X1, X2, y

# Entrainement
def train_model():
    print("Chargement des données...")
    X1, X2, y = create_dataset()
    print(f"{len(y)} paires totales : {np.sum(y==1)} similaires, {np.sum(y==0)} différentes")

    model = build_siamese((INPUT_LENGTH,))
    es = EarlyStopping(patience=5, restore_best_weights=True)

    print("Entraînement du modèle...")
    model.fit([X1, X2], y, validation_split=0.2, epochs=50, batch_size=32, callbacks=[es])

    print(f"Sauvegarde du modèle dans {MODEL_PATH}")
    model.save(MODEL_PATH, save_format='keras')

# Lancement du programme
if __name__ == "__main__":
    train_model()
