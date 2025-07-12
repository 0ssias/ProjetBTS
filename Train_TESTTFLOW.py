import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import random
from tensorflow.keras.callbacks import EarlyStopping

# === PARAMÈTRES ===
INPUT_LENGTH = 16000
DATA_DIR = os.path.join(os.path.dirname(__file__), 'données')
MODEL_PATH = "siamese_ecg_model.keras"
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# === LECTURE DU FICHIER ===
def read_signal(filename):
    path = os.path.join(DATA_DIR, filename + '.txt')
    with open(path, 'r') as f:
        data = f.read().replace('[','').replace(']','').split(',')
        return np.array([float(x.strip()) for x in data if x.strip()])

# === PRÉTRAITEMENT ===
def preprocess(signal, length=INPUT_LENGTH):
    signal = np.array(signal)
    if len(signal) > length:
        signal = signal[:length]
    else:
        signal = np.pad(signal, (0, length - len(signal)), 'constant')
    return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

# === AUGMENTATION LÉGÈRE ===
def augment(signal):
    signal = signal + np.random.normal(0, 0.02, len(signal))
    shift = np.random.randint(-5, 5)
    return np.roll(signal, shift)

# === ENCODEUR ===
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

# === CONTRASTIVE LOSS ===
@tf.keras.utils.register_keras_serializable()
def contrastive_loss(y_true, y_pred):
    margin = 1.0
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.squeeze(y_pred, axis=-1)
    return tf.reduce_mean(y_true * tf.square(y_pred) + (1 - y_true) * tf.square(tf.maximum(margin - y_pred, 0)))

# === RÉSEAU SIAMOIS ===
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

# === GÉNÉRATION DES DONNÉES ===
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

# === ENTRAÎNEMENT ===
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

# === LANCEMENT ===
if __name__ == "__main__":
    train_model()

'''
# === PARAMÈTRES ===
INPUT_LENGTH = 798
MODEL_PATH = "siamese_ecg_model.keras"

# === FONCTION DE PRÉTRAITEMENT ===
def preprocess(signal, target_length=INPUT_LENGTH):
    signal = np.array(signal)
    if len(signal) > target_length:
        signal = signal[:target_length]
    elif len(signal) < target_length:
        signal = np.pad(signal, (0, target_length - len(signal)), mode='constant')
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
    return signal

# === ENCODEUR CONV1D ===
def build_encoder(input_shape):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Reshape((input_shape[0], 1)),
        tf.keras.layers.Conv1D(32, 5, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(64, 3, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(64, activation='relu')
    ])
@tf.keras.utils.register_keras_serializable()
def contrastive_loss(y_true, y_pred):
    margin = 1.0
    y_true = tf.cast(y_true, tf.float32)
    return tf.reduce_mean(
        y_true * tf.square(y_pred) +
        (1.0 - y_true) * tf.square(tf.maximum(margin - y_pred, 0))
    )
# === RÉSEAU SIAMESE ===
def build_siamese(input_shape):
    encoder = build_encoder(input_shape)
    input1 = tf.keras.Input(shape=input_shape)
    input2 = tf.keras.Input(shape=input_shape)
    encoded1 = encoder(input1)
    encoded2 = encoder(input2)
    diff = tf.keras.layers.Lambda(lambda x: tf.abs(x[0] - x[1]))([encoded1, encoded2])
    output = tf.keras.layers.Dense(1, activation='sigmoid')(diff)
    model = tf.keras.Model(inputs=[input1, input2], outputs=output)
    model.compile(optimizer='adam', loss=contrastive_loss)
    return model

def read_signal(file):
    input_file = os.path.join(os.path.dirname(__file__) + '/données', file + '.txt')
    with open(input_file, 'r') as f:
        read = f.read().replace('[','').replace(']','').split(',')
        for i in range(0,len(read)):
            read[i] = float(read[i])
    values = read
    return values
def nuancer(signal):
    signal = signal + np.random.normal(0, 0.05, len(signal))  # Bruit
    shift = np.random.randint(-10, 10)  # Décalage
    signal = np.roll(signal, shift)
    return signal


def charger_donnees_reelles():
    LUCAS = ["LUCAS1", "LUCAS2", "LUCAS3", "LUCAS4"],
    KIKI = ["KIKI1", "KIKI2", "KIKI3", "KIKI4"],
    MARC = ["MARC1", "MARC2","MARC3","MARC4"],
    CYPRIEN = ["CYPRIEN1","CYPRIEN2","CYPRIEN3","CYPRIEN4"]
    TOTAL = [LUCAS,KIKI,MARC,CYPRIEN]
    pairs = []
    labels = []
    # === GÉNÉRER PAIRES SIMILAIRES (label 1) ===
    for i in range(0,len(TOTAL)):
        for j in range(0,len(TOTAL[i]))
            for k in range(0,len(TOTAL[i]))
    # === GÉNÉRER PAIRES DIFFÉRENTES (label 0) ===
    noms = list(fichiers.keys())
    for nom1, nom2 in itertools.permutations(noms, 2):
        for f1 in fichiers[nom1]:
            for f2 in fichiers[nom2]:
                s1 = preprocess(read_signal(f1))
                s2 = preprocess(read_signal(f2))
                pairs.append((s1, s2))
                labels.append(0)

    indices = list(range(len(pairs)))
    random.shuffle(indices)

    # Séparer en X1, X2, y
    X1 = [p[0] for p in pairs]
    X2 = [p[1] for p in pairs]
    y = np.array(labels)

    return np.array(X1), np.array(X2), y
def charger_donnees_reelles():
    LUCAS = ["LUCAS1", "LUCAS2", "LUCAS3", "LUCAS4", "LUCAS5"]
    KIKI = ["KIKI1", "KIKI2", "KIKI3", "KIKI4", "KIKI5", "KIKI6", "KIKI7"]
    MARC = ["MARC1", "MARC2","MARC3","MARC4", "MARC5","MARC6","MARC7", "MARC8","MARC9"]
    CYPRIEN = ["CYPRIEN1","CYPRIEN2","CYPRIEN3","CYPRIEN4","CYPRIEN5","CYPRIEN6","CYPRIEN7","CYPRIEN8","CYPRIEN9"]
    JOJO = ["JOJO1","JOJO2"]
    TOTAL = [LUCAS,KIKI,MARC,CYPRIEN,JOJO]
    pairs = []
    labels = []
    for i in range(0,len(TOTAL)):
        for j in range(0,len(TOTAL[i])):
            for k in range(0,len(TOTAL)):
                for l in range(0,len(TOTAL[k])):
                    signal1 = preprocess(nuancer(read_signal(TOTAL[i][j])))
                    signal2 = preprocess(nuancer(read_signal(TOTAL[k][l])))
                    pairs.append((signal1,signal2))
                    if TOTAL[i] == TOTAL[k]:
                        labels.append(1)
                    else:
                        labels.append(0)
    indices = random.sample(range(len(pairs)), len(pairs))
    pairs = [pairs[i] for i in indices]  
    labels = [labels[i] for i in indices]
    X1 = [p[0] for p in pairs]
    X2 = [p[1] for p in pairs]
    y = np.array(labels)

    return np.array(X1), np.array(X2), y 

# === ENTRAÎNEMENT ===
def train_and_save():
    print("Génération des données...")
    X1, X2, y = charger_donnees_reelles()
    print("Nombre total de paires :", len(y))
    print("Nombre de paires similaires (label=1) :", np.sum(y==1))
    print("Nombre de paires différentes (label=0) :", np.sum(y==0))
    model = build_siamese((INPUT_LENGTH,))
    print("Entraînement...")
    es = EarlyStopping(patience=10, restore_best_weights=True)
    model.fit([X1, X2], y, epochs=65, batch_size=32, validation_split=0.2, callbacks=[es])
    print(f"Sauvegarde du modèle dans : {MODEL_PATH}")
    model.save(MODEL_PATH, save_format="keras")

if __name__ == "__main__":
    train_and_save()
'''