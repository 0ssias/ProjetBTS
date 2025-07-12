import tensorflow as tf
import numpy as np
import os
from tkinter import Tk
import tkinter.filedialog as filedlog

INPUT_LENGTH = 798
MODEL_PATH = "siamese_ecg_model.keras"

# Prétraitement des données
def preprocess(signal, target_length=INPUT_LENGTH):
    signal = np.array(signal)
    if len(signal) > target_length:
        signal = signal[:target_length]
    elif len(signal) < target_length:
        signal = np.pad(signal, (0, target_length - len(signal)), mode='constant')
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
    return signal
@tf.keras.utils.register_keras_serializable()
def contrastive_loss(y_true, y_pred):
    margin = 1.0
    y_true = tf.cast(y_true, tf.float32)
    return tf.reduce_mean(
        y_true * tf.square(y_pred) +
        (1.0 - y_true) * tf.square(tf.maximum(margin - y_pred, 0)))
# Comparaison des signaux
def compare_signals(signal1, signal2):
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'contrastive_loss': contrastive_loss})
    s1 = preprocess(signal1).reshape(1, -1)
    s2 = preprocess(signal2).reshape(1, -1)
    pred = model.predict([s1, s2])[0][0]
    '''print(f"🔍 Similarité prédite : {pred:.4f}")
    if pred > 0.3:
        print("✅ Signatures similaires (même origine probable).")
    else:
        print("❌ Signatures différentes.")'''
    return pred

def read_signal(file):
    input_file = os.path.join(os.path.dirname(__file__), file)
    with open(input_file, 'r') as f:
        read = f.read().replace('[','').replace(']','').split(',')
        for i in range(0,len(read)):
            read[i] = float(read[i])
    values = read
    timestamps = []
    var = 2
    for i in range(0,len(read)):
        var += 8/len(read)
        timestamps.append(var)
    return values , timestamps


# Exemple d'utilisation
if __name__ == "__main__":
    # Exemple : deux signaux
    filename1 = filedlog.askopenfilename()
    filename2 = filedlog.askopenfilename()
    signal1 = read_signal(filename1)[0]
    signal2 = read_signal(filename2)[0]
    # signal2 = np.random.normal(0, 1, INPUT_LENGTH)  # Pour tester une différence

    print(compare_signals(signal1, signal2))
