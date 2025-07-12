import serial
import time
import matplotlib.pyplot as plt
import os
from tkinter import Tk
import tkinter.filedialog as filedlog
import inquirer
Tk().withdraw()
questions = [
    inquirer.List(
        "choice",
        message="Que voulez vous faire ?",
        choices=["Write", "Read"],
    ),
]
choice = inquirer.prompt(questions)["choice"]

if choice == 'Write' :
    # Configuration du port série
    ser = serial.Serial(port='/dev/ttyUSB0', baudrate=115200, timeout=1)

    output_file = os.path.join(os.path.dirname(__file__), "donnees_tension.txt")

    # Listes pour stocker les données
    timestamps = []
    values = []

    # Attente initiale
    questions = [
    inquirer.List(
        "choice",
        message="Temps d'attente",
        choices=["0","2.5", "5"],
    ),
]
    waittime = float(inquirer.prompt(questions)["choice"])
    print("Préparation... Attente de " + str(waittime) + " secondes avant le début de l'acquisition.")
    time.sleep(waittime)

    # Lancement du chronomètre
    acquisition_duration = 10  # durée totale
    start_time = time.time()
    print("Acquisition en cours... (les 2 premières secondes seront ignorées)")

    try:
        while time.time() - start_time < acquisition_duration:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()  
                try:
                    value = float(line)
                    current_time = time.time() - start_time
                    if current_time >= 2:  # Ignorer les 2 premières secondes
                        timestamps.append(current_time)
                        values.append(value)
                        print(f"{current_time:.2f}s -> {value}")
                
                except ValueError:
                    pass  # Ignore les lignes non valides
                
    finally:
        ser.close()
if choice == 'Read':
    filename = filedlog.askopenfilename()
    input_file = os.path.join(os.path.dirname(__file__), filename)
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

print(len(values))
# Affichage du graphique
plt.figure(figsize=(10, 5))
plt.plot(timestamps, values, marker='o', linestyle='-', color='blue')
if choice == 'Write':
    plt.title("Tension reçue via port série (après 2s d'acquisition)")
if choice == 'Read':
    plt.title("Tension reçue via port série (après 2s d'acquisition) Dossier" + filename)
plt.xlabel("Temps (s)")
plt.ylabel("Tension (mV)")
plt.ylim(500, 3400)  # Échelle fixée
plt.grid(True)
plt.tight_layout()
plt.show()
if choice == 'Write':
    output_file = filedlog.asksaveasfilename(defaultextension=".txt")
    if output_file == None:
        output_file = "donnees_tension.txt"
    with open(output_file, 'w') as f:
        f.writelines(str(values).replace('.0',''))