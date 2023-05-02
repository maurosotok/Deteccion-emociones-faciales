import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import ImageTk, Image
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# Cargando el clasificador de cascada frontal para deteccion de caras
face_classifier = cv2.CascadeClassifier(
    './haarcascade_frontalface_default.xml')

# Cargando el modelo pre-entrenado de deteccion de emociones
classifier = keras.models.load_model('Emotion_Detection.h5')

# Definiendo las etiquetas de clase para cada emocion detectada
class_labels = ['Enojo', 'Feliz', 'Neutral', 'Triste', 'Sorprendido']

faces = []

root = tk.Tk()
# Create a label to display the selected image
label = tk.Label(root)
label.pack()

def open_file():    

    # Recibir la imagen
    file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.jpg *.jpeg *.png *.gif")])
    
    # Redimensionamos la imagen y convertimos a gray
    face = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (48, 48))
    face2 = tf.keras.utils.img_to_array(face)
    face2 = np.expand_dims(face2,axis=0)

    faces.append(face2) #enviamos la imagen a la lista    

    if len(faces) == 2:
        faces.pop(0)
        
    # El modelo estima la predicción
    preds = classifier.predict(faces)    
    result = class_labels[np.argmax(preds)]

    if np.argmax(preds) == 4:
        w = tk.Label(root, padx = 10, text = f'Emoción detectada: {result}',font=("Arial", 16, "bold"), bg="#3ED3BF", fg="#F7FFF7")    
    else:
        w = tk.Label(root, padx = 10, text = f'Emoción detectada: {result}         ',font=(
        "Arial", 16, "bold"), bg="#3ED3BF", fg="#F7FFF7")        
    
    w.pack()
    w.place(x=85, y=80)
   
# Funcion para comenzar la deteccion de emociones
def start_detection():

    # Inicializando la camara
    cap = cv2.VideoCapture(0)

    # Ciclo para capturar imagenes continuamente de la camara
    while True:

        # Capturando un fotograma de la camara
        ret, frame = cap.read()

        # Inicializando una lista para almacenar las etiquetas de las emociones detectadas en cada cara
        labels = []

        # Convirtiendo la imagen capturada a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectando caras en la imagen en escala de grises
        faces = face_classifier.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5)

        # Iterando a traves de cada cara detectada
        for (x, y, w, h) in faces:

            # Dibujando un rectangulo alrededor de la cara detectada
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Recortando la region de interes (ROI) de la cara detectada en la imagen en escala de grises
            roi_gray = gray[y:y+h, x:x+w]

            # Redimensionando la ROI a 48x48 pixeles para que coincida con el tamano del conjunto de datos utilizado para entrenar el modelo
            roi_gray = cv2.resize(roi_gray, (48, 48),
                                  interpolation=cv2.INTER_AREA)

            # Si la ROI no esta vacia
            if np.sum([roi_gray]) != 0:

                # Normalizando la ROI y convirtiendola en un array
                roi = roi_gray.astype('float')/255.0
                roi = tf.keras.utils.img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                # Utilizando el modelo para predecir la emocion en la ROI
                preds = classifier.predict(roi)[0]

                # Obteniendo la etiqueta de clase de la emocion con la probabilidad mas alta
                label = class_labels[preds.argmax()]

                # Almacenando la etiqueta de la emocion en la lista de etiquetas
                labels.append(label)

                # Dibujando la etiqueta de la emocion sobre la cara detectada
                label_position = (x, y)
                cv2.putText(frame, label, label_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

            # Si la ROI esta vacia, se muestra un mensaje de "No se encontro cara"
            else:
                cv2.putText(frame, 'No se encontro una cara', (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        # Mostrando el frame en una ventana
        cv2.imshow('Detector de Emociones (ESC Salir)', frame)

        # Salir del ciclo si se presiona la tecla ESC
        if cv2.waitKey(1) == 27:
            break

    # Liberando la camara y cerrando todas las ventanas
    cap.release()
    cv2.destroyAllWindows()

# Funcion para cerrar la interfaz
def cerrar():
    exit()

# Funcion principal

def start_emotion_detection():

    root.title("Detector de Emociones")
    width = 500
    height = 300
    x = 540
    y = 250
    root.geometry(f'{width}x{height}+{x}+{y}')
    root.configure(bg="#3D405B")
    # Botones
    start_button = tk.Button(root, text="Detectar Emociones Faciales", font=(
        "Arial", 16, "bold"), bg="#E07A5F", fg="#F7FFF7", command=start_detection)
    exit_button = tk.Button(root, text="Salir", font=(
        "Arial", 12, "bold"), bg="#E07A5F", fg="#F7FFF7", command=cerrar)

    button = tk.Button(root, text="Subir una imagen", font=(
        "Arial", 16, "bold"), bg="#E07A5F", fg="#F7FFF7", command=open_file)

    start_button.place(x=100, y=150)
    exit_button.place(x=230, y=220)
    button.pack()
    root.mainloop()

# Iniciamos la aplicacion
start_emotion_detection()
