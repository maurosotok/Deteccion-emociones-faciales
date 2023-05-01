import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tkinter as tk

# Cargando el clasificador de cascada frontal para deteccion de caras
face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

# Cargando el modelo pre-entrenado de deteccion de emociones
classifier = load_model('Emotion_Detection.h5')

# Definiendo las etiquetas de clase para cada emocion detectada
class_labels = ['Enojo', 'Feliz', 'Neutral', 'Triste', 'Sorprendido']

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
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # Iterando a traves de cada cara detectada
        for (x, y, w, h) in faces:

            # Dibujando un rectangulo alrededor de la cara detectada
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Recortando la region de interes (ROI) de la cara detectada en la imagen en escala de grises
            roi_gray = gray[y:y+h, x:x+w]

            # Redimensionando la ROI a 48x48 pixeles para que coincida con el tamano del conjunto de datos utilizado para entrenar el modelo
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            # Si la ROI no esta vacia
            if np.sum([roi_gray]) != 0:

                # Normalizando la ROI y convirtiendola en un array
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
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
    
    root = tk.Tk()
    root.title("Detector de Emociones")
    root.geometry("400x200")              
    root.configure(bg="#3D405B")          

    #Botones
    start_button = tk.Button(root, text="Detectar Emociones Faciales", font=("Arial", 16, "bold"), bg="#E07A5F", fg="#F7FFF7", command=start_detection)
    exit_button = tk.Button(root, text="Salir", font=("Arial", 12, "bold"), bg="#E07A5F", fg="#F7FFF7", command=cerrar)

    start_button.pack(pady=50)
    exit_button.pack(pady=0)

    root.mainloop()

# Iniciamos la aplicacion
start_emotion_detection()