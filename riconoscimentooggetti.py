from ultralytics import YOLO
import cv2
import cvzone
import math
import time
# Carica il modello YOLO per il rilevamento degli oggetti
model = YOLO("yolov8n.pt")

camera = cv2.VideoCapture(0)
camera.set(3, 1280)  # Larghezza
camera.set(4, 720)  # Altezza

# Lista delle classi di oggetti che il modello è in grado di rilevare
class_names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
               "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
               "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
               "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
               "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
               "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
               "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
               "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
               "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
               "teddy bear", "hair drier", "toothbrush"]

scelta = input("Vuoi analizzare (1) un'immagine caricata da file o (2) tramite la webcam? (Inserisci 1 o 2): ")

if scelta == '1':
    # Carica l'immagine
    image_path = r"C:\Users\Cristian\PycharmProjects\PythonProject\scuola\raybay-kG71BXh8KFw-unsplash-1024x806.jpg"  # Sostituisci con il percorso dell'immagine
    image = cv2.imread(image_path)

    if image is None:
        print("Errore nel caricamento dell'immagine!")
        exit()

    # Esegui le predizioni con il modello YOLO
    predictions = model(image)  # Esegui il modello sul frame corrente

    # Verifica se ci sono oggetti rilevati
    if predictions:
        for prediction in predictions:
            bounding_boxes = prediction.boxes
            if bounding_boxes is not None:
                for box in bounding_boxes:
                    # Estrai le coordinate del bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    width, height = x2 - x1, y2 - y1

                    # Disegna il rettangolo attorno all'oggetto rilevato
                    cvzone.cornerRect(image, (x1, y1, width, height))

                    # Calcola la confidenza e la classe dell'oggetto
                    confidence = math.ceil(box.conf[0] * 100)  # Calcola la confidenza in percentuale
                    class_id = int(box.cls[0])

                    # Aggiungi il testo con il nome dell'oggetto e la confidenza sull'immagine
                    cvzone.putTextRect(image, f'{class_names[class_id]} {confidence}%', (max(0, x1), max(35, y1)),
                                       scale=1, thickness=1)

    # Mostra l'immagine con le rilevazioni
    cv2.imshow("Rilevamento Oggetti", image)

    # Aspetta che venga premuto un tasto per chiudere la finestra
    cv2.waitKey(0)
    cv2.destroyAllWindows()
elif scelta == '2':
    # Variabili per il calcolo del frame rate (FPS)
    previous_frame_time = 0

    # Ciclo principale per l'acquisizione dei frame dalla fotocamera
    while True:
        current_frame_time = time.time()
        success, frame = camera.read()

        if not success:
            print("Errore nel catturare l'immagine")
            break

        # Fai le predizioni con il modello YOLO sul frame corrente
        predictions = model(frame)  # Esegui il modello sul frame

        if predictions is not None:
            for prediction in predictions:
                bounding_boxes = prediction.boxes
                if bounding_boxes is not None:
                    for box in bounding_boxes:
                        # Estrai le coordinate del bounding box
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        width, height = x2 - x1, y2 - y1

                        # Disegna il rettangolo attorno all'oggetto
                        cvzone.cornerRect(frame, (x1, y1, width, height))

                        # Calcola la confidenza e la classe dell'oggetto
                        confidence = math.ceil(box.conf[0] * 100)  # Confidenza in percentuale
                        class_id = int(box.cls[0])

                        # Aggiungi il testo con la classe dell'oggetto e la confidenza sul frame
                        cvzone.putTextRect(frame, f'{class_names[class_id]} {confidence}%', (max(0, x1), max(35, y1)),
                                           scale=1, thickness=1)

        # Calcola e stampa il frame rate (FPS)
        fps = 1 / (current_frame_time - previous_frame_time)
        previous_frame_time = current_frame_time
        print(f"FPS: {fps:.2f}")

        # Mostra il frame con le rilevazioni
        cv2.imshow("Rilevamento Oggetti in Tempo Reale", frame)

        # Esci dal ciclo con il tasto 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Rilascia la videocamera e chiudi le finestre
    camera.release()
    cv2.destroyAllWindows()





