import numpy as np
import cv2
import json

# Definirea unei liste de nume pentru clasele de obiecte pe care le poate detecta rețeaua
classNames = {0: 'background',
              1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
              5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
              10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
              14: 'motorbike', 15: 'person', 16: 'pottedplant',
              17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}

# Definirea căilor către fișierele cu prototipul și greutățile modelului
proto = r"C:\Users\andre\Desktop\Cod.sper\MobileNetSSD_deploy.prototxt"
weights = r"C:\Users\andre\Desktop\Cod.sper\MobileNetSSD_deploy.caffemodel"

# Citirea rețelei neuronale din fișierele prototip și greutăți
net = cv2.dnn.readNetFromCaffe(proto, weights)

# Lista cu numele fișierelor de imagine ce vor fi procesate
image_files = ["doggo.jpg", "bicycle.jpg", "motorbike.jpg", "masina.jpg"]

# Listă pentru a stoca toate obiectele detectate
all_detected_objects = []

# Iterare prin fiecare fișier de imagine
for image_file in image_files:
    # Citirea imaginii
    img = cv2.imread(image_file)

    # Redimensionarea imaginii la 300x300 pixeli
    img_resized = cv2.resize(img, (300, 300))

    # Crearea unui blob din imagine pentru a putea fi introdus în rețea
    blob = cv2.dnn.blobFromImage(img_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)

    # Setarea blob-ului ca input pentru rețea
    net.setInput(blob)

    # Obținerea detectărilor de la rețea
    detections = net.forward()

    # Eliminarea dimensiunilor inutile din tensorul de detectări
    final = detections.squeeze()

    # Obținerea dimensiunilor originale ale imaginii
    height, width, _ = img.shape

    # Definirea fontului pentru textul ce va fi afișat pe imagine
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Listă pentru a stoca obiectele detectate în această imagine
    detected_objects = []

    # Iterare prin fiecare detectare
    for i in range(final.shape[0]):
        # Obținerea încrederii pentru detectare
        conf = final[i, 2]

        # Verificarea dacă încrederea este mai mare de 0.5
        if conf > 0.5:
            # Obținerea numelui clasei
            class_name = classNames[int(final[i, 1])]

            # Obținerea coordonatelor normalizate ale boundurilor
            x1n, y1n, x2n, y2n = final[i, 3:7]

            # Transformarea coordonatelor normalizate în coordonate reale
            x1 = int(x1n * width)
            y1 = int(y1n * height)
            x2 = int(x2n * width)
            y2 = int(y2n * height)

            # Definirea punctelor colțului de sus-stânga și jos-dreapta
            top_left = (x1, y1)
            bottom_right = (x2, y2)

            # Desenarea unui dreptunghi pe imagine în jurul obiectului detectat
            img = cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 3)

            # Adăugarea obiectului detectat în lista locală
            detected_objects.append({
                'class_name': class_name,
                'coordinates': {
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2
                }
            })

            # Crearea unui text descriptiv pentru obiect
            text = f'{class_name}: ({x1}, {y1}) - ({x2}, {y2})'

            # Afișarea textului pe imagine
            cv2.putText(img, text, (x1, y1 - 10), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Adăugarea obiectelor detectate din această imagine în lista globală
    all_detected_objects.extend(detected_objects)

    # Afișarea imaginii cu detectările
    cv2.imshow("Detectii", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Salvarea tuturor obiectelor detectate într-un fișier JSON
output_file = "all_detected_objects.json"
with open(output_file, 'w') as f:
    json.dump(all_detected_objects, f)
