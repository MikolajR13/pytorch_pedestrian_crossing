import time as time
import timeit
from time import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
import os
import pandas as pd
from skimage import io
from Data_generator import CrossroadsDataset

# jak to ma działać?
# 1. Dostaje zdjecia/film z apki
# 2. Dziele zdjęcie - 1/3 lub 1/4 crop - stworzyć zmienną do ogarniania tego
# 3. Tnę zdjęcie na 9 sektorów - sprawdzić czy zawsze jest 1 TL 9 BR
# 4. 9 sektorów to.tensor i robie z tego batcha
# 5. Eval
# 6. Wyciągnięcie z tego tych indeksów które zwróciły 1
# 7. Przekazanie ich
# 8. Liczymy czas całej operacji, który przy każdym przejściu pętli też przekazujemy
# 9. Uruchomienie gdy coś się stanie - ktoś kliknie na ekran czy coś
# 10. Zrobić interfejs testowy żebym mógł to przetestować


loaded_model = torch.load('nazwa_modelu')
crop = 3 # = 3 bo 1/3, dla 4 = 1/4

def image_operations(folder_name):
    folder = 'images_test' #wstawić folder_name
    images_paths = os.listdir(folder) #folder_name
    for images in images_paths:  # jeżeli sama klatka będzie przekazywana to bez tego fora - tylko do testów
        if images.endswith(('.png', '.jpg', '.jpeg', '.JPG')):
            image_path = os.path.join(folder, images)

            image = Image.open(image_path)
            width, height = image.size
            cropped_image = image.crop((0, height // crop, width, height))
            pieces = []
            piece_width, piece_height = cropped_image.size
            for i in range(3):
                for j in range(3):
                    left = j * (piece_width // crop)
                    top = i * (piece_height // crop)
                    right = left + (piece_width // crop)
                    bottom = top +( piece_height // crop)
                    piece = cropped_image.crop((left, top, right, bottom))
                    pieces.append(piece)
            transform = transforms.Compose([ transforms.Resize((224, 224)), transforms.ToTensor()])

            transformed_pieces = [transform(piece) for piece in pieces]

            batch_img = torch.stack(transformed_pieces)

    return batch_img


def model_eval(batch_img):
    with torch.no_grad():
        outputs = loaded_model(batch_img)

    predicted = torch.argmax(outputs, dim=1)

    ones = torch.nonzero(predicted == 1).squeeze()

    out = ones.tolist()
    return out


while 1:

    #przyjmowanie obrazu idk czy video czy klatki
    # jeżeli video to trzeba dodać dzielenie na klatki live czy coś
    # można dodać filtry obrazu
    time_start = timeit.timeit()
    grid_numbers = model_eval(image_operations('images_test')) #tutaj przyjmuje klatki - folder tylko do testów
    time_end = timeit.timeit()
    delay = time_end - time_start
    #przekazywanie delayu i tablicy z numerami gridów do podświetlenia

