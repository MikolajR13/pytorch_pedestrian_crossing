import timeit
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
class BottleneckLayer(nn.Module):
    def __init__(self, in_channels, expansion, out_channels, stride=1):
        super(BottleneckLayer, self).__init__()

        self.expansion = expansion
        # warstwa rozszerzenia - konwolucyjna 1x1 rozszerza ilość konałów - map cech
        self.conv1 = nn.Conv2d(in_channels, expansion, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False )
        self.bn1 = nn.BatchNorm2d(expansion)
        # warstwa konwolucyjna główna - konwolucja depthwise
        self.conv2 = nn.Conv2d(expansion, expansion, kernel_size=(3, 3), stride=stride, padding=(1, 1),
                               groups=expansion, bias=False)
        self.bn2 = nn.BatchNorm2d(expansion)
        # warstwa redukcji - konwolucyjna 1x1
        self.conv3 = nn.Conv2d(expansion, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1),stride=stride,  bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU6()

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.conv3(y)
        y = self.bn3(y)
        #print(y.size())
        shortcut = self.shortcut(x)
        #print(shortcut.size(), 1)
        y += shortcut
        y = self.relu(y)

        #print(y.size())
        return y


class FirstTransormation(nn.Module):
    def __init__(self, in_channels):
        super(FirstTransormation, self).__init__()

        #normalna konwolucyjna - 3 kanały rgb, wychodzą 32 mapy cech, kernel 3x3 stride 1 czyli przesuwa się co 1 piksel i padding czyli ramka też 1
        self.conv2d = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # redukcja zanikającego gradientu, mniej wrażliwa na wyższe współczynniki uczenia, przyspieszenie uczenia?
        self.bn = nn.BatchNorm2d(32)
        # zamiast RELU bo RELU6 lepiej na mobilnych podobno działa ale ogólnie ograniczenie funkcji aktywacji do przedziału (0, 6)
        self.relu6 = nn.ReLU6()
        # maxpooling - max z kernela 2x2 i o przesunięciu = 2 = dwukrotne zmniejszenie rozmiaru mapy cech ( każdej)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        x = self.relu6(x)
        x = self.maxpool(x)
        return x


class LastLayers(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LastLayers, self).__init__()
        # warstwa konwolucyjna zwiększająca ilość map cech
        self.conv1 = nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=(1, 1), stride=(1, 1),
                               padding=(0, 0), bias=False)
        # batch normalization
        self.bn1 = nn.BatchNorm2d(out_channels)
        #relu 6
        self.relu6 = nn.ReLU6()
        # avg pooling ale imo będzie lepiej działało z maxpoolingiem bo będzie wyciągało najbardziej odznaczające się cechy
        self.avg = nn.AdaptiveAvgPool2d((1, 1)) # można sprawdzić wersję z maxpoolingiem jeszcze

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu6(x)
        x = self.avg(x)

        return x


class Output(nn.Module):
    def __init__(self, in_channels, drop):
        super(Output, self).__init__()
        size = in_channels
        self.fc1 = nn.Linear(size, int(size/4))
        self.fc2 = nn.Linear(int(size/4), int(size/16))
        self.fc3 = nn.Linear(int(size/16), 2)
        self.softmax = nn.Softmax(1)
        self.dropout = nn.Dropout(drop)
        self.relu = nn.ReLU6()

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.softmax(x)

        return x


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        #rozmiar się zmniejsza x2 a dodajemy mapy cech - 32 będą czyli z 3, 224, 224 robi się 32x112x112
        self.input = FirstTransormation(3)
        # 32 wejściowe kanały, expansion = 2 , out_channel = 16 i stride = 1 czyli nie zmniejszamy rozmiaru
        stride = 1
        #wchodzi 32x112x112
        self.Bb1 = BottleneckLayer(32, 6, 16, 1)
        #wchodzi 16x112x112
        self.Bb2 = BottleneckLayer(16, 6, 32, 2)
        #wchodzi 32x112x112
        self.Bb3 = BottleneckLayer(32, 6, 32, 1)
        #wchodzi 64x56x56
        self.Bb4 = BottleneckLayer(32, 6, 64, 2)
        # #wchodzi 128x28x28
        self.Bb5 = BottleneckLayer(64, 6, 64, 1)
        # #wchodzi 64x28x28
        self.Bb6 = BottleneckLayer(64, 6, 128, 2)
        # #wchodzi 128x28x28
        self.Last = LastLayers(128, 986)
        self.Out = Output(986, 0.5)

    def forward(self, x):
        x = self.input(x)
        #print(x.size())
        #print(1)
        x = self.Bb1(x)
        #print(x.size())
        #print(2)
        x = self.Bb2(x)
        #print(x.size())
        #print(3)
        x = self.Bb3(x)
        #print(x.size())
        #print(4)
        x = self.Bb4(x)
        #print(x.size())
        #print(5)
        x = self.Bb5(x)
        #print(x.size())
        #print(6)
        x = self.Bb6(x)
        #print(x.size())
        #print(7)
        x = self.Last(x)
        #print(x.size())
        #print(8)
        x = self.Out(x)
        #print(x.size())
        return x




# inicjalizacja urzadzenia do nauki - cpu lub gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loaded_model = Network()
loaded_model.load_state_dict(torch.load('crosswalks_detection_test.pth', map_location=device))
loaded_model.eval()
for param in loaded_model.parameters():
    print(1)
crop = 3 # = 3 bo 1/3, dla 4 = 1/4


def image_operations(folder_name):
    folder = 'wtf' #wstawić folder_name
    images_paths = os.listdir(folder) #folder_name
    i = 0
    for images in images_paths:  # jeżeli sama klatka będzie przekazywana to bez tego fora - tylko do testów
        print(images)
        time_start = timeit.timeit()
        if images.endswith(('.png', '.jpg', '.jpeg', '.JPG')):

            image_path = os.path.join(folder, images)
            image = Image.open(image_path)
            width, height = image.size
            cropped_image = image.crop((0, height // crop, width, height))
            pieces = []
            k = 0
            piece_width, piece_height = cropped_image.size
            os.mkdir(f"wyniki/{images}")
            for i in range(3):
                for j in range(3):
                    left = j * (piece_width // crop)
                    top = i * (piece_height // crop)
                    right = left + (piece_width // crop)
                    bottom = top +( piece_height // crop)
                    piece = cropped_image.crop((left, top, right, bottom))
                    piece.save(os.path.join(f"wyniki/{images}", f"{k}.jpg"))
                    k += 1
                    pieces.append(piece)
            transform = transforms.Compose([ transforms.Resize((224, 224)), transforms.ToTensor()])
            transformed_pieces = [transform(piece) for piece in pieces]
            batch_img = torch.stack(transformed_pieces)
        with torch.no_grad():
            outputs = loaded_model(batch_img)
        predicted = torch.argmax(outputs, dim=1)
        print(predicted)
        ones = predicted.tolist()
        time_end = timeit.timeit()
        time = time_end - time_start
        print(time, "pierwszy")
        time = time_start - time_end
        print(time, "drugi")
        file_path = f"wyniki/{images}/{images}.txt"

        # Zapis przewidywanych wartości do pliku tekstowego
        with open(file_path, 'w') as file:
            for value in ones:
                file.write(f"{value}\n")
    return 1

    #przyjmowanie obrazu idk czy video czy klatki
    # jeżeli video to trzeba dodać dzielenie na klatki live czy coś
    # można dodać filtry obrazu
image_operations('images_test')
    #time_start = timeit.timeit()
    #grid_numbers = model_eval(image_operations('images_test')) #tutaj przyjmuje klatki - folder tylko do testów
    #time_end = timeit.timeit()
    #delay = time_end - time_start
    #przekazywanie delayu i tablicy z numerami gridów do podświetlenia

