import torch
import os
import pandas as pd
from skimage import io
from torch.utils.data import Dataset


#ładowanie danych
# do jednego folderu ładujemy wszystkie zdjęcia i indeksujemy je - jest przejście/nie ma przejścia
# indeksowanie można zrobić za pomocą skryptu w pythonie, który wyciąga nam z naszych dwóch folderów - nie ma przejścia i jest przejście
# nazwy klatek, i załóżmy dla przejścia daje 1 a nie dla przejścia 0 - pewnie da się to jeszcze automatycznie do excela wrzucić

class CrossroadsDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file) #ile jest danych np. 25000
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)  #na przykład 25000

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)