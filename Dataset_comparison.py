import os
import pandas as pd

# Ścieżka do folderu z plikami
folder_path = 'dataset'

# Ścieżka do pliku CSV
csv_file = 'nazwy_plikow.csv'

# Pobranie listy plików z folderu
files_in_folder = os.listdir(folder_path)

# Wczytanie pliku CSV
csv_data = pd.read_csv(csv_file)

# Pobranie nazw plików z pierwszej kolumny pliku CSV
files_in_csv = csv_data.iloc[:, 0].tolist()

# Znajdowanie plików, których nie ma w pliku CSV
missing_files = set(files_in_folder) - set(files_in_csv)
missing_files_dataset = set(files_in_csv) - set(files_in_folder)
# Wyświetlenie nazw plików, których brakuje w pliku CSV
if missing_files:
    print("Pliki, których brakuje w pliku CSV:")
    for file in missing_files:
        print(file)
else:
    print("Nie ma brakujących plików w folderze względem pliku CSV.")

if missing_files_dataset:
    print("Pliki, których jest za duzo :")
    for file in missing_files_dataset:
        print(file)
else:
    print("Nie ma brakujących plików w folderze względem pliku CSV.")
