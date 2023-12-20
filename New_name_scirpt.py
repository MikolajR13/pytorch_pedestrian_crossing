import os

# Ścieżka do folderu zawierającego zdjęcia
folder_path = 'merged_poziom_nie_uciete'

# Pobranie listy plików z folderu
files = os.listdir(folder_path)

# Nowa nazwa pliku, którą chcesz nadać plikom
new_file_name = 'nowa_nazwa1423_{}.jpg'  # Możesz dostosować format nazwy pliku według potrzeb

# Licznik do numerowania plików
counter = 1

# Iteracja przez wszystkie pliki w folderze i zmiana nazw
for file_name in files:
    # Rozszerzenie pliku
    extension = os.path.splitext(file_name)[1]

    # Tworzenie nowej nazwy pliku
    new_name = new_file_name.format(counter)

    # Pełna ścieżka do pliku przed zmianą nazwy
    old_file_path = os.path.join(folder_path, file_name)

    # Pełna ścieżka do pliku po zmianie nazwy
    new_file_path = os.path.join(folder_path, new_name)

    # Zmiana nazwy pliku
    os.rename(old_file_path, new_file_path)

    # Zwiększenie licznika dla kolejnego pliku
    counter += 1
