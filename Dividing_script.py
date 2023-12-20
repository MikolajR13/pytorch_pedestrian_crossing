import os
from PIL import Image

# Ścieżka do folderu zawierającego zdjęcia
input_folder_path = 'na_pol_pol'

# Ścieżka do folderu, gdzie zostaną zapisane nowe zdjęcia
output_folder_path = 'na_pol_rdy'

# Sprawdzenie i utworzenie folderu wyjściowego, jeśli nie istnieje
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# Pobranie listy plików z folderu wejściowego
input_files = os.listdir(input_folder_path)
i = 0
for file_name in input_files:
    # Ładowanie obrazu

    img = Image.open(os.path.join(input_folder_path, file_name))
    width, height = img.size

    # Odcinanie górnej połowy
    upper_half = img.crop((0, 0, width, height // 2))

    # Dzielenie dolnej połowy na dwie części (lewa i prawa)
    lower_half_left = img.crop((0, height // 2, width // 2, height))
    lower_half_right = img.crop((width // 2, height // 2, width, height))

    # Zapisywanie dwóch nowych zdjęć
    lower_half_left.save(os.path.join(output_folder_path, f"{i}_left.jpg"))
    lower_half_right.save(os.path.join(output_folder_path, f"{i}_right.jpg"))
    i += 1
