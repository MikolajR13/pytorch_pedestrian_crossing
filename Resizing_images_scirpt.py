from PIL import Image
import os

# Ścieżki do pięciu wybranych folderów
folder_paths = ['PTL_Dataset_876x657']

for folder_path in folder_paths:
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('jpg', 'JPG', 'png', 'bmp'))]

    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)

        img = Image.open(img_path)

        img_resized = img.resize((224, 224), Image.LANCZOS)

        # Pobieramy nazwę pliku i jego rozszerzenie
        name, ext = os.path.splitext(img_file)

        # Zapisujemy obraz z rozszerzeniem JPG
        img_resized.save(os.path.join('ptl_nowy', f"{name}.jpg"))