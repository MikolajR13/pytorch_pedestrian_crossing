from PIL import Image
import os

# Ścieżki do pięciu wybranych folderów
folder_paths = ['poziome_wsz', 'pasy', 'bez_pasow']

for folder_path in folder_paths:
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('jpg', 'JPG', 'png', 'bmp'))]
    print(image_files)
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)

        img = Image.open(img_path)

        img_resized = img.resize((224, 224), Image.LANCZOS)

        # Pobieramy nazwę pliku i jego rozszerzenie
        name, ext = os.path.splitext(img_file)

        # Zapisujemy obraz z rozszerzeniem JPG
        img_resized.save(os.path.join('dataset', f"{name}.jpg"))