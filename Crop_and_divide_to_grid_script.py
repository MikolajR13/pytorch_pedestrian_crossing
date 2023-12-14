from PIL import Image
import time

def crop_and_divide_image(image_path):
    try:
        # Wczytaj obraz
        img = Image.open(image_path)

        # Pobierz wymiary obrazu
        width, height = img.size

        # Ucinamy 1/4 od góry
        cropped_img = img.crop((0, height // 4, width, height))

        # Dziel pozostałe 3/4 na 9 równych kawałków
        new_width, new_height = cropped_img.size
        tile_width = new_width // 3
        tile_height = new_height // 3

        divided_images = []
        for y in range(0, new_height, tile_height):
            for x in range(0, new_width, tile_width):
                box = (x, y, x + tile_width, y + tile_height)
                divided_images.append(cropped_img.crop(box))

        return divided_images

    except Exception as e:
        print("Wystąpił błąd:", e)
        return None


image_path = 'heon_IMG_0542.jpg'

start_time = time.time()


result = crop_and_divide_image(image_path)

end_time = time.time()

if result:
    print(f"Podzielono obraz na {len(result)} części.")
    print(f"Czas wykonania operacji: {end_time - start_time:.4f} sekund.")
    for i, img_part in enumerate(result):
        img_part.save(f"czesc_{i+1}.jpg")  # Zapis do pliku jako czesc_1.jpg, czesc_2.jpg, itd.
else:
    print("Nie udało się podzielić obrazu.")