import cv2
import os


def split_and_resize(image_path, output_folder):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Nie można wczytać obrazu: {image_path}")
        return False

    height, width = image.shape[:2]
    if height < 224 * 3 or width < 224 * 3:
        print(f"Obraz {image_path} jest za mały, aby podzielić na 9 części 224x224.")
        return False

    step_height = height // 3
    step_width = width // 3
    count = 1

    for y in range(0, height, step_height):
        for x in range(0, width, step_width):
            cropped = image[y:y + step_height, x:x + step_width]
            resized = cv2.resize(cropped, (224, 224))

            output_path = os.path.join(output_folder,
                                       f"{os.path.splitext(os.path.basename(image_path))[0]}_part_{count}.jpg")
            cv2.imwrite(output_path, resized)
            count += 1

    return True


def process_images_in_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    images = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    if not images:
        print("Brak obrazów w folderze.")
        return

    for image_name in images:
        image_path = os.path.join(input_folder, image_name)
        success = split_and_resize(image_path, output_folder)
        if success:
            print(f"Zdjęcie {image_name} zostało pomyślnie przetworzone.")

    print(f"Wszystkie zdjęcia zostały przetworzone. Zapisano w folderze: {output_folder}")


def main():
    input_folder = "klatki_nie1"  # Ścieżka do folderu ze zdjęciami
    output_folder = "klatki_nie11"  # Ścieżka do folderu docelowego

    process_images_in_folder(input_folder, output_folder)


if __name__ == "__main__":
    main()
