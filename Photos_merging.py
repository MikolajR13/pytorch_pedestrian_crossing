import cv2
import os

def merge_images(folder1_path, folder2_path, output_folder):
    # Sprawdzenie i utworzenie folderu wynikowego
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Pobranie listy plików z folderów
    images_folder1 = [f for f in os.listdir(folder1_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    images_folder2 = [f for f in os.listdir(folder2_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    if not images_folder1 or not images_folder2:
        print("Brak obrazów w co najmniej jednym z folderów.")
        return

    # Sortowanie listy plików
    images_folder1.sort()
    images_folder2.sort()
    print(len(images_folder1))
    print(len(images_folder2))

    # Złożenie i zmiana rozmiaru 1000 zdjęć
    for i in range(min(len(images_folder1), len(images_folder2))):
        image1_path = os.path.join(folder1_path, images_folder1[i])
        image2_path = os.path.join(folder2_path, images_folder2[3*i])
        image3_path = os.path.join(folder2_path, images_folder2[3*i + 1])
        image4_path = os.path.join(folder2_path, images_folder2[3*i + 2])

        img1 = cv2.imread(image1_path)
        img2 = cv2.imread(image2_path)
        img3 = cv2.imread(image3_path)
        img4 = cv2.imread(image4_path)
        img1 = cv2.resize(img1, (224, 224))
        img2 = cv2.resize(img2, (224, 224))
        img3 = cv2.resize(img3, (224, 224))
        img4 = cv2.resize(img4, (224, 224))
        if any(img is None for img in [img1, img2, img3, img4]):
            print(f"Nie można wczytać wszystkich obrazów dla iteracji {i + 1}.")
            continue

        if i % 4 == 1:
            img1, img2 = img2, img1
        elif i % 4 == 2:
            img1, img3 = img3, img1
        elif i % 4 == 3:
            img1, img4 = img4, img1

        result = cv2.vconcat([cv2.hconcat([img1, img2]), cv2.hconcat([img3, img4])])

        # Zmiana rozmiaru połączonego obrazu
        resized_result = cv2.resize(result, (224, 224))

        output_path = os.path.join(output_folder, f"merged_image_{i + 1}.jpg")
        cv2.imwrite(output_path, resized_result)

        print(f"Zdjęcia zostały połączone, zmieniono rozmiar i zapisano jako: {output_path}")

    print(f"Stworzono {i + 1} sklejonych i zmienionych rozmiarów zdjęć.")

def main():
    folder1_path = "to_merge_poziom"  # Ścieżka do folderu 1
    folder2_path = "to_merge_nie_pasy"  # Ścieżka do folderu 2
    output_folder = "merged_poziom_nie_uciete"  # Ścieżka do folderu wynikowego

    merge_images(folder1_path, folder2_path, output_folder)

if __name__ == "__main__":
    main()
