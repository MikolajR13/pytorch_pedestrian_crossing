import cv2
import os
import random
import numpy as np


def rotate_image(image, angle):
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image


def adjust_brightness(image, factor):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    v_channel = v_channel.astype(float)
    v_channel *= factor / 100.0
    v_channel = np.clip(v_channel, 0, 255)
    hsv[:, :, 2] = v_channel.astype(np.uint8)
    brightened_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return brightened_image


def add_noise(image, a):
    if a % 6 == 0:
        noise = np.random.normal(0,20, image.shape)
        noisy_image = np.clip(image + noise, 0, 255).astype(np.int32)
    else:
        noise = np.random.normal(0, 30, image.shape)
        noisy_image = np.clip(image + noise, 0, 255).astype(np.int32)
    return noisy_image


def add_shadow(image, i):
    print(i)
    if i % 3 == 0:
        shadow_mask = np.zeros_like(image[:, :, 0])
        shadow_mask[:, :image.shape[1]] = 255  # Dodanie zacienienia w lewej połowie obrazu
        shadow_intensity = random.randint(75, 200)  # Intensywność zacienienia
        shadow = np.where(shadow_mask == 255, shadow_intensity, 0)
        image[:, :, 2] += shadow  # Zmniejszenie wartości kanału niebieskiego dla zacienienia
    elif i % 3 == 1:
        shadow_mask = np.zeros_like(image[:, :, 1])
        shadow_mask[:, :image.shape[1]] = 255  # Dodanie zacienienia w lewej połowie obrazu
        shadow_intensity = random.randint(40, 110)  # Intensywność zacienienia
        shadow = np.where(shadow_mask == 255, shadow_intensity, 0)
        image[:, :, 1] -= shadow  # Zmniejszenie wartości kanału niebieskiego dla zacienienia
    elif i % 3 == 2:
        shadow_mask = np.zeros_like(image[:, :, 0])
        shadow_mask[:, :image.shape[1]] = 255  # Dodanie zacienienia w lewej połowie obrazu
        shadow_intensity = random.randint(40, 110)  # Intensywność zacienienia
        shadow = np.where(shadow_mask == 255, shadow_intensity, 0)
        image[:, :, 0] += shadow  # Zmniejszenie wartości kanału niebieskiego dla zacienienia
    return image


def change_perspective(image, scale):
    height, width = image.shape[:2]
    pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    pts2 = np.float32([[0, 0], [width * scale, 0], [0, height], [width * (1 - scale), height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    perspective_image = cv2.warpPerspective(image, matrix, (width, height))
    return perspective_image


def main():
    folder_path = "bez_pasow_2"
    output_path = "bez_pasow_3"
    images = os.listdir(folder_path)

    if not images:
        print("Brak zdjęć w folderze.")
        return
    i = 0
    a = 0
    for image_name in images:
        image_path = os.path.join(folder_path, image_name)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Nie można wczytać obrazu: {image_name}")
            continue
        #perspective_scale = random.uniform(0.5, 0.6)
        #perspective_image = change_perspective(image, perspective_scale)
        rotation_angle = random.randint(10, 75)
        rotated_image = rotate_image(image, rotation_angle)

        brightness_factor = random.randint(60, 80)
        adjusted_image = adjust_brightness(rotated_image, brightness_factor)
        if i % 6 == 0 :
            noisy_image = add_noise(adjusted_image, i)
            shadowed_image = add_shadow(noisy_image, a)
            transpose = cv2.transpose(shadowed_image, 3)
            flipped = cv2.flip(transpose, 1)
            output_image_path = os.path.join(output_path, f"modified_{image_name}")
            cv2.imwrite(output_image_path, flipped)
            a += 1
        else:
            noisy_image = add_noise(adjusted_image, i)
            transpose = cv2.transpose(noisy_image, 3)
            flipped = cv2.flip(transpose, 1)
            output_image_path = os.path.join(output_path, f"modified_{image_name}")
            cv2.imwrite(output_image_path, flipped)
        i += 1

    print(f"Zmodyfikowane zdjęcia zapisano w folderze: {output_path}")


if __name__ == "__main__":
    main()
