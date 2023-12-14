import os
import csv
# Ścieżki do pięciu wybranych folderów
folder_paths = ['images', 'images1', 'images2', 'ptl_nowy', 'aaa', 'klatki']

csv_file = 'nazwy_plikow.csv'
i = 0
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    for folder_path in folder_paths:
        file_names = os.listdir(folder_path)
        for file_name in file_names:
            if i>3:
                number = 0
            else:
                number = 1
            print(i)
            writer.writerow([file_name, number])
        i += 1