import os
import csv
# Ścieżki do pięciu wybranych folderów
folder_paths = ['bez_pasow_1', 'bez_pasow_3', 'bez_pasow_4', 'pasy_1', 'pasy_2', 'pasy_3', 'pasy_4', 'pasy_5', 'pasy_6']

csv_file = 'nazwy_plikow.csv'
i = 0
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    for folder_path in folder_paths:
        file_names = os.listdir(folder_path)
        for file_name in file_names:
            if i>2:
                number = 1
            else:
                number = 0
            print(i)
            writer.writerow([file_name, number])
        i += 1