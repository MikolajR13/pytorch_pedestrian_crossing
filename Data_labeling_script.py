import os
import csv
# Ścieżki do pięciu wybranych folderów
folder_paths = ['bez_pasow', 'pasy', 'poziome_wsz']

csv_file = 'nazwy_plikow.csv'
i = 0
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    for folder_path in folder_paths:
        file_names = os.listdir(folder_path)
        for file_name in file_names:
            if i>0:
                number = 1
            else:
                number = 0
            print(i)
            writer.writerow([file_name, number])
        i += 1


