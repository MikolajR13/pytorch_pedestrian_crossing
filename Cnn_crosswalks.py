#importy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from Data_generator import CrossroadsDataset
import matplotlib.pyplot as plt
from torch.utils.mobile_optimizer import optimize_for_mobile


#zmienne
path_to_dataset = 'dataset' #aaa - testowy dataset - żeby sprawdzić czy działa,  dataset - dataset do nauki
batch_size = 64

dataset = CrossroadsDataset(csv_file='nazwy_plikow.csv', root_dir=path_to_dataset,
                            transform=transforms.ToTensor()) # nazwy_plikow1.csv - żeby sprawdzić czy działa,
                                                             # nazwy_plikow.csv - do nauki
size = len(dataset)
val_len = int(size/100*10)
test_len = int(size/100*10)
train_len = size - val_len - test_len

train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_len, val_len, test_len])

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)


class BottleneckLayer(nn.Module):
    def __init__(self, in_channels, expansion, out_channels, stride=1):
        super(BottleneckLayer, self).__init__()

        self.expansion = expansion
        # warstwa rozszerzenia - konwolucyjna 1x1 rozszerza ilość konałów - map cech
        self.conv1 = nn.Conv2d(in_channels, expansion, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False )
        self.bn1 = nn.BatchNorm2d(expansion)
        # warstwa konwolucyjna główna - konwolucja depthwise
        self.conv2 = nn.Conv2d(expansion, expansion, kernel_size=(3, 3), stride=stride, padding=(1, 1),
                               groups=expansion, bias=False)
        self.bn2 = nn.BatchNorm2d(expansion)
        # warstwa redukcji - konwolucyjna 1x1
        self.conv3 = nn.Conv2d(expansion, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1),stride=stride,  bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU6()

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.conv3(y)
        y = self.bn3(y)
        #print(y.size())
        shortcut = self.shortcut(x)
        #print(shortcut.size(), 1)
        y += shortcut
        y = self.relu(y)

        #print(y.size())
        return y


class FirstTransormation(nn.Module):
    def __init__(self, in_channels):
        super(FirstTransormation, self).__init__()

        #normalna konwolucyjna - 3 kanały rgb, wychodzą 32 mapy cech, kernel 3x3 stride 1 czyli przesuwa się co 1 piksel i padding czyli ramka też 1
        self.conv2d = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # redukcja zanikającego gradientu, mniej wrażliwa na wyższe współczynniki uczenia, przyspieszenie uczenia?
        self.bn = nn.BatchNorm2d(32)
        # zamiast RELU bo RELU6 lepiej na mobilnych podobno działa ale ogólnie ograniczenie funkcji aktywacji do przedziału (0, 6)
        self.relu6 = nn.ReLU6()
        # maxpooling - max z kernela 2x2 i o przesunięciu = 2 = dwukrotne zmniejszenie rozmiaru mapy cech ( każdej)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        x = self.relu6(x)
        x = self.maxpool(x)
        return x


class LastLayers(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LastLayers, self).__init__()
        # warstwa konwolucyjna zwiększająca ilość map cech
        self.conv1 = nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=(1, 1), stride=(1, 1),
                               padding=(0, 0), bias=False)
        # batch normalization
        self.bn1 = nn.BatchNorm2d(out_channels)
        #relu 6
        self.relu6 = nn.ReLU6()
        # avg pooling ale imo będzie lepiej działało z maxpoolingiem bo będzie wyciągało najbardziej odznaczające się cechy
        self.avg = nn.AdaptiveAvgPool2d((1, 1)) # można sprawdzić wersję z maxpoolingiem jeszcze

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu6(x)
        x = self.avg(x)

        return x


class Output(nn.Module):
    def __init__(self, in_channels, drop):
        super(Output, self).__init__()
        size = in_channels
        self.fc1 = nn.Linear(size, int(size/4))
        self.fc2 = nn.Linear(int(size/4), int(size/16))
        self.fc3 = nn.Linear(int(size/16), 2)
        self.softmax = nn.Softmax(1)
        self.dropout = nn.Dropout(drop)
        self.relu = nn.ReLU6()

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.softmax(x)

        return x


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        #rozmiar się zmniejsza x2 a dodajemy mapy cech - 32 będą czyli z 3, 224, 224 robi się 32x112x112
        self.input = FirstTransormation(3)
        # 32 wejściowe kanały, expansion = 2 , out_channel = 16 i stride = 1 czyli nie zmniejszamy rozmiaru
        stride = 1
        #wchodzi 32x112x112
        self.Bb1 = BottleneckLayer(32, 6, 16, 1)
        #wchodzi 16x112x112
        self.Bb2 = BottleneckLayer(16, 6, 32, 2)
        #wchodzi 32x112x112
        self.Bb3 = BottleneckLayer(32, 6, 32, 1)
        #wchodzi 64x56x56
        self.Bb4 = BottleneckLayer(32, 6, 64, 2)
        # #wchodzi 128x28x28
        self.Bb5 = BottleneckLayer(64, 6, 64, 1)
        # #wchodzi 64x28x28
        self.Bb6 = BottleneckLayer(64, 6, 128, 2)
        # #wchodzi 128x28x28
        self.Last = LastLayers(128, 986)
        self.Out = Output(986, 0.5)

    def forward(self, x):
        x = self.input(x)
        #print(x.size())
        #print(1)
        x = self.Bb1(x)
        #print(x.size())
        #print(2)
        x = self.Bb2(x)
        #print(x.size())
        #print(3)
        x = self.Bb3(x)
        #print(x.size())
        #print(4)
        x = self.Bb4(x)
        #print(x.size())
        #print(5)
        x = self.Bb5(x)
        #print(x.size())
        #print(6)
        x = self.Bb6(x)
        #print(x.size())
        #print(7)
        x = self.Last(x)
        #print(x.size())
        #print(8)
        x = self.Out(x)
        #print(x.size())
        return x


model = Network()

# inicjalizacja urzadzenia do nauki - cpu lub gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#parametry sieci
in_channel = 3
learning_rate = 0.00001
batch_size = 64
weight_decay = 0.0001
epochs = 5 # default 20 powinno być ale dla testów dałem 100
min_loss = float('inf')
patience = 6
counter = 0

#ładujemy model do urządzenia
model = Network().to(device)

#określenie funkcji straty i optymalizatora

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
#sprawdzanie accuracy


def check_accuracy(loader, model):
    num_correct =0
    num_samples = 0
    loss = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            current_loss = criterion(scores, y)
            loss += current_loss.item()

            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        accuracy = float(num_correct)/float(num_samples)*100
        average_loss = loss / len(loader)

    model.train()
    return average_loss, accuracy


#uczymy sieć

val_losses = []
val_accuracies = []
train_losses = []
train_accuracies = []


for epoch in range(epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)
        # forward
        scores = model(data)
        loss = criterion(scores, targets)
        # backward
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

    model.eval()  # Ustawienie modelu w tryb ewaluacji
    val_loss, val_accuracy = check_accuracy(val_loader, model)  # Ocena na zbiorze walidacyjnym
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    print(f"Epoch [{epoch + 1}/{epochs}] Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
    train_loss, train_accuracy = check_accuracy(train_loader, model)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    print(f"Epoch [{epoch + 1}/{epochs}] Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

    if train_loss < min_loss:
        min_loss = train_loss
        counter = 0
    else:
        counter += 1

    if counter == patience:
        print("Patience break")
        break

    model.train()  # Powrót do trybu treningowego


# save modelu

test_loss, test_acc = check_accuracy(test_loader, model)
print(f" Test Loss: {train_loss:.4f}, Test Accuracy: {train_accuracy:.2f}%")
model.eval()

torch.save(model.state_dict(), 'crosswalks_detection_trained.pth')
example = torch.rand(9, 3, 224, 224)
traced_module = torch.jit.trace(model, example)
traced_module_optimized = optimize_for_mobile(traced_module)
traced_module_optimized._save_for_lite_interpreter("model_z_filmiku.ptl")

#torch.onnx.export(traced_module_optimized, example, 'model_onnx.onnx', export_params=True, opset_version=11)
traced_module_optimized.save('model_torch_script.pt')
# for param in model.parameters():
#     print(1)
# loaded_model = Network()
# loaded_model.load_state_dict(torch.load("crosswalks_detection_test.pth"))
# loaded_model.eval()
# for param in loaded_model.parameters():
#     print(2)

plt.figure(figsize=(12, 4))

# Wykres val_loss i train_loss
plt.subplot(1, 2, 1)
plt.plot(val_losses, label='Validation Loss')
plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation & Train Loss')
plt.legend()

# Wykres val_accuracy i train_accuracy
plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Validation Accuracy')
plt.plot(train_accuracies, label='Train Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Validation & Train Accuracy')
plt.legend()

plt.tight_layout()
plt.show()