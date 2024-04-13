import os
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


def load_audio_files_from_dir(directory, duration=3):
    audio_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                audio_files.append((file_path, duration))  # Передаем длительность аудио
    return audio_files


def trim_audio_file(audio_file, target_duration=3):
    waveform, sample_rate = librosa.load(audio_file, sr=None)
    target_length = int(target_duration * sample_rate)
    
    # Если длина волновой формы меньше целевой, добавить нули в конец
    if len(waveform) < target_length:
        waveform = librosa.util.fix_length(waveform, size = target_length)
    # Если длина волновой формы больше целевой, обрезать до нужной длины
    elif len(waveform) > target_length:
        waveform = waveform[:target_length]
    
    return waveform, sample_rate

def trim_audio_files_in_directory(directory, target_duration=3):
    trimmed_waveforms = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".wav"):
                audio_file = os.path.join(root, file)
                waveform, sample_rate = trim_audio_file(audio_file, target_duration)
                trimmed_waveforms.append((waveform, sample_rate))
    return trimmed_waveforms

# Пример использования
directory = r"C:\Users\valera\Desktop\python\goo\speech" # место расположения 
trimmed_waveforms = trim_audio_files_in_directory(directory)



# путь к файлам
audio_files = load_audio_files_from_dir(directory)
fake_dir = r"C:\Users\valera\Desktop\python\goo\speech\fake"
real_dir = r"C:\Users\valera\Desktop\python\goo\speech\real"

# загрузка файлов
fake_audio_files = load_audio_files_from_dir(fake_dir)
real_audio_files = load_audio_files_from_dir(real_dir)


fake_labels = [0] * len(fake_audio_files)  
real_labels = [1] * len(real_audio_files)  


all_audio_files = fake_audio_files + real_audio_files
all_labels = fake_labels + real_labels

def create_dataset_and_labels(audio_files):
    X = []
    y = []
    for file_path, duration in audio_files:
        waveform, _ = trim_audio_file(file_path, duration)  # Используем trim_audio_file для обрезки аудио
        X.append(waveform)
        label = 0 if "fake" in file_path else 1  # Пример: использование имени папки для определения класса
        y.append(label)
    return np.array(X), np.array(y)



X, y = create_dataset_and_labels(all_audio_files)


train_size = int(0.8 * len(X))
val_size = int(0.1 * len(X))
test_size = len(X) - train_size - val_size

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

# Преобразовать данные в тензоры PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Создать Dataset для каждого набора данных
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Создать DataLoader для каждого набора данных
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class AudioClassifier(nn.Module):
    def __init__(self):
        super(AudioClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 2)  # 2 класса: реальные и поддельные аудиофайлы

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Создайте экземпляр модели
model = AudioClassifier()

# Определите функцию потерь и оптимизатор
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение модели
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # Установите модель в режим обучения
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Оценка производительности модели на валидационном наборе данных
    model.eval()  # Установите модель в режим оценки
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch + 1}/{num_epochs}], Тест1: {accuracy:.2f}%')

# Оценка производительности модели на тестовом наборе данных
model.eval()  # Установите модель в режим оценки
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Тест2: {accuracy:.2f}%')

