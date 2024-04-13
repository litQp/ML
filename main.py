import os
import librosa
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import math


def load_audio_files_from_dir(directory):
    audio_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".wav") or file.endswith(".mp3"):
                audio_files.append(os.path.join(root, file))
    return audio_files


def audio_to_melspectrogram(audio_file):
    y, sr = librosa.load(audio_file, sr=None)  # Load the full audio file
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_DB = librosa.power_to_db(S, ref=np.max)
    return S_DB.astype(np.float32)

def pad_sequences(sequences):
    max_len = max(seq.shape[1] for seq in sequences)
    padded_sequences = np.zeros((len(sequences), sequences[0].shape[0], max_len))
    for i, seq in enumerate(sequences):
        padded_sequences[i, :, :seq.shape[1]] = seq
    return padded_sequences

def create_dataset_and_labels(audio_files):
    X, y = [], []
    for file_path, label in audio_files:
        try:
            spectrogram = audio_to_melspectrogram(file_path)
            X.append(spectrogram)
            y.append(label)
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")
    X = pad_sequences(X)
    X = np.array(X)[:, np.newaxis, :, :]  # Ensure channel dimension is added
    return X, np.array(y)


# Define dataset directories
directory = r"/content/drive/MyDrive/DEMONSTRATION/KAGGLE/AUDIO"
fake_dir = r"/content/drive/MyDrive/DEMONSTRATION/KAGGLE/AUDIO/FAKE"
real_dir = r"/content/drive/MyDrive/DEMONSTRATION/KAGGLE/AUDIO/REAL"
# Исправленное списковое включение
fake_audio_files = [(f, 0) for f in load_audio_files_from_dir(r"/content/drive/MyDrive/DEMONSTRATION/KAGGLE/AUDIO/FAKE")]
real_audio_files = [(f, 1) for f in load_audio_files_from_dir(r"/content/drive/MyDrive/DEMONSTRATION/KAGGLE/AUDIO/REAL")]  # Assuming label 1 for real

# Prepare datasets
all_audio_files = fake_audio_files + real_audio_files
X, y = create_dataset_and_labels(all_audio_files)
print("Shape of X with channel dimension:", X.shape)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors and create DataLoader
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# Neural network architecture
class AudioClassifier(nn.Module):
    def __init__(self):
        super(AudioClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        # Dynamically calculate the flattened size
        dummy_input = torch.zeros(1, 1, 128, X.shape[-1])  # Use the maximum sequence length
        dummy_output = self.forward_features(dummy_input)
        num_features = dummy_output.numel()
        self.fc1 = nn.Linear(num_features, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward_features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = AudioClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training and validation
for epoch in range(20):
    model.train()
    for data, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(data.float())
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, targets in val_loader:
            outputs = model(data.float())
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = 100 * correct / total
    print(f'Epoch {epoch+1}, Loss: {loss.item()}, Validation Accuracy: {accuracy:.2f}%')