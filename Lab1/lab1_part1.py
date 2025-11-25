import os
import tarfile
import urllib.request
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import re
from collections import Counter
from tqdm import tqdm


# CONFIGURATION
MAX_VOCAB_SIZE = 5000  # Top 5000 mots
MAX_SEQ_LEN = 200  # Longueur max d'une review
BATCH_SIZE = 50
EMBEDDING_DIM = 100  # Taille des vecteurs de mots
HIDDEN_DIM = 256
NUM_CLASSES = 2
NUM_EPOCHS = 5
LR = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Utilisation du device : {device}")


# 1. TÉLÉCHARGEMENT & CHARGEMENT DES DONNÉES
def download_and_load_imdb():
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    filename = "aclImdb_v1.tar.gz"
    dirname = "aclImdb"

    # 1. Télécharger
    if not os.path.exists(dirname):
        if not os.path.exists(filename):
            print("Téléchargement du dataset IMDB (80MB)... Cela peut prendre une minute.")
            urllib.request.urlretrieve(url, filename)

        print("Extraction des fichiers...")
        with tarfile.open(filename, 'r:gz') as tar:
            if hasattr(tarfile.TarFile, 'extraction_filter'):
                tar.extractall(filter='data')
            else:
                tar.extractall()

    # 2. Lire les fichiers texte
    print("Lecture des données...")
    reviews = []
    labels = []

    for split in ['train', 'test']:
        for sentiment in ['pos', 'neg']:
            path = os.path.join(dirname, split, sentiment)
            files = os.listdir(path)
            for fname in tqdm(files, desc=f"Lecture {split}/{sentiment}"):
                with open(os.path.join(path, fname), 'r', encoding='utf-8') as f:
                    reviews.append(f.read())
                    labels.append(1 if sentiment == 'pos' else 0)

    return reviews, np.array(labels)

# Exécution du chargement
raw_reviews, raw_labels = download_and_load_imdb()
print(f"Nombre total de reviews chargées : {len(raw_reviews)}")


# 2. PRÉTRAITEMENT
def tokenizer(text):
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower().split()

print("Construction du vocabulaire...")
all_words = []
for review in raw_reviews[:10000]:
    all_words.extend(tokenizer(review))

count_words = Counter(all_words)
sorted_words = count_words.most_common(MAX_VOCAB_SIZE)
vocab_to_int = {w: i + 1 for i, (w, c) in enumerate(sorted_words)}  # 0 = padding

print("Encodage des reviews (Tokenization)...")
reviews_int = []
for review in raw_reviews:
    r = [vocab_to_int.get(w, 0) for w in tokenizer(review)]
    reviews_int.append(r)

# Padding
print("Padding des séquences...")
features = np.zeros((len(reviews_int), MAX_SEQ_LEN), dtype=int)
for i, row in enumerate(reviews_int):
    if len(row) > 0:
        features[i, -len(row):] = np.array(row)[:MAX_SEQ_LEN]

# Création des Tenseurs
# 80% pour Train, 20% pour Val
split_idx = int(len(features) * 0.8)
train_x = torch.from_numpy(features[:split_idx]).to(device)
val_x = torch.from_numpy(features[split_idx:]).to(device)
train_y = torch.from_numpy(raw_labels[:split_idx]).long().to(device)
val_y = torch.from_numpy(raw_labels[split_idx:]).long().to(device)

train_data = TensorDataset(train_x, train_y)
val_data = TensorDataset(val_x, val_y)

train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)



# 3. MODÈLE LSTM
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(LSTMClassifier, self).__init__()
        # Embedding : transforme les entiers en vecteurs denses
        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=0)
        # LSTM
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        # Dropout pour éviter l'overfitting
        self.dropout = nn.Dropout(0.5)
        # Couche Fully Connected
        self.fc = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        embeds = self.embedding(x)
        # LSTM renvoie (output, (hidden, cell))
        _, (hidden, _) = self.lstm(embeds)
        # On prend le dernier état caché
        last_hidden = hidden[-1]
        out = self.dropout(last_hidden)
        return self.fc(out)

model = LSTMClassifier(len(vocab_to_int), EMBEDDING_DIM, HIDDEN_DIM, NUM_CLASSES)
model = model.to(device)


# 4. ENTRAÎNEMENT
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

train_losses = []
val_accuracies = []

print("\nDébut de l'entraînement...")

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    for inputs, labels in loop:
        inputs, labels = inputs.to(device), labels.to(device)  # N'oublie pas le device !
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            output = model(inputs)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    val_accuracies.append(acc)
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {acc:.2f}%")


# 5. GRAPHIQUES
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

# Loss
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss', color='red')
plt.title('Courbe de Perte (Loss)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Validation Accuracy', color='blue')
plt.title("Précision de Validation (Accuracy)")
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.grid(True)

plt.tight_layout()
plt.savefig('LSTM_IMDB_resultats.png')
print("\nGraphique sauvegardé sous 'LSTM_IMDB_resultats.png'. À insérer dans ton rapport LaTeX.")
plt.show()