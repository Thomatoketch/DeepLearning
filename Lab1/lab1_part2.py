import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# 1. CONFIGURATION & GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Utilisation du device : {device}")

BATCH_SIZE = 64
EPOCHS = 5


# 2. PRÉPARATION DES DONNÉES (Fashion-MNIST)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalisation standard
])

print("Téléchargement de Fashion-MNIST...")
train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

classes = ('T-shirt', 'Pantalon', 'Pull', 'Robe', 'Manteau', 'Sandale', 'Chemise', 'Basket', 'Sac', 'Bottine')


# 3. ARCHITECTURE LENET
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 1 canal entrée (gris) -> 6 canaux sortie, noyau 5x5
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.AvgPool2d(2, 2)
        # 6 entrées -> 16 sorties, noyau 5x5
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Fully connected layers
        # Image 28x28 -> Conv1(24x24) -> Pool(12x12) -> Conv2(8x8) -> Pool(4x4)
        # Donc 16 canaux * 4 * 4
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Conv1 -> ReLU -> Pool
        x = self.pool(torch.relu(self.conv1(x)))
        # Conv2 -> ReLU -> Pool
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 4. FONCTION D'ENTRAÎNEMENT GÉNÉRIQUE
def train_and_evaluate(optimizer_name, learning_rate=0.001):
    print(f"\n--- Entraînement avec {optimizer_name} ---")

    model = LeNet().to(device)
    criterion = nn.CrossEntropyLoss()

    if optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    losses = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        # Barre de progression tqdm
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader)
        losses.append(epoch_loss)

    # Évaluation finale
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Précision finale ({optimizer_name}): {acc:.2f}%")

    return losses, model


# ==========================================
# 5. EXÉCUTION & COMPARAISON
# ==========================================
if __name__ == '__main__':
    # Entraînement SGD
    loss_sgd, model_sgd = train_and_evaluate("SGD")

    # Entraînement Adam
    loss_adam, model_adam = train_and_evaluate("Adam")

    # Plot Comparaison
    plt.figure(figsize=(10, 5))
    plt.plot(loss_sgd, label='SGD', marker='o')
    plt.plot(loss_adam, label='Adam', marker='o')
    plt.title('Comparaison de Convergence: SGD vs Adam')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('LeNet_optimizer_comparison.png')
    print("Graphique de comparaison sauvegardé.")


    # 6. VISUALISATION DES FILTRES
    def visualize_filters(model):
        filters = model.conv1.weight.data.cpu()

        # Normalisation entre 0 et 1 pour l'affichage
        f_min, f_max = filters.min(), filters.max()
        filters = (filters - f_min) / (f_max - f_min)

        plt.figure(figsize=(10, 4))
        for i in range(6):
            plt.subplot(1, 6, i + 1)
            plt.imshow(filters[i][0], cmap='gray')
            plt.axis('off')
            plt.title(f'Filtre {i + 1}')

        plt.suptitle('Filtres appris (Couche Conv1)')
        plt.savefig('LeNet_filters.png')
        print("Filtres sauvegardés.")


    # On visualise les filtres du modèle Adam (souvent plus nets)
    visualize_filters(model_adam)
    plt.show()