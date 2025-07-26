import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([
    transforms.ToTensor(),
])


dataset = datasets.ImageFolder("variations", transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = models.resnet50(pretrained=True)

model.fc = nn.Linear(model.fc.in_features, 12)
model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


num_epochs = 5

for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Epoch [{epoch+1}/{num_epochs}]  Loss: {running_loss:.4f}  Accuracy: {100 * correct / total:.2f}%")

# --- Step 7: Save model ---
os.makedirs("model_output", exist_ok=True)
torch.save(model.state_dict(), "model_output/resnet50_fruit_classifier.pth")
