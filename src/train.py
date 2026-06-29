import torch
import torch.optim as optim
import torch.nn as nn
from src.data_loader import get_data_loaders
from src.model import WasteClassifier

def train_model():
    train_loader, _ = get_data_loaders()
    model = WasteClassifier(num_classes=3)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(5):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    torch.save(model.state_dict(), "models/waste_classifier.pth")
    print("Model saved successfully!")
