import torch
from src.data_loader import get_data_loaders
from src.model import WasteClassifier

def evaluate_model():
    _, test_loader = get_data_loaders()
    model = WasteClassifier(num_classes=3)
    model.load_state_dict(torch.load("models/waste_classifier.pth"))
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print(f"Accuracy: {100 * correct / total:.2f}%")
