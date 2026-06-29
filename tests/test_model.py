import torch
from src.model import WasteClassifier

def test_model_forward():
    model = WasteClassifier(num_classes=3)
    sample_input = torch.randn(1, 3, 128, 128)
    output = model(sample_input)
    assert output.shape[1] == 3
