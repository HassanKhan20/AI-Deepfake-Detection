import torch
import torchvision.transforms as transforms
from PIL import Image

from app.models.deepfake_model import DeepfakeClassifier


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'backend/model_weights/deepfake_classifier.pth'

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def build_model():
    model = DeepfakeClassifier()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    model.to(DEVICE)
    return model


_model = None


def get_model():
    global _model
    if _model is None:
        _model = build_model()
    return _model


def predict(image: Image.Image):
    model = get_model()
    tensor = _transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=-1).squeeze(0)

    fake_score = float(probs[1])
    real_score = float(probs[0])
    label = 'fake' if fake_score > real_score else 'real'

    return {
        'label': label,
        'fake_confidence': fake_score,
        'real_confidence': real_score,
    }
