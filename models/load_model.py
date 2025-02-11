import torch
from models.cnn_model import EmotionCNN

def load_trained_model(model_path = r"C:\Users\rayan\Desktop\IdeaProjects\EmotionDetector\models\emotion_modelV3.pth"):
    model = EmotionCNN(num_classes = 7)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model