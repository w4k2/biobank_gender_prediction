import torch
import torch
import timm
import torch.nn as nn

MODEL_PATH = "gender_classifier_model.pth" 

base_model = timm.create_model('resnet50.a1_in1k', pretrained=False)
num_features = base_model.get_classifier().in_features
base_model.reset_classifier(0)


class GenderClassifier(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.head = nn.Linear(num_features, 1)

    def forward(self, x):
        features = self.base(x)
        return self.head(features).squeeze(1)



model = GenderClassifier(base_model)
state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()

print("Model loaded successfully!")