import torch
import timm
import torch.nn as nn
import numpy as np
import shap
import pandas as pd
from age_sex_prediction import DKIMDataset
from torch import from_numpy
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm 

MODEL_PATH = "age_classifier_model_resnet50_3e_5k.pth"
IMG_SIZE = 224
SEED = 1234
MAX_EVALS_SHAP = 2000
N_SAMPLES = 6

def age_to_range(age):
    return min((age // 10) - 2, 6)

def f(x):
    x = from_numpy(np.moveaxis(x, 3, 1)).float()
    return model(x)

class AgePredictor(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.head = nn.Linear(num_features, 7) # range 20 - 82 -> 7 classes

    def forward(self, x):
        features = self.base(x)
        return self.head(features)

classes_translated = ['20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89']

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

base_model = timm.create_model('resnet50.a1_in1k', pretrained=False)
num_features = base_model.get_classifier().in_features
base_model.reset_classifier(0)

model = AgePredictor(base_model)
state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()


dataframe_path = 'labels_sex_age.xlsx'
full_df = pd.read_excel(dataframe_path)[18000:18473]
full_df['age'] = full_df['age'].apply(age_to_range)

dataset = DKIMDataset(full_df, transform=transform)

logits_list = []


for idx in tqdm(range(len(dataset))):
    image, gender, age = dataset[idx]
    image = image.unsqueeze(0)
    logits = model(image).detach().numpy().squeeze()
    logits_list.append(logits)


logits_array = np.array(logits_list)
confidence_scores = np.max(np.abs(logits_array), axis=1)
top_indices = np.argsort(confidence_scores)[-N_SAMPLES:][::-1]
print(f"Top {N_SAMPLES} samples selected based on model confidence:\n{top_indices}")


images, ages = zip(*[dataset[idx][:2] for idx in top_indices])
ages = np.array(ages)
images = np.stack(images)
images = np.moveaxis(images, 1, 3)


masker_blur = shap.maskers.Image("blur(128,128)", images[0].shape)
explainer = shap.Explainer(f, masker_blur, output_names=classes_translated)

shap_values = explainer(images, max_evals=MAX_EVALS_SHAP, batch_size=8,
                        outputs=shap.Explanation.argsort.flip[:7])

shap_values.data = shap_values.data
shap_values.values = [val for val in np.moveaxis(shap_values.values, -1, 0)]

shap.image_plot(shap_values=shap_values.values,
                pixel_values=shap_values.data,
                labels=shap_values.output_names, show=False,
                true_labels=[l for l in np.array(classes_translated)[ages[:N_SAMPLES]]])

plt.savefig('shap_plot.png')
plt.close()
print('SHAP plot saved as shap_plot.png')