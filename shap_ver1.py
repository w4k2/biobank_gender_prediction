import torch
import torch
import timm
import torch.nn as nn
import numpy as np
import shap
import pandas as pd
from age_sex_prediction import DKIMDataset
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import RepeatedStratifiedKFold
from torch import from_numpy, argmax
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

MODEL_PATH = "age_classifier_model_resnet50_3e_5k.pth" 
NUM_FOLDS = 2
NUM_REPEATS = 1
NUM_EPOCHS = 10
BATCH_SIZE = 4
IMG_SIZE = 224
SEED = 1234
MAX_EVALS_SHAP = 20

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


# classes_translated = ['Male', 'Female']
classes_translated = ['20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89']

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

base_model = timm.create_model('resnet50.a1_in1k', pretrained=False)
num_features = base_model.get_classifier().in_features
base_model.reset_classifier(0)


dataframe_path = 'labels_sex_age.xlsx'
full_df = pd.read_excel(dataframe_path)[:100]

# Gender
# # 1 -- changed to --> 0    -- man
# # 2 -- changed to --> 1    -- woman
# full_df['age'] = full_df['age'].replace({1: 0, 2: 1})

full_df['age'] = full_df['age'].apply(age_to_range)

model = AgePredictor(base_model)
state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()

rskf = RepeatedStratifiedKFold(n_splits=NUM_FOLDS, n_repeats=NUM_REPEATS, random_state=SEED)

for train_idx, val_idx in rskf.split(full_df, full_df['age']):
    val_df = full_df.iloc[val_idx].reset_index(drop=True)
    val_dataset = DKIMDataset(val_df, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    n_samples = 3

    for images, gender, age in val_loader:
        # Logits
        # logits = model(images)
        # probs = torch.nn.functional.softmax(-logits, dim=1).detach().numpy()
        # output_names = np.argsort(probs).astype("U")
        # output_names[output_names=='0'] = "Male"
        # output_names[output_names=='1'] = "Female"

        images = torch.moveaxis(images, 1, 3)

        masker_blur = shap.maskers.Image("blur(128,128)", images[0].shape)
        explainer = shap.Explainer(f, masker_blur, output_names=classes_translated)

        shap_values = explainer(images[:n_samples], max_evals=MAX_EVALS_SHAP, batch_size=8,
                                outputs=shap.Explanation.argsort.flip[:7])

        shap_values.data = shap_values.data.cpu().numpy()
        shap_values.values = [val for val in np.moveaxis(shap_values.values,-1, 0)]

        shap.image_plot(shap_values=shap_values.values,
                        pixel_values=shap_values.data,
                        labels=shap_values.output_names, show=False,
                        true_labels=[l for l in np.array(classes_translated)[age[:n_samples].int()]])

        print([l for l in np.array(classes_translated)[age[:n_samples].int()]])
        print(shap_values.output_names)

        # plt.tight_layout()
        plt.savefig('shap_plot.png')
        plt.close()
        print('SHAP plot saved as shap_plot.png')
        exit()