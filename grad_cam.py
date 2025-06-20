import timm
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, BinaryClassifierOutputTarget, RawScoresOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
from age_sex_prediction import GenderClassifierRETFound, GenderClassifierResnet, load_RETFound
from pytorch_grad_cam.utils.image import preprocess_image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import pydicom
from PIL import Image
from cv2 import resize
import torchvision.transforms as transforms


RESNET_MODEL_PATH = "weights/gender_classifier_model_resnet50_15e_16000.pth"
RETFOUND_MODEL_PATH = "weights/gender_classifier_model_retfound_10e_16000.pth"


class DKIMDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        dcm_path = row['filepath']
        gender = torch.tensor(row['gender'], dtype=torch.float32)  # 0 or 1
        # age = torch.tensor(row['age_range'], dtype=torch.long)
        _ = None

        try:
            dcm = pydicom.dcmread(dcm_path)
            img = dcm.pixel_array.astype(np.float32)
            img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-5)
            img = (img * 255).astype(np.float32)

            if len(img.shape) == 2:
                img = np.stack([img] * 3, axis=-1)

            # img = Image.fromarray(img)

            if self.transform:
                img = self.transform(img)

            return img, gender, _

        except Exception as e:
            print(f"Error {e}")
            exit()


def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


##### Model part

model = timm.create_model('resnet50.a1_in1k', pretrained=False)
num_features = model.get_classifier().in_features
model.reset_classifier(0)
model = GenderClassifierResnet(model, num_features)

state_dict = torch.load(RESNET_MODEL_PATH, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()
target_layers = [model.base.layer4[-1]]


# base_model = load_RETFound('cfp')
# model = GenderClassifierRETFound(base_model)
# state_dict = torch.load(RETFOUND_MODEL_PATH, map_location=torch.device('cpu'))
# model.load_state_dict(state_dict)
# model.eval()
# target_layers = [model.backbone.blocks[23].norm1]


#####
# ViT: model.blocks[-1].norm1
# Resnet: model.layer4[-1]

dataframe_path = 'labels_sex_age.xlsx'
full_df = pd.read_excel(dataframe_path)[15000:]
full_df['gender'] = full_df['gender'].replace({1: 0, 2: 1})
dataset = DKIMDataset(full_df, transform=None)

top_indices = np.array([771, 1389,  540, 1754,  905, 2979,  494, 2077, 2014])
images, gender, ages = zip(*[dataset[idx][:3] for idx in top_indices])
gender = np.array([g.item() for g in gender])

IMAGE_FROM_TOP_INDICES_INDEX = 7
CLASS = 0

#  [0. 0. 0. 1. 1. 1. 1. 1. 1.]
print(gender)

rgb_img = resize(images[IMAGE_FROM_TOP_INDICES_INDEX], (224, 224))
rgb_img = np.float32(rgb_img) / 255
input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
print(gender[IMAGE_FROM_TOP_INDICES_INDEX])


targets = [BinaryClassifierOutputTarget(CLASS)]

# RETFound
# with GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform) as cam:

# ResNet50
with GradCAM(model=model, target_layers=target_layers) as cam:
    grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
    cam_image = show_cam_on_image(rgb_img, grayscale_cams[0, :], use_rgb=True)

    out = cam.outputs
    probs = torch.nn.functional.softmax(out, dim=1).detach().numpy()
    print(probs)

plt.imshow(cam_image)
plt.savefig(f"gradcam_resnet_finetuned_{IMAGE_FROM_TOP_INDICES_INDEX}_{CLASS}.png")