import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import RepeatedStratifiedKFold
import timm
from tqdm import tqdm
import pandas as pd
import pydicom
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from util.pos_embed import interpolate_pos_embed
from timm.models.layers import trunc_normal_
import models.models_vit

NUM_EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-4
NUM_FOLDS = 2
NUM_REPEATS = 1
SEED = 1234
IMG_SIZE = 224
SUMMARY_LOG = "results_5x1.txt"
GENDER_MODEL_SAVE_NAME = "gender_classifier_model"
AGE_MODEL_SAVE_NAME = "age_classifier_model"
NUMBER_OF_RECORDS = 5000
TRAIN_RATIO = 0.8


def load_RETFound(pretraining_dataset):
    model = models.models_vit.__dict__['vit_large_patch16'](num_classes=2, drop_path_rate=0.2, global_pool="avg")

    checkpoint = torch.load(f'weights/RETFound_{pretraining_dataset}_weights.pth', map_location='cpu')
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    interpolate_pos_embed(model, checkpoint_model)
    msg = model.load_state_dict(checkpoint_model, strict=False)

    assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}

    trunc_normal_(model.head.weight, std=2e-5)
    return model


torch.manual_seed(SEED)
np.random.seed(SEED)

# base_model = timm.create_model('resnet50.a1_in1k', pretrained=True)
base_model = load_RETFound('cfp')

# num_features = base_model.get_classifier().in_features
# base_model.reset_classifier(0)


class GenderClassifierRETFound(nn.Module):
    def __init__(self, vit_model):
        super().__init__()
        self.backbone = vit_model
        self.backbone.head = nn.Identity()  # remove classification head
        self.fc_norm = self.backbone.fc_norm  # optional, used in ViT
        self.head = nn.Linear(1024, 1)  # binary classification

    def forward(self, x):
        x = self.backbone.forward_features(x)  # [B, 1024]
        x = self.fc_norm(x)
        return self.head(x)

    def extract_features(self, x):
        x = self.backbone.forward_features(x)
        x = self.fc_norm(x)
        return x

# class GenderClassifierRETFound(nn.Module):
#     def __init__(self, vit_model):
#         super().__init__()
#         self.backbone = vit_model
#         self.backbone.head = nn.Identity()  # remove classification head
#         self.fc_norm = self.backbone.fc_norm  # reuse LayerNorm
#         self.head = nn.Linear(1024, 1)  # or 2 for multiclass

#     def forward(self, x):
#         x = self.backbone.forward_features(x)  # should return [B, 1024]
#         x = self.fc_norm(x)
#         return self.head(x)
    
class GenderClassifierResnet(nn.Module):
    def __init__(self, base_model, num_features):
        super().__init__()
        self.base = base_model
        self.head = nn.Linear(num_features, 1)

    def forward(self, x):
        features = self.base(x)
        return self.head(features).squeeze(1)  # logits

    def extract_features(self, x):
        return self.base(x)  # feature vector before classification


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
            img = (img * 255).astype(np.uint8)

            if len(img.shape) == 2:
                img = np.stack([img] * 3, axis=-1)

            img = Image.fromarray(img)

            if self.transform:
                img = self.transform(img)

            return img, gender, _

        except Exception as e:
            print(f"Skipping corrupt DICOM: {dcm_path} — {str(e)}")
            exit()


transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])


def train_gender():
    dataframe_path = 'labels_sex_age.xlsx'
    # limit to 10k images
    full_df = pd.read_excel(dataframe_path)[:NUMBER_OF_RECORDS]

    # 1 -- changed to --> 0    -- man
    # 2 -- changed to --> 1    -- woman
    full_df['gender'] = full_df['gender'].replace({1: 0, 2: 1})
    full_df['age_range'] = full_df['age'].apply(age_to_range)

    rskf = RepeatedStratifiedKFold(n_splits=NUM_FOLDS, n_repeats=NUM_REPEATS, random_state=SEED)

    train_losses = []
    val_losses = []
    all_accuracies = []
    all_precisions = []
    all_recalls = []
    all_f1s = []

    for fold, (train_idx, val_idx) in enumerate(rskf.split(full_df, full_df['gender'])):
        print(f"\nFold {fold + 1}/{NUM_FOLDS*NUM_REPEATS}")
        train_df, val_df = train_test_split(full_df, train_size=TRAIN_RATIO, stratify=full_df['gender'] ,random_state=SEED)

        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

        train_dataset = DKIMDataset(train_df, transform=transform)
        val_dataset = DKIMDataset(val_df, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = GenderClassifierRETFound(base_model)
        #  to be removed
        # state_dict = torch.load('gender_classifier_model_10e_16000.pth')
        # model.load_state_dict(state_dict)
        # to be removed

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        for epoch in range(NUM_EPOCHS):
            model.train()
            total_loss = 0
            for images, genders, age in tqdm(train_loader, desc=f"Train Epoch {epoch+1}", leave=False):
                preds = model(images)
                loss = criterion(preds.squeeze(1), genders)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            print(f"Fold {fold+1} | Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f}")

            model.eval()
            val_loss = 0
            all_preds = []
            all_targets = []

            with torch.no_grad():
                for images, genders, age in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}", leave=False):
                    preds = model(images)
                    loss = criterion(preds.squeeze(1), genders)
                    val_loss += loss.item()

                    probs = torch.sigmoid(preds)
                    binary_preds = (probs > 0.5).int().cpu().numpy()
                    targets = genders.int().cpu().numpy()

                    all_preds.extend(binary_preds)
                    all_targets.extend(targets)

            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            acc = accuracy_score(all_targets, all_preds)
            prec = precision_score(all_targets, all_preds, zero_division=0)
            rec = recall_score(all_targets, all_preds, zero_division=0)
            f1 = f1_score(all_targets, all_preds, zero_division=0)

            all_accuracies.append(acc)
            all_precisions.append(prec)
            all_recalls.append(rec)
            all_f1s.append(f1)

            print(f"Fold {fold+1} | Epoch {epoch+1} | Val Loss: {avg_val_loss:.4f}")
            print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
        
        torch.save(model.state_dict(), f"{GENDER_MODEL_SAVE_NAME}_{NUM_EPOCHS}e_{NUMBER_OF_RECORDS}.pth")
        print(f"\nModel weights saved to: {GENDER_MODEL_SAVE_NAME}")
        break

    summary = []
    summary.append("\n=== Summary over all folds ===")
    summary.append(f"Epochs: {NUM_EPOCHS}, Batch Size: {BATCH_SIZE}, Number of Folds: {NUM_FOLDS}")
    summary.append(f"Train Losses per fold: {train_losses}")
    summary.append(f"Val Losses per fold: {val_losses}")
    summary.append(f"Accuracy: {all_accuracies}")
    summary.append(f"Precision: {all_precisions}")
    summary.append(f"Recall: {all_recalls}")
    summary.append(f"F1: {all_f1s}")
    summary.append(f"Average Accuracy: {np.mean(all_accuracies):.4f}")
    summary.append(f"Average Precision: {np.mean(all_precisions):.4f}")
    summary.append(f"Average Recall: {np.mean(all_recalls):.4f}")
    summary.append(f"Average F1: {np.mean(all_f1s):.4f}")

    for line in summary:
        print(line)

    SUMMARY_LOG = "results_e10_gender_resnet50.txt"

    with open(SUMMARY_LOG, "a") as f:
        for line in summary:
            f.write(line + "\n")


def age_to_range(age):
    return min((age // 10) - 2, 6)

def train_age_pred():
    dataframe_path = 'labels_sex_age.xlsx'
    full_df = pd.read_excel(dataframe_path)[:NUMBER_OF_RECORDS]

    # Convert age to age ranges
    full_df['age_range'] = full_df['age'].apply(age_to_range)

    rskf = RepeatedStratifiedKFold(n_splits=NUM_FOLDS, n_repeats=NUM_REPEATS, random_state=SEED)

    train_losses = []
    val_losses = []
    all_accuracies = []
    all_precisions = []
    all_recalls = []
    all_f1s = []

    for fold, (train_idx, val_idx) in enumerate(rskf.split(full_df, full_df['age_range'])):
        print(f"\nFold {fold + 1}/{NUM_FOLDS*NUM_REPEATS}")
        train_df, val_df = train_test_split(full_df, train_size=TRAIN_RATIO, stratify=full_df['age_range'] ,random_state=SEED)

        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

        # train_df = full_df.iloc[train_idx].reset_index(drop=True)
        # val_df = full_df.iloc[val_idx].reset_index(drop=True)

        train_dataset = DKIMDataset(train_df, transform=transform)
        val_dataset = DKIMDataset(val_df, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = AgePredictor(base_model)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        for epoch in range(NUM_EPOCHS):
            model.train()
            total_loss = 0
            for images, gender, age in tqdm(train_loader, desc=f"Train Epoch {epoch+1}", leave=False):
                preds = model(images)
                loss = criterion(preds, age)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            print(f"Fold {fold+1} | Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f}")

            model.eval()
            val_loss = 0
            all_preds = []
            all_targets = []

            with torch.no_grad():
                for images, gender, age in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}", leave=False):
                    preds = model(images)
                    loss = criterion(preds, age)
                    val_loss += loss.item()

                    preds = torch.argmax(preds, dim=1).cpu().numpy()
                    targets = age.cpu().numpy()

                    all_preds.extend(preds)
                    all_targets.extend(targets)

            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            acc = accuracy_score(all_targets, all_preds)
            prec = precision_score(all_targets, all_preds, average='weighted',zero_division=0)
            rec = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
            f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)

            all_accuracies.append(acc)
            all_precisions.append(prec)
            all_recalls.append(rec)
            all_f1s.append(f1)

            print(f"Fold {fold+1} | Epoch {epoch+1} | Val Loss: {avg_val_loss:.4f}")
            print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

        torch.save(model.state_dict(), f"{AGE_MODEL_SAVE_NAME}_{NUM_EPOCHS}e_{NUMBER_OF_RECORDS}.pth")
        print(f"\nModel weights saved to: {AGE_MODEL_SAVE_NAME}")
        break

    summary = []
    summary.append("=== Summary over all folds ===")
    summary.append(f"Epochs: {NUM_EPOCHS}, Batch Size: {BATCH_SIZE}, Number of Folds: {NUM_FOLDS}")
    summary.append(f"Train Losses per fold: {train_losses}")
    summary.append(f"Val Losses per fold: {val_losses}")
    summary.append(f"Accuracy: {all_accuracies}")
    summary.append(f"Precision: {all_precisions}")
    summary.append(f"Recall: {all_recalls}")
    summary.append(f"F1: {all_f1s}")
    summary.append(f"Average Accuracy: {np.mean(all_accuracies):.4f}")
    summary.append(f"Average Precision: {np.mean(all_precisions):.4f}")
    summary.append(f"Average Recall: {np.mean(all_recalls):.4f}")
    summary.append(f"Average F1: {np.mean(all_f1s):.4f}")

    for line in summary:
        print(line)

    SUMMARY_LOG = "results_e10_age_resnet50.txt"

    with open(SUMMARY_LOG, "a") as f:
        for line in summary:
            f.write(line + "\n")

if __name__ == "__main__":
    train_gender()
