import torch

import models.models_vit
import timm

from util.pos_embed import interpolate_pos_embed
from timm.models.layers import trunc_normal_


def load_model(model_name: str):
    if model_name == 'RETFound_cfp':
        model = load_RETFound('cfp')
    elif model_name == 'RETFound_oct':
        model = load_RETFound('oct')
    elif model_name == 'ResNet50':
        model = timm.create_model('resnet50.a1_in1k', pretrained=True)
    elif model_name == 'ResNet50_clear':
        model = timm.create_model('resnet50.a1_in1k', pretrained=False)
    else:
        raise ValueError(f'Unknown model: {model_name}')

    # print("Model = %s" % str(model))
    return model


def load_RETFound(pretraining_dataset):
    model = models.models_vit.__dict__['vit_large_patch16'](num_classes=2, drop_path_rate=0.2, global_pool=True)

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
