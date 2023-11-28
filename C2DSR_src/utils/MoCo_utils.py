import torch
from tqdm import tqdm
import ipdb
from .torch_utils import *

def compute_features(opt, eval_loader, model, domain="X"):
    print(f"Computing features for domain {domain}")
    model.eval()
    features = torch.zeros(eval_loader.num_examples, opt["hidden_units"]).cuda()
    for i, batch in enumerate(eval_loader):
        index = batch[0]
        if domain == "X":
            augmented_data = batch[27]
            # augmented_data  = batch[2] # no augmentation - x domain sequence
            ts = batch[8]
        elif domain == "Y":
            augmented_data = batch[28]
            # augmented_data  = batch[3] # no augmentation- x domain sequence
            ts = batch[9]
        elif domain == "mixed":
            augmented_data = batch[1] # no augmentation- cross domain sequence
            ts = batch[7]
        # ipdb.set_trace()
        with torch.no_grad():
            augmented_data = augmented_data.cuda()
            
            if domain == "mixed":
                feat = get_embedding_for_ssl(opt, augmented_data, model.encoder, model.item_emb, projector=None, encoder_causality_mask = False, ts=ts)
            else:
                if augmented_data.ndim == 2:
                    augmented_data = augmented_data.unsqueeze(1)
                feat = model(augmented_data[:, 0, :], is_eval=True,ts=ts)
            features[index] = feat
    return features.cpu()
