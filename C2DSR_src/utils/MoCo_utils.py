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
def compute_embedding_for_target_user(opt, dataloader, model, name):
    print(f"Computing features for {name} user")
    # model.eval()
    # ipdb.set_trace()
    if name == "non-overlap":
        max_id = max([k[0] for k in dataloader.all_data])
    elif name == "overlap":
        max_id = max([k[0] for k in dataloader.all_data])
    
    pooling_features = torch.zeros(max_id+1, opt["hidden_units"])
    features = torch.zeros(max_id+1, opt["maxlen"], opt["hidden_units"])
    if opt['cuda']:
        features = features.cuda()
        pooling_features = pooling_features.cuda()
    for i, batch in enumerate(dataloader):
        index = batch[0]
        target_seq = batch[1] # no augmentation- cross domain sequence
        if opt['cuda']:
            target_seq = target_seq.cuda()
        seq_feat, feat = get_sequence_embedding(opt, target_seq, model.encoder_Y, model.item_emb_Y, projector=None, encoder_causality_mask = False, ts=None)
        # ipdb.set_trace()
        features[index] = feat
        pooling_features[index] = seq_feat
    return pooling_features, features
