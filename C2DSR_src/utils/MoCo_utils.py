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
        if opt['time_encode']:
            if name == "non-overlap":
                ts = batch[6]
            elif name == "overlap":
                ts = batch[9]
        else:
            ts = None
        seq_feat, feat = get_sequence_embedding(opt, target_seq, model.encoder_Y, model.item_emb_Y, projector=None, encoder_causality_mask = False, ts=ts)
        features[index] = feat
        pooling_features[index] = seq_feat
    return pooling_features, features
def compute_embedding_for_female_mixed_user(opt, all_data, model, name):
    model.eval()
    all_data = [[opt["source_item_num"] + opt["target_item_num"]]*(opt["maxlen"] - len(seq[1])) + [ x[0] for x in seq[1]] for seq in all_data]
    item_seq  = torch.tensor(all_data)
    batch_size = 256
    batch_item_seq = [item_seq[i*batch_size:(i+1)*batch_size] for i in range(len(item_seq)//batch_size+1)]
    # print(f"Computing features for {name} user")
    pooling_features = []
    features = []
    for i, item_seq in enumerate(batch_item_seq):
        # pad_len = opt["maxlen"] - len(seq[1])
        # item_seq = [opt["source_item_num"] + opt["target_item_num"]]*pad_len + [ x[0] for x in seq[1]]
        # item_seq  = torch.tensor(item_seq).unsqueeze(0)
        # if opt['time_encode']:
        #     ts = torch.tensor([x[1] for x in seq[1]])
        # else:
            # ts = None
        if opt['cuda']:
            item_seq = item_seq.cuda()
        seq_feat, feat = get_sequence_embedding(opt, item_seq, model.encoder, model.item_emb, projector=None, encoder_causality_mask = False, ts=None)
        features.append(feat)
        pooling_features.append(seq_feat)
    pooling_features = torch.cat(pooling_features).squeeze(1)
    features = torch.cat(features).squeeze(1)
    if opt['cuda']:
        pooling_features = pooling_features.cuda()
        features = features.cuda()
    return pooling_features, features
def compute_features_for_I2C(opt, dataloader, model):
    model.eval()
    max_id = max([k[0] for k in dataloader.all_data])
    features = torch.zeros(max_id+1, opt["hidden_units"]).cuda()
    gender = torch.full((max_id+1,), 2)
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            index = batch[0]
            mixed_seq = batch[1] # no augmentation- cross domain sequence
            ts = batch[7] if opt['time_encode'] else None
            # ipdb.set_trace()
            mixed_seq = mixed_seq.cuda() if opt['cuda'] else mixed_seq
            feat = get_embedding_for_ssl(opt, mixed_seq, model.encoder, model.item_emb, projector=None, encoder_causality_mask = False, ts=ts)
            features[index] = feat
            gender[index] = batch[-1][:,0]
    return features, gender
