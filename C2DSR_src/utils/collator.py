from argparse import Namespace
import torch
from utils.augmentation import *
import ipdb
from sklearn.preprocessing import StandardScaler
import pandas as pd
import json
from utils.torch_utils import get_item_embedding_for_sequence
class CLDataCollator:
    def __init__(self, opt, eval, mixed_generator,attribute_predictor=None,model=None) -> None:
        self.opt = opt
        self.augmentation = {"dropout": None,
                            'crop' : Crop(tao = opt["crop_prob"])}
        self.transform = self.augmentation[opt["augment_type"]]
        self.eval = eval
        self.mixed_generator = mixed_generator
        self.attribute_predictor = attribute_predictor
        self.model = model
    def __call__(self, batch):
        if self.eval == -1: #for training
            augmented_d = self.augment(batch[1],all_ts = batch[7], gender=batch[-1]) #[B,2,max_len]
            # augmented_xd = self.augment(batch[26])
            # augmented_yd = self.augment(batch[27])     
            return (torch.LongTensor(batch[0]), torch.LongTensor(batch[1]), torch.LongTensor(batch[2]), torch.LongTensor(batch[3]),torch.LongTensor(batch[4]), torch.LongTensor(batch[5]), torch.LongTensor(batch[6]), torch.LongTensor(batch[7]),torch.LongTensor(batch[8]), torch.LongTensor(batch[9]), torch.LongTensor(batch[10]), torch.LongTensor(batch[11]), torch.LongTensor(batch[12]), torch.LongTensor(batch[13]), torch.LongTensor(batch[14]), torch.LongTensor(batch[15]), torch.LongTensor(batch[16]), torch.LongTensor(batch[17]),\
                        torch.LongTensor(batch[18]),torch.LongTensor(batch[19]),torch.LongTensor(batch[20]),torch.LongTensor(batch[21]),torch.LongTensor(batch[22]),torch.LongTensor(batch[23]),torch.LongTensor(batch[24]),torch.LongTensor(batch[25]),torch.LongTensor(batch[26]), torch.LongTensor(augmented_d),torch.LongTensor(batch[28]))
        elif self.eval == 2: #for  validation
            augmented_d = self.augment(batch[0])
            augmented_xd = self.augment(batch[1])
            augmented_yd = self.augment(batch[2])
            return (torch.LongTensor(batch[0]), torch.LongTensor(batch[1]), torch.LongTensor(batch[2]), torch.LongTensor(batch[3]),torch.LongTensor(batch[4]), torch.LongTensor(batch[5]), torch.tensor(batch[6]), torch.tensor(batch[7]),torch.tensor(batch[8]), torch.LongTensor(batch[9]),\
                    torch.LongTensor(batch[10]),torch.LongTensor(batch[11]),torch.LongTensor(batch[12]),torch.LongTensor(batch[13]), torch.LongTensor(batch[14]), torch.LongTensor(batch[15]), torch.LongTensor(batch[16]), torch.LongTensor(batch[17]), torch.LongTensor(batch[18]), torch.LongTensor(augmented_d), torch.LongTensor(augmented_xd), torch.LongTensor(augmented_yd))
    def generate(self,seq, positions, timestamp, female_IR):
        
        torch_seq = torch.LongTensor(seq) #[X,max_len]
        mask = torch_seq == self.opt['itemnum']
        torch_position = torch.LongTensor(positions) #[X,max_len]
        torch_ts = torch.LongTensor(timestamp) #[X,max_len]
        self.mixed_generator.eval()
        if self.opt['cuda']:
            self.mixed_generator =  self.mixed_generator.cuda()
            torch_seq = torch_seq.cuda()
            torch_position = torch_position.cuda()
            torch_ts = torch_ts.cuda()
        with torch.no_grad():
            if self.opt['time_encode']:
                seq_fea =  self.mixed_generator(torch_seq,torch_position,torch_ts) #[X,max_len,item_num]
            else:
                seq_fea =  self.mixed_generator(torch_seq,torch_position)
            target_fea = seq_fea[mask]
            target_fea /= torch.max(target_fea, dim=1, keepdim=True)[0]
            probabilities = torch.nn.functional.softmax(target_fea, dim=1)
            sampled_indices = torch.multinomial(probabilities, 10, replacement=False).squeeze() #[X,10]
            ### insert highest female interaction ratio item ### 
            mapping = torch.zeros(self.opt['source_item_num']+self.opt['target_item_num'],dtype=torch.float32).cuda()
            for k in female_IR.keys():
                mapping[int(k)] = female_IR[k]
            selected_idx_index = mapping[sampled_indices].argmax(dim=-1)
            sampled_indices = sampled_indices[torch.arange(len(sampled_indices)),selected_idx_index]
            if type =="Y":
                sampled_indices = sampled_indices + self.opt['source_item_num']
            torch_seq[mask] = sampled_indices
            new_seq = torch_seq.tolist() 
       
        # new_seq = [(g,[[x,ts] for x,ts in zip(sublist,timestamp) if x != self.opt['source_item_num'] + self.opt['target_item_num']]) for g, sublist, timestamp in zip(gender, new_seq, torch_ts.tolist())]
        return new_seq
        
    def augment(self,item_seqs, all_ts, gender):
        with open(f"./fairness_dataset/Movie_lens_time/{self.opt['data_dir']}/male_IR.json","r") as f:
            male_IR = json.load(f)
        with open(f"./fairness_dataset/Movie_lens_time/{self.opt['data_dir']}/female_IR.json","r") as f:
            female_IR = json.load(f)
        ratio = self.opt['substitute_ratio']
        augmented_d = []
        for _ in range(2):
            new_seqs = []
            new_positions = []
            for i in range(len(item_seqs)):
                item_seq = np.array(item_seqs[i].copy())
                ts= all_ts[i]
                # item_IR = {item:male_IR[str(item)] if str(item) in male_IR.keys() else 0 for item in item_seq}
                item_IR = {item:female_IR[str(item)] if str(item) in female_IR.keys() else 0 for item in item_seq}
                item_seq_len = len(item_seq[item_seq!=(self.opt['source_item_num'] + self.opt['target_item_num'])])
                
                ###find male item to substitute with attention weight###
                if self.attribute_predictor is not None and self.model is not None and self.opt['substitute_mode'] == "attention_weight":
                    item_seq = torch.tensor(item_seq).unsqueeze(0)
                    item_seq = item_seq.cuda() if self.opt['cuda'] else item_seq
                    seqs_fea = get_item_embedding_for_sequence(self.opt,item_seq,self.model.encoder,self.model.item_emb, self.model.CL_projector,encoder_causality_mask = False,cl =False)
                    _, weight = self.attribute_predictor(seqs_fea)
                    weight = weight.squeeze()
                    item_seq= item_seq.squeeze()
                    mask = item_seq!=(self.opt['source_item_num'] + self.opt['target_item_num'])
                    idx = torch.nonzero(mask)[0]
                    sorted_idx = torch.argsort(weight[idx:])[-item_seq_len//2:]#取最高的一半 attention weight
                    values = weight[idx:][sorted_idx] 
                    #sample item from the highest 50% attention weight items
                    probabilities = torch.nn.functional.softmax(values, dim=0)
                    sampled_pos = torch.multinomial(probabilities, max(int(item_seq_len * ratio),1), replacement=False).squeeze()
                    seleted_idx = sorted_idx[sampled_pos]
                    selected_item = item_seq[idx:][seleted_idx].cpu().detach().numpy()
                    item_seq = item_seq.cpu().detach().numpy()
                    try:
                        substitute_idxs = np.where(item_seq ==selected_item[:, None])[1] #more than one item
                    except:
                        substitute_idxs = np.where(item_seq ==selected_item)[0] #only one item
                elif self.opt['substitute_mode'] =="IR":
                ###find male item to substitute with IR###
                    # choose the highest 50% male IR items
                    # pair = dict(sorted(item_IR.items(), key=lambda item: item[1], reverse=True)[:item_seq_len//2]) #取male IR最高的一半
                    pair = dict(sorted(item_IR.items(), key=lambda item: item[1], reverse=True)[item_seq_len//2:]) #取female IR最低的一半
                    values = np.array(list(pair.values()))
                    # sample item from the highest 50% male IR items
                    probabilities = np.exp(values - np.max(values)) / np.sum(np.exp(values - np.max(values)))
                    sampled_pos = np.random.choice(len(values), size=max(int(item_seq_len * ratio), 1), replace=False, p=probabilities)
                    selected_item = [k for i,(k,v) in enumerate(pair.items()) if i in sampled_pos]
                    substitute_idxs = np.where(item_seq == np.array(selected_item)[:, None])[1]
                else:
                    raise ValueError("substitute_mode should be attention_weight or IR")
                for index in substitute_idxs:
                    item_seq[index] = self.opt['itemnum']
                position = [0]*(self.opt['maxlen']-item_seq_len) + list(range(item_seq_len+1)[1:])[-self.opt['maxlen']:]
                new_seq = item_seq.tolist()[-self.opt['maxlen']:]
                new_seqs.append(new_seq)
                new_positions.append(position)
            augmented_seq= self.generate(new_seqs, new_positions, ts, female_IR)
            augmented_d.append(augmented_seq)
            # augmented_xd.append([[i for i in seq if i < self.opt['source_item_num']] for seq in new_seq])
            # augmented_yd.append([[i for i in seq if i >= self.opt['source_item_num']] for seq in new_seq])
        return augmented_d
class GDataCollator:
    def __init__(self, opt) -> None:
        self.opt = opt
        
    def __call__(self, batch):
        seq,target_sentences, masked_seqs, neg_seqs = self.augment(batch[1])    
        # ipdb.set_trace()
        return (torch.LongTensor(batch[0]), torch.LongTensor(seq), torch.LongTensor(batch[2]), torch.LongTensor(batch[3]),\
                torch.LongTensor(batch[4]),torch.LongTensor(batch[5]),torch.LongTensor(masked_seqs),torch.LongTensor(neg_seqs),torch.LongTensor(target_sentences),torch.LongTensor(batch[6]))
                        
    def augment(self,item_seqs):
        max_len = self.opt["maxlen"]
        target_sentences = []
        masked_seqs = []
        neg_seqs = []
        d = []
        for seq in item_seqs:
            target_sentence = []
            masked_seq = []
            neg_seq = []
            #確保至少mask一個X或Y item
            last_x = [s for s in seq if s < self.opt["source_item_num"]][-1]
            x_idx = seq.index(last_x)
            last_y = [s for s in seq if s >= self.opt["source_item_num"]][-1]
            y_idx = seq.index(last_y)
            for i in seq:
                if self.opt["generate_type"] == "X":
                    condition = i < self.opt["source_item_num"]
                    
                elif self.opt["generate_type"] == "Y":
                    condition = (i >= self.opt["source_item_num"]) and (i!= self.opt["source_item_num"] + self.opt["target_item_num"])
                else:
                    condition = i!= self.opt["source_item_num"] + self.opt["target_item_num"]
                if condition and (random.random() < self.opt['mask_prob']):
                    masked_seq.append(self.opt["source_item_num"] + self.opt["target_item_num"]+1)
                    neg = random.sample(set(range(self.opt['source_item_num']+self.opt['target_item_num'])) - set(seq),1)[0]
                    neg_seq.append(neg)
                    target_sentence.append(i)
                else:
                    masked_seq.append(i)
                    neg_seq.append(i)
                    target_sentence.append(self.opt["source_item_num"] + self.opt["target_item_num"])
            if self.opt["generate_type"] == "mixed":
                masked_seq[-1] = self.opt["source_item_num"] + self.opt["target_item_num"]+1
                neg_seq[-1] = random.sample(set(range(self.opt['source_item_num']+self.opt['target_item_num'])) - set(seq),1)[0]
                target_sentence[-1] = seq[-1]
            elif self.opt["generate_type"] == "X":
                masked_seq[x_idx] = self.opt["source_item_num"] + self.opt["target_item_num"]+1
                neg_seq[x_idx] = random.sample(set(range(self.opt['source_item_num']+self.opt['target_item_num'])) - set(seq),1)[0]
                target_sentence[x_idx] = seq[x_idx]
            elif self.opt["generate_type"] == "Y":
                masked_seq[y_idx] = self.opt["source_item_num"] + self.opt["target_item_num"]+1
                neg_seq[y_idx] = random.sample(set(range(self.opt['source_item_num']+self.opt['target_item_num'])) - set(seq),1)[0]
                target_sentence[y_idx] = seq[y_idx]
            
            masked_seq = [self.opt["source_item_num"] + self.opt["target_item_num"]]*(max_len - len(masked_seq)) + masked_seq
            neg_seq = [self.opt["source_item_num"] + self.opt["target_item_num"]]*(max_len - len(neg_seq)) + neg_seq
            target_sentence = [self.opt["source_item_num"] + self.opt["target_item_num"]]*(max_len - len(target_sentence)) + target_sentence
            seq = [self.opt["source_item_num"] + self.opt["target_item_num"]]*(max_len - len(seq)) + seq
            d.append(seq)
            target_sentences.append(target_sentence)
            masked_seqs.append(masked_seq)
            neg_seqs.append(neg_seq)
        return d,target_sentences, masked_seqs, neg_seqs
