from argparse import Namespace
import torch
from utils.augmentation import *
import ipdb
from sklearn.preprocessing import StandardScaler
import pandas as pd
class CLDataCollator:
    def __init__(self, opt, eval) -> None:
        self.opt = opt
        self.augmentation = {"dropout": None,
                            'crop' : Crop(tao = opt["crop_prob"])}
        self.transform = self.augmentation[opt["augment_type"]]
        self.eval = eval
        
    def __call__(self, batch):
        if self.eval == -1: #for training
            augmented_d = self.augment(batch[1])
            augmented_xd = self.augment(batch[26])
            augmented_yd = self.augment(batch[27])     
            return (torch.LongTensor(batch[0]), torch.LongTensor(batch[1]), torch.LongTensor(batch[2]), torch.LongTensor(batch[3]),torch.LongTensor(batch[4]), torch.LongTensor(batch[5]), torch.LongTensor(batch[6]), torch.LongTensor(batch[7]),torch.LongTensor(batch[8]), torch.LongTensor(batch[9]), torch.LongTensor(batch[10]), torch.LongTensor(batch[11]), torch.LongTensor(batch[12]), torch.LongTensor(batch[13]), torch.LongTensor(batch[14]), torch.LongTensor(batch[15]), torch.LongTensor(batch[16]), torch.LongTensor(batch[17]),\
                        torch.LongTensor(batch[18]),torch.LongTensor(batch[19]),torch.LongTensor(batch[20]),torch.LongTensor(batch[21]),torch.LongTensor(batch[22]),torch.LongTensor(batch[23]),torch.LongTensor(batch[24]),torch.LongTensor(batch[25]),torch.LongTensor(augmented_d),torch.LongTensor(augmented_xd),torch.LongTensor(augmented_yd))
        elif self.eval == 2: #for  validation
            augmented_d = self.augment(batch[0])
            augmented_xd = self.augment(batch[1])
            augmented_yd = self.augment(batch[2])
            return (torch.LongTensor(batch[0]), torch.LongTensor(batch[1]), torch.LongTensor(batch[2]), torch.LongTensor(batch[3]),torch.LongTensor(batch[4]), torch.LongTensor(batch[5]), torch.tensor(batch[6]), torch.tensor(batch[7]),torch.tensor(batch[8]), torch.LongTensor(batch[9]),\
                    torch.LongTensor(batch[10]),torch.LongTensor(batch[11]),torch.LongTensor(batch[12]),torch.LongTensor(batch[13]), torch.LongTensor(batch[14]), torch.LongTensor(batch[15]), torch.LongTensor(batch[16]), torch.LongTensor(batch[17]), torch.LongTensor(batch[18]), torch.LongTensor(augmented_d), torch.LongTensor(augmented_xd), torch.LongTensor(augmented_yd))
    def augment(self,item_seqs):
        max_len = self.opt["maxlen"]
        augmented_seqs = []
        if self.transform is None:
            output_item_seqs = []
            for item_seq in item_seqs:
                pad_len = max(0,max_len - len(item_seq))
                if self.opt["is_pooling"] and self.opt['pooling']=="bert":  
                    if len(item_seq)>=max_len:
                        item_seq = item_seq[-max_len+1:]
                    pad_item_seq =  [self.opt["itemnum"]] + item_seq + [self.opt["source_item_num"] + self.opt["target_item_num"]] * (pad_len-1)
                else:
                    pad_item_seq =  [self.opt["source_item_num"] + self.opt["target_item_num"]] * pad_len + item_seq 
                output_item_seqs.append(pad_item_seq[-max_len:])
            return output_item_seqs
        for i in range(len(item_seqs)):
            item_seq = np.array(item_seqs[i])
            item_seq = item_seq[item_seq != (self.opt["source_item_num"] + self.opt["target_item_num"])].tolist()
            augmented_seq = []
            for _ in range(2):
                aug_item_seq, len_input = self.transform(item_seq)
                pad_len = max(0,max_len - len_input)
                if self.opt["is_pooling"] and self.opt['pooling']=="bert":
                    if len(aug_item_seq)>=max_len:
                        aug_item_seq = aug_item_seq[-max_len+1:]
                    aug_item_seq =  [self.opt["itemnum"]] + aug_item_seq + [self.opt["source_item_num"] + self.opt["target_item_num"]] * (pad_len-1)
                else:
                    aug_item_seq =  [self.opt["source_item_num"] + self.opt["target_item_num"]] * pad_len + aug_item_seq 
        
                aug_item_seq = aug_item_seq[-max_len:]

                assert len(aug_item_seq) == max_len
                augmented_seq.append(aug_item_seq)
            augmented_seqs.append(augmented_seq)
        return augmented_seqs
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
