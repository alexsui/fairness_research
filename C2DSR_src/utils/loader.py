"""
Data loader for TACRED json files.
"""
import argparse
import json
import random
import torch
import numpy as np
import codecs
import copy
import pdb
import ipdb
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from utils.augmentation import *
from utils.MoCo_utils import compute_embedding_for_female_mixed_user
import pandas as pd 
from model.C2DSR import Generator

class DataLoader(DataLoader):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, filename, batch_size, opt, evaluation, collate_fn = None, generator = None, model = None):
        self.batch_size = batch_size
        self.opt = opt
        self.eval = evaluation
        self.filename  = filename
        self.collate_fn = collate_fn
        self.model = model 
        if generator is not None:
            self.source_generator = generator[0]
            self.target_generator = generator[1]
            self.mixed_generator = generator[2]
        
        self.opt["maxlen"] = 50
        # ************* item_id *****************
        opt["source_item_num"], opt["target_item_num"] = self.read_item("./fairness_dataset/Movie_lens_time/" + filename + "/train.txt")
        opt['itemnum'] = opt['source_item_num'] + opt['target_item_num'] + 1
        
        # print(opt["source_item_num"] )
        # print(opt["target_item_num"] )

        # ************* sequential data *****************
        # if self.opt['domain'] =="cross":
        source_train_data = "./fairness_dataset/Movie_lens_time/" + filename + f"/train.txt"
        source_valid_data = "./fairness_dataset/Movie_lens_time/" + filename + f"/valid.txt"
        source_test_data = "./fairness_dataset/Movie_lens_time/" + filename + f"/test.txt"
      

        if evaluation < 0:
            self.train_data = self.read_train_data(source_train_data)
            if self.opt['data_augmentation']=="user_generation" and self.model is not None:
                self.overlap_user_generation()
            if self.opt['data_augmentation']=="item_augmentation" and generator is not None:
                self.item_generate()
            data = self.preprocess()
            self.num_examples = len(data)
            
            self.all_data = data
            print("train_data length:",len(data))
        elif evaluation == 2:
            self.test_data = self.read_test_data(source_valid_data)
            data = self.preprocess_for_predict()
        else :
            self.test_data = self.read_test_data(source_test_data)
            data = self.preprocess_for_predict()
            print("test_data length:",len(data))
        
        # shuffle for training
        if evaluation == -1:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
            if batch_size > len(data):
                batch_size = len(data)
                self.batch_size = batch_size
            if len(data)%batch_size != 0:
                data += data[:batch_size]
            data = data[: (len(data)//batch_size) * batch_size]
        else :
            batch_size = 256
        

        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
    def overlap_user_generation(self):
        print("*"*50)
        print("\033[33mOverlap user generation\033[0m")
        female_seq = [d for d in self.train_data if d[0]==0]
        overlap_seq_feat, _ = compute_embedding_for_female_mixed_user(self.opt, female_seq, self.model, name = 'overlap')
        sim = overlap_seq_feat@overlap_seq_feat.T #dot product
        # sim = torch.nn.CosineSimilarity(dim=1)(overlap_seq_feat,overlap_seq_feat) #cosine similarity
        top_k_values, top_k_indice = torch.topk(sim, 20, dim=1)
        n_repeat = 2
        augmented_seq= []
        for _ in range(n_repeat):
            lookup_dict = { i: seq[1] for i, seq in enumerate(female_seq)}
            top_k_data = [[lookup_dict[idx] for idx in sorted(random.sample(indices, 10))] for indices in top_k_indice.tolist()] #從20個中隨機選10個
            try:
                for topk in top_k_data:
                    item_seq = [[self.opt["source_item_num"] + self.opt["target_item_num"]]*(self.opt["maxlen"] - len(seq)) + [i_t[0] for i_t in seq] for seq in topk]
                    item_seq = torch.tensor(item_seq)
                    if self.opt['cuda']:
                        item_seq = item_seq.cuda()
                        
                    # pick similar sequential item
                    embedding = self.model.item_emb(item_seq)# [10, 50, 128]
                    vectors_i = embedding[:, :-1, :].permute(1,0,2)    # [49, 10, 128]
                    vectors_i_plus_1 = embedding[:, 1:, :].permute(1,2,0)  # [49, 128, 10]
                    sim = vectors_i@vectors_i_plus_1 #[49, 10, 10] 
                    all_idx = torch.argmax(sim, dim=2) #[49, 10] 
                    cur = random.randint(0, item_seq.size(0)-1)
                    selected_idx = [cur]
                    for idx in all_idx:
                        selected_idx.append(idx[cur].item())
                        cur = idx[cur].item()
                    item_seq = item_seq[selected_idx, torch.arange(item_seq.size(-1))] 
                    
                    # random pick item per position
                    # indices = torch.randint(item_seq.size(0), (item_seq.size(-1),)) 
                    # item_seq = item_seq[indices, torch.arange(item_seq.size(-1))]  
                    
                    new_item_seq = item_seq[item_seq!=self.opt["source_item_num"] + self.opt["target_item_num"]]
                    if len([s for s in new_item_seq if s >= self.opt["source_item_num"]]) <3:
                        continue
                    ts = [i_t[1] for seq in topk for i_t in seq]
                    sorted_ts = sorted(ts)
                    new_ts = random.sample(sorted_ts, len(new_item_seq))
                    new_ts = sorted(new_ts)
                    new_data = [0,[[i,t] for i, t in zip(new_item_seq.tolist(), new_ts)]]
                    augmented_seq.append(new_data)
            except:
                ipdb.set_trace()
        self.train_data = self.train_data + augmented_seq
        print("Number of female augmented sequence:",len(augmented_seq))
        print("Number of female sequence:",len([s for s in self.train_data if s[0]==0]))
        print("Number of male sequence:",len([s for s in self.train_data if s[0]==1]))
        print("*"*50)
    def generate(self,seq, positions, timestamp, type):
        if type == "X":
            generator = self.source_generator
        elif type == "Y":
            generator = self.target_generator
        elif type == "mixed":
            generator = self.mixed_generator
        try:
            torch_seq = torch.LongTensor(seq) #[X,max_len]
            mask = torch_seq == self.opt['itemnum']
            torch_position = torch.LongTensor(positions) #[X,max_len]
            torch_ts = torch.LongTensor(timestamp) #[X,max_len]
            generator.eval()
            if self.opt['cuda']:
                generator = generator.cuda()
                torch_seq = torch_seq.cuda()
                torch_position = torch_position.cuda()
                torch_ts = torch_ts.cuda()
            with torch.no_grad():
                if self.opt['time_encode']:
                    seq_fea = generator(torch_seq,torch_position,torch_ts) #[X,max_len,item_num]
                else:
                    seq_fea = generator(torch_seq,torch_position)
                target_fea = seq_fea[mask]
                target_fea /= torch.max(target_fea, dim=1, keepdim=True)[0]
                probabilities = torch.nn.functional.softmax(target_fea, dim=1)
                sampled_indices = torch.multinomial(probabilities, 1, replacement=True).squeeze()
                if type =="Y":
                    sampled_indices = sampled_indices + self.opt['source_item_num']
                torch_seq[mask] = sampled_indices
                new_seq = torch_seq.tolist() 
        except:
            ipdb.set_trace()
        new_seq = [(0,[[x,ts] for x,ts in zip(sublist,timestamp) if x != self.opt['source_item_num'] + self.opt['target_item_num']]) for sublist, timestamp in zip(new_seq,torch_ts.tolist())]
        return new_seq
    def item_generate(self):
        with open(f"./fairness_dataset/Movie_lens_time/{self.filename}/average_sequence_length.json","r")  as f:
            avg_length = json.load(f)
        source_flag = 1 if avg_length['source_male'] < avg_length['source_female'] else 0
        target_flag = 1 if avg_length['target_male'] < avg_length['target_female'] else 0
        source_item_inserted_num = abs(avg_length['source_male'] -avg_length['source_female']) if abs(avg_length['source_male'] -avg_length['source_female'])>0 else 1
        target_item_inserted_num = abs(avg_length['target_male'] -avg_length['target_female']) if abs(avg_length['target_male'] -avg_length['target_female'])>0 else 1
        if source_flag and  target_flag:
            male_insert_type = "both"
            female_insert_type = None
        elif source_flag:
            male_insert_type = "X"
            female_insert_type = "Y"
        elif target_flag:
            male_insert_type = "Y"
            female_insert_type ="X"
        else:
            male_insert_type = None
            female_insert_type = "both"
        print("\033[33mitem generation\033[0m")
        female_seq = [d for d in self.train_data if d[0]==0]
        male_seq = [d for d in self.train_data if d[0]==1]
        source_l, source_positions, source_timestamp = [], [], []
        target_l, target_positions, target_timestamp = [], [], []
        both_l, both_positions, both_timestamp = [], [], []

        for gender, seq in male_seq:
            #確保mixed sequence不全是source item
            if all([x[0] < self.opt['source_item_num'] for x in seq]):
                continue
            insert_type = male_insert_type if gender == 1 else female_insert_type
            if insert_type == "X" :
                source_item_seq_len = len([i for i,t in seq if i<self.opt['source_item_num']])
                max_item_num = int(source_item_seq_len/2) if source_item_seq_len>3 else 2
                
                sampled_item_inserted_num = torch.clamp(torch.poisson(torch.tensor([source_item_inserted_num], dtype = torch.float32)),min=1, max = max_item_num).int()
            elif insert_type == "Y":
                target_item_seq_len = len([i for i,t in seq if i>=self.opt['source_item_num']])
                max_item_num = int(target_item_seq_len/2) if target_item_seq_len>3 else 2
                sampled_item_inserted_num = torch.clamp(torch.poisson(torch.tensor([target_item_inserted_num], dtype = torch.float32)),min=1, max = max_item_num).int()
            else:
                max_item_num = int(len(seq)/2) if len(seq)>3 else 2
                if source_item_inserted_num + target_item_inserted_num >max_item_num:
                    sampled_item_inserted_num = max_item_num
                else:
                    sampled_item_inserted_num = source_item_inserted_num + target_item_inserted_num
            insert_idxs = np.argsort(np.abs(np.diff(np.array([s[1] for s in seq]))))[-sampled_item_inserted_num :] + 1
            # insert_idxs = random.choices(list(range(0, len(seq))), k = sampled_item_inserted_num)
            new_seq = copy.deepcopy(seq)
            for index in insert_idxs:
                if index == len(new_seq)-1:
                    new_seq.append([self.opt['itemnum'],new_seq[index][1]])
                    continue
                t1 = new_seq[index-1][1]
                t2 = new_seq[index][1]
                new_ts = int((t1+t2)/2)
                new_seq.insert(index, [self.opt['itemnum'],new_ts])
            position = list(range(len(new_seq)+1))[1:]
            ts = [0] * (self.opt["maxlen"] - len(new_seq)) + [t for i,t in new_seq]
            new_seq = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (self.opt["maxlen"] - len(new_seq)) + [i for i,t in new_seq]
            position = [0] * (self.opt["maxlen"] - len(position)) + position
            if (gender==0 and female_insert_type =="X") or (gender==1 and male_insert_type =="X"): 
                source_l.append(new_seq)
                source_positions.append(position)
                source_timestamp.append(ts)
            elif (gender==0 and female_insert_type =="Y") or (gender==1 and male_insert_type =="Y"):
                target_l.append(new_seq)
                target_positions.append(position)
                target_timestamp.append(ts)
            else:
                both_l.append(new_seq)
                both_positions.append(position)
                both_timestamp.append(ts)
        
        # if female_insert_type is not None:
        #     for gender,seq in female_seq:
        #         #確保mixed sequence不全是source item
        #         if all([x[0] < self.opt['source_item_num'] for x in seq]):
        #             continue
        #         idxs = random.choices(list(range(0, len(seq))), k = self.opt['generate_num'])
        #         new_seq = copy.deepcopy(seq)
        #         for index in idxs:
        #             if index == len(new_seq)-1:
        #                 new_seq.append([self.opt['itemnum'],new_seq[index][1]])
        #                 continue
        #             t1 = new_seq[index-1][1]
        #             t2 = new_seq[index][1]
        #             new_ts = int((t1+t2)/2)
        #             new_seq.insert(index, [self.opt['itemnum'],new_ts])
        #         position = list(range(len(new_seq)+1))[1:]
        #         ts = [0] * (self.opt["maxlen"] - len(new_seq)) + [t for i,t in new_seq]
        #         new_seq = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (self.opt["maxlen"] - len(new_seq)) + [i for i,t in new_seq]
        #         position = [0] * (self.opt["maxlen"] - len(position)) + position
        #         l.append(new_seq)
        #         positions.append(position)
        #         timestamp.append(ts)
        new_seq1, new_seq2, new_seq3 = [], [], []
        if source_l:
            new_seq1 = self.generate(source_l, source_positions, source_timestamp, type = "X")
        if target_l:
            new_seq2 = self.generate(target_l, target_positions, target_timestamp, type = "Y")
        if both_l:
            new_seq3 = self.generate(both_l, both_positions, both_timestamp, type ="mixed")
        self.train_data = new_seq1 + new_seq2 + new_seq3 + female_seq
        # print("Number of source item sequnece detected:",count)
    def read_item(self, fname):
        with codecs.open(fname, "r", encoding="utf-8") as fr:
            item_num = [int(d.strip()) for d in fr.readlines()[:2]]
        return item_num

    def read_train_data(self, train_file):
        def takeSecond(elem):
            return elem[1]
        with codecs.open(train_file, "r", encoding="utf-8") as infile:
            train_data = []
            for id, line in enumerate(infile.readlines()):
                if id<2:
                    continue
                res = []
                line = line.strip()
                gender = list(map(int, line.split()[1]))[0]
                i_t = [list(map(int,tmp.split("|"))) for tmp in line.split()[2:]]
                # data = (line[0],line[1:])
                data = (gender,i_t)
                train_data.append(data)
        return train_data

    def read_test_data(self, test_file):
        def takeSecond(elem):
            return elem[1]
        with codecs.open(test_file, "r", encoding="utf-8") as infile:
            test_data = []
            for id, line in enumerate(infile.readlines()):
                if id<2:
                    continue
                res = []
                line = line.strip()
                gender = list(map(int, line.split()[1]))[0]
                i_t = [list(map(int,tmp.split("|"))) for tmp in line.split()[2:]]
                res = (gender, i_t)
                res_2 = []
                for r in res[1][:-1]:
                    res_2.append(r)

                if res[1][-1][0] >= self.opt["source_item_num"]: # denoted the corresponding validation/test entry , 若為y domain的item_id
                    test_data.append([res_2, 1, res[1][-1][0], res[0]]) #[整個test sequence, 1, 最後一個item_id, gender]
                else :
                    test_data.append([res_2, 0, res[1][-1][0], res[0]])
        return test_data
    
    def subtract_min(self,ts):
        min_value = min(filter(lambda x: x != 0, ts))
        result_list = [x - min_value if x != 0 else 0 for x in ts]
        return result_list
    def encode_time_features(self, timestamps):
      
        datetimes = pd.to_datetime(timestamps, unit='s')

        times_of_day = datetimes.hour + datetimes.minute / 60
        days_of_week = datetimes.weekday
        days_of_year = datetimes.dayofyear
        times_of_day_sin = np.sin(2 * np.pi * times_of_day / 24)
        times_of_day_cos = np.cos(2 * np.pi * times_of_day / 24)
        days_of_week_sin = np.sin(2 * np.pi * days_of_week / 7)
        days_of_week_cos = np.cos(2 * np.pi * days_of_week / 7)
        days_of_year_sin = np.sin(2 * np.pi * days_of_year / 365)
        days_of_year_cos = np.cos(2 * np.pi * days_of_year / 365)

        time_features = np.vstack((times_of_day_sin, times_of_day_cos,
                                days_of_week_sin, days_of_week_cos,
                                days_of_year_sin, days_of_year_cos)).T

        scaler = StandardScaler()
        time_features = scaler.fit_transform(time_features)
        return time_features
    # def preprocess_for_predict(self):

    #     if "Enter" in self.filename:
    #         max_len = 30
    #         self.opt["maxlen"] = 30
    #     else:
    #         max_len = 50
    #         self.opt["maxlen"] = 50

    #     processed=[]
    #     for index, data in enumerate(self.test_data): # the pad is needed! but to be careful. [res[0], res_2, 1, res[1][-1]]
    #         ipdb.set_trace()
    #         gender = data[-1]
    #         data[0] = [[tmp,0] for tmp in data[0]]
    #         position = list(range(len(data[0])+1))[1:]
    #         xd = []
    #         xcnt = 1
    #         x_position = []
    #         ts_xd = []

    #         yd = []
    #         ycnt = 1
    #         y_position = []
    #         ts_yd = []

            
    #         masked_xd = []
    #         neg_xd = []
    #         masked_yd = []
    #         neg_yd = []
    #         for item,ts in data[0]:
                
    #             if item < self.opt["source_item_num"]:
    #                 xd.append(item)
    #                 x_position.append(xcnt)
    #                 xcnt += 1
    #                 yd.append(self.opt["source_item_num"] + self.opt["target_item_num"])
    #                 y_position.append(0)
    #                 ts_xd.append(ts)
    #                 ts_yd.append(0)
    #                 if self.opt['ssl'] =="mask_prediction":
    #                     if random.random() < self.opt['mask_prob']:
    #                         masked_xd.append(self.opt["source_item_num"] + self.opt["target_item_num"])
    #                         neg_xd.append(random.sample(set(range(self.opt['source_item_num'])) - set(xd),1)[0])
    #                     else:
    #                         masked_xd.append(item)
    #                         neg_xd.append(item)
    #                     masked_yd.append(self.opt["source_item_num"] + self.opt["target_item_num"])
    #                     neg_yd.append(self.opt["source_item_num"] + self.opt["target_item_num"])
                    
    #             else:
    #                 xd.append(self.opt["source_item_num"] + self.opt["target_item_num"])
    #                 x_position.append(0)
    #                 yd.append(item)
    #                 y_position.append(ycnt)
    #                 ycnt += 1
    #                 ts_xd.append(0)
    #                 ts_yd.append(ts)
    #                 if self.opt['ssl'] =="mask_prediction":
    #                     masked_xd.append(self.opt["source_item_num"] + self.opt["target_item_num"])
    #                     neg_xd.append(self.opt["source_item_num"] + self.opt["target_item_num"])
    #                     if random.random()<self.opt['mask_prob']:
    #                         masked_yd.append(self.opt["source_item_num"] + self.opt["target_item_num"])
    #                         neg_yd.append(random.sample(set(range(self.opt['source_item_num'],self.opt['source_item_num']+self.opt['target_item_num'])) - set(yd),1)[0])
    #                     else:
    #                         masked_yd.append(item)
    #                         neg_yd.append(item)
    #         if self.opt['ssl'] =="mask_prediction":
    #             try:
    #                 reversed_list = masked_xd[::-1]
    #                 idx = len(masked_xd) - reversed_list.index(next(filter(lambda x: x != self.opt["source_item_num"] + self.opt["target_item_num"], reversed_list))) - 1
    #                 masked_xd[idx] = self.opt["source_item_num"] + self.opt["target_item_num"]
    #                 neg_xd[idx] =random.sample(set(range(self.opt['source_item_num'])) - set(xd),1)[0]
    #                 reversed_list = masked_yd[::-1]
    #                 idx = len(masked_yd) - reversed_list.index(next(filter(lambda x: x != self.opt["source_item_num"] + self.opt["target_item_num"], reversed_list))) - 1
    #                 masked_yd[idx] =self.opt["source_item_num"] + self.opt["target_item_num"]
    #                 neg_yd[idx] =random.sample(set(range(self.opt['source_item_num'],self.opt['source_item_num']+self.opt['target_item_num'])) - set(yd),1)[0]
    #             except StopIteration:
    #                 continue
                
    #         ts_d = [t for i,t in data[0]]
    #         seq = [item for item,ts in data[0]]
    #         # if self.opt['time_encode']:
    #         #     ts_d = self.time_transformation(ts_d)
    #         #     ts_xd = self.time_transformation(ts_xd)
    #         #     ts_yd = self.time_transformation(ts_yd)
    #         if len(data[0]) < max_len:
    #             position = [0] * (max_len - len(data[0])) + position
    #             x_position = [0] * (max_len - len(data[0])) + x_position
    #             y_position = [0] * (max_len - len(data[0])) + y_position

    #             xd = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (max_len - len(data[0])) + xd
    #             yd = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (max_len - len(data[0])) + yd
    #             seq = [self.opt["source_item_num"] + self.opt["target_item_num"]]*(max_len - len(data[0])) + seq
                
    #             ts_xd = [0] * (max_len - len(ts_xd)) + ts_xd
    #             ts_yd = [0] * (max_len - len(ts_yd)) + ts_yd
    #             ts_d = [0]*(max_len - len(ts_d)) + ts_d
    #             masked_xd = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (max_len - len(masked_xd)) + masked_xd
    #             neg_xd = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (max_len - len(neg_xd)) + neg_xd
    #             masked_yd = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (max_len - len(masked_yd)) + masked_yd
    #             neg_yd = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (max_len - len(neg_yd)) + neg_yd
    #             gender = [gender]*max_len
    #         x_last = -1
    #         for id in range(len(x_position)):
    #             id += 1
    #             if x_position[-id]:
    #                 x_last = -id
    #                 break

    #         y_last = -1
    #         for id in range(len(y_position)):
    #             id += 1
    #             if y_position[-id]:
    #                 y_last = -id
    #                 break

    #         negative_sample = []
    #         for i in range(999):
    #             while True:
    #                 if data[1] : # in Y domain, the validation/test negative samples
    #                     sample = random.randint(0, self.opt["target_item_num"] - 1)
    #                     if sample != data[2] - self.opt["source_item_num"]: #若沒有sample到最後一個item_id
    #                         negative_sample.append(sample)
    #                         break
    #                 else : # in X domain, the validation/test negative samples
    #                     sample = random.randint(0, self.opt["source_item_num"] - 1)
    #                     if sample != data[2]:
    #                         negative_sample.append(sample)
    #         if data[1]:
    #             processed.append([seq, xd, yd, position, x_position, y_position, ts_d, ts_xd, ts_yd, x_last, y_last, data[1],
    #                               data[2]-self.opt["source_item_num"], negative_sample, masked_xd, neg_xd, masked_yd, neg_yd, index,gender])
    #         else:
    #             processed.append([seq, xd, yd, position, x_position, y_position, ts_d, ts_xd, ts_yd, x_last, y_last, data[1],
    #                               data[2], negative_sample, masked_xd, neg_xd, masked_yd, neg_yd, index, gender])
    #         # ipdb.set_trace()
    #         print(index)
    #     return processed
    def preprocess_for_predict(self):

        if "Enter" in self.filename:
            max_len = 30
            self.opt["maxlen"] = 30
        else:
            max_len = 50
            self.opt["maxlen"] = 50

        processed=[]
        for index, d in enumerate(self.test_data): # the pad is needed! but to be careful. [res[0], res_2, 1, res[1][-1]]
            gender = d[-1]
            # d[0] = [[tmp,0] for tmp in d[0]]
            # ipdb.set_trace()
            position = list(range(len(d[0])+1))[1:]
            xd = []
            xcnt = 1
            x_position = []
            ts_xd = []

            yd = []
            ycnt = 1
            y_position = []
            ts_yd = []

            
            masked_xd = []
            neg_xd = []
            masked_yd = []
            neg_yd = []
            for item,ts in d[0]:
                
                if item < self.opt["source_item_num"]:
                    xd.append(item)
                    x_position.append(xcnt)
                    xcnt += 1
                    yd.append(self.opt["source_item_num"] + self.opt["target_item_num"])
                    y_position.append(0)
                    ts_xd.append(ts)
                    ts_yd.append(0)
                    if self.opt['ssl'] =="mask_prediction":
                        if random.random() < self.opt['mask_prob']:
                            masked_xd.append(self.opt["source_item_num"] + self.opt["target_item_num"])
                            neg_xd.append(random.sample(set(range(self.opt['source_item_num'])) - set(xd),1)[0])
                        else:
                            masked_xd.append(item)
                            neg_xd.append(item)
                        masked_yd.append(self.opt["source_item_num"] + self.opt["target_item_num"])
                        neg_yd.append(self.opt["source_item_num"] + self.opt["target_item_num"])
                    
                else:
                    xd.append(self.opt["source_item_num"] + self.opt["target_item_num"])
                    x_position.append(0)
                    yd.append(item)
                    y_position.append(ycnt)
                    ycnt += 1
                    ts_xd.append(0)
                    ts_yd.append(ts)
                    if self.opt['ssl'] =="mask_prediction":
                        masked_xd.append(self.opt["source_item_num"] + self.opt["target_item_num"])
                        neg_xd.append(self.opt["source_item_num"] + self.opt["target_item_num"])
                        if random.random()<self.opt['mask_prob']:
                            masked_yd.append(self.opt["source_item_num"] + self.opt["target_item_num"])
                            neg_yd.append(random.sample(set(range(self.opt['source_item_num'],self.opt['source_item_num']+self.opt['target_item_num'])) - set(yd),1)[0])
                        else:
                            masked_yd.append(item)
                            neg_yd.append(item)
            if self.opt['ssl'] =="mask_prediction":
                try:
                    reversed_list = masked_xd[::-1]
                    idx = len(masked_xd) - reversed_list.index(next(filter(lambda x: x != self.opt["source_item_num"] + self.opt["target_item_num"], reversed_list))) - 1
                    masked_xd[idx] = self.opt["source_item_num"] + self.opt["target_item_num"]
                    neg_xd[idx] =random.sample(set(range(self.opt['source_item_num'])) - set(xd),1)[0]
                    reversed_list = masked_yd[::-1]
                    idx = len(masked_yd) - reversed_list.index(next(filter(lambda x: x != self.opt["source_item_num"] + self.opt["target_item_num"], reversed_list))) - 1
                    masked_yd[idx] =self.opt["source_item_num"] + self.opt["target_item_num"]
                    neg_yd[idx] =random.sample(set(range(self.opt['source_item_num'],self.opt['source_item_num']+self.opt['target_item_num'])) - set(yd),1)[0]
                except StopIteration:
                    continue
                
            ts_d = [t for i,t in d[0]]
            seq = [item for item,ts in d[0]]

            # ts_d = self.subtract_min(ts_d)
            # ts_xd = self.subtract_min(ts_xd)
            # ts_yd = self.subtract_min(ts_yd)
            if len(d[0]) < max_len:
                position = [0] * (max_len - len(d[0])) + position
                x_position = [0] * (max_len - len(d[0])) + x_position
                y_position = [0] * (max_len - len(d[0])) + y_position

                xd = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (max_len - len(d[0])) + xd
                yd = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (max_len - len(d[0])) + yd
                seq = [self.opt["source_item_num"] + self.opt["target_item_num"]]*(max_len - len(d[0])) + seq
                
                ts_xd = [0] * (max_len - len(ts_xd)) + ts_xd
                ts_yd = [0] * (max_len - len(ts_yd)) + ts_yd
                ts_d = [0]*(max_len - len(ts_d)) + ts_d
                masked_xd = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (max_len - len(masked_xd)) + masked_xd
                neg_xd = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (max_len - len(neg_xd)) + neg_xd
                masked_yd = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (max_len - len(masked_yd)) + masked_yd
                neg_yd = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (max_len - len(neg_yd)) + neg_yd
                gender = [gender]*max_len
            x_last = -1
            for id in range(len(x_position)):
                id += 1
                if x_position[-id]:
                    x_last = -id
                    break
            x_last_3 = []
            c = 0
            for id in range(len(x_position)):
                id += 1
                if x_position[-id]:
                    c+=1
                    x_last_3.insert(0,-id)
                    if c==3:
                        break
            y_last = -1
            for id in range(len(y_position)):
                id += 1
                if y_position[-id]:
                    y_last = -id
                    break
            y_last_3 = []
            c = 0
            for id in range(len(y_position)):
                id += 1
                if y_position[-id]:
                    c+=1
                    y_last_3.insert(0,-id)
                    if c==3:
                        break
            count = 0
            if len(x_last_3)<3 or len(y_last_3)<3 :
                count+=1
                continue
            negative_sample = []
            for i in range(999):
                while True:
                    if d[1] : # in Y domain, the validation/test negative samples
                        sample = random.randint(0, self.opt["target_item_num"] - 1)
                        if sample != d[2] - self.opt["source_item_num"]: #若沒有sample到最後一個item_id
                            negative_sample.append(sample)
                            break
                    else : # in X domain, the validation/test negative samples
                        sample = random.randint(0, self.opt["source_item_num"] - 1)
                        if sample != d[2]:
                            negative_sample.append(sample)
                            break
            if d[1]:
                processed.append([seq, xd, yd, position, x_position, y_position, ts_d, ts_xd, ts_yd, x_last, y_last, d[1],
                                  d[2]-self.opt["source_item_num"], negative_sample, masked_xd, neg_xd, masked_yd, neg_yd, index,gender,x_last_3,y_last_3])
            else:
                processed.append([seq, xd, yd, position, x_position, y_position, ts_d, ts_xd, ts_yd, x_last, y_last, d[1],
                                  d[2], negative_sample, masked_xd, neg_xd, masked_yd, neg_yd, index, gender, x_last_3,y_last_3])
        print("Number of test/valid sequence detected:",count)
        return processed

    def preprocess(self):

        def myprint(a):
            for i in a:
                print("%6d" % i, end="")
            print("")
        """ Preprocess the data and convert to ids. """
        processed = []
        female_delete_num = 0
        male_delete_num = 0

        if "Enter" in self.filename:
            max_len = 30
            self.opt["maxlen"] = 30
        else:
            max_len = 50
            self.opt["maxlen"] = 50

        for index, d in enumerate(self.train_data): # the pad is needed! but to be careful.
            gender = d[0]
            d = d[1]
            # d = [[tmp,0] for tmp in d]
            i_t = copy.deepcopy(d)
            d = [i[0] for i in d]
            ground = copy.deepcopy(d)[1:]

            share_x_ground = []
            share_x_ground_mask = []
            share_y_ground = []
            share_y_ground_mask = []
            for w in ground:
                if w < self.opt["source_item_num"]:
                    share_x_ground.append(w)
                    share_x_ground_mask.append(1)
                    share_y_ground.append(self.opt["target_item_num"])
                    share_y_ground_mask.append(0)
                else:
                    share_x_ground.append(self.opt["source_item_num"])
                    share_x_ground_mask.append(0)
                    share_y_ground.append(w - self.opt["source_item_num"])
                    share_y_ground_mask.append(1)
            
            d = d[:-1]  # delete the ground truth
            
            position = list(range(len(d)+1))[1:]
            ground_mask = [1] * len(d)



            xd = [] # the input of X domain
            xcnt = 1 
            x_position = [] # the position of X domain
            ts_xd = [] # the timestamp of X domain

            yd = [] 
            ycnt = 1
            y_position = []
            ts_yd = [] # the timestamp of Y domain
            
            # a list of (item,ts) pairs
            i_t = i_t[:-1]
            i_t_yd = []
            i_t_xd = []
            
            #augmentation
            augment_xd = []
            augment_yd = []
            #corru_x和corru_y是為了後續的contrastive learning而構建的關聯序列。corru_x包含原始的X域項目加上隨機的Y域項目;corru_y包含原始的Y域項目加上隨機的X域項目。
            corru_x = [] # the corrupted input of X domain
            corru_y = [] # the corrupted input of Y domain
            
            
            masked_xd = []
            neg_xd = []
            masked_yd = []
            neg_yd = []
            masked_d = []
            neg_d = []
        
            for i, k in enumerate(i_t):
                item = k[0]
                ts = k[1]
                if item < self.opt["source_item_num"]:
                    corru_x.append(item)
                    xd.append(item)
                    augment_xd.append(item)
                    x_position.append(xcnt)
                    xcnt += 1
                    corru_y.append(random.randint(0, self.opt["source_item_num"] - 1))
                    yd.append(self.opt["source_item_num"] + self.opt["target_item_num"])
                    y_position.append(0)
                    i_t_xd.append(k)
                    ts_xd.append(ts)
                    ts_yd.append(0)
                    if self.opt['ssl'] =="mask_prediction":
                        if random.random()<self.opt['mask_prob']:
                            masked_xd.append(self.opt["source_item_num"] + self.opt["target_item_num"])
                            neg_xd.append(random.sample(set(range(self.opt['source_item_num'])) - set(xd),1)[0])
                        else:
                            masked_xd.append(item)
                            neg_xd.append(item)
                        masked_yd.append(self.opt["source_item_num"] + self.opt["target_item_num"])
                        neg_yd.append(self.opt["source_item_num"] + self.opt["target_item_num"])
                    
                else: #if item is y-domain
                    corru_x.append(random.randint(self.opt["source_item_num"], self.opt["source_item_num"] + self.opt["target_item_num"] - 1))
                    xd.append(self.opt["source_item_num"] + self.opt["target_item_num"])
                    x_position.append(0)
                    corru_y.append(item)
                    yd.append(item)
                    augment_yd.append(item)
                    y_position.append(ycnt)
                    ycnt += 1
                    i_t_yd.append(k)
                    ts_yd.append(ts)
                    ts_xd.append(0)
                    if self.opt['ssl'] =="mask_prediction":
                        masked_xd.append(self.opt["source_item_num"] + self.opt["target_item_num"])
                        neg_xd.append(self.opt["source_item_num"] + self.opt["target_item_num"])
                        if random.random()<self.opt['mask_prob']:
                            masked_yd.append(self.opt["source_item_num"] + self.opt["target_item_num"])
                            neg_yd.append(random.sample(set(range(self.opt['source_item_num'],self.opt['source_item_num']+self.opt['target_item_num'])) - set(yd),1)[0])
                        else:
                            masked_yd.append(item)
                            neg_yd.append(item)
            if self.opt['ssl'] =="mask_prediction":
                try:
                    reversed_list = masked_xd[::-1]
                    idx = len(masked_xd) - reversed_list.index(next(filter(lambda x: x != self.opt["source_item_num"] + self.opt["target_item_num"], reversed_list))) - 1
                    masked_xd[idx] = self.opt["source_item_num"] + self.opt["target_item_num"]
                    neg_xd[idx] =random.sample(set(range(self.opt['source_item_num'])) - set(xd),1)[0]
                    reversed_list = masked_yd[::-1]
                    idx = len(masked_yd) - reversed_list.index(next(filter(lambda x: x != self.opt["source_item_num"] + self.opt["target_item_num"], reversed_list))) - 1
                    masked_yd[idx] = self.opt["source_item_num"] + self.opt["target_item_num"]
                    neg_yd[idx] = random.sample(set(range(self.opt['source_item_num'],self.opt['source_item_num']+self.opt['target_item_num'])) - set(yd),1)[0]
                except StopIteration:
                    continue
            # for i in d[:-1]:
            #     if (i!= self.opt["source_item_num"] + self.opt["target_item_num"]) and (random.random() < self.opt['mask_prob']):
            #         masked_d.append(self.opt["source_item_num"] + self.opt["target_item_num"])
            #         neg = random.sample(set(range(self.opt['source_item_num']+self.opt['target_item_num'])) - set(d),1)[0]
            #         neg_d.append(neg)
            #     else:
            #         masked_d.append(i)
            #         neg_d.append(i)
            # masked_d.append(self.opt["source_item_num"] + self.opt["target_item_num"])
            # neg_d.append(random.sample(set(range(self.opt['source_item_num']+self.opt['target_item_num'])) - set(d),1)[0])
            
            ts_d = [t for i,t in i_t]
            #產生單域序列的ground truth
            now = -1
            x_ground = [self.opt["source_item_num"]] * len(xd) # caution!
            x_ground_mask = [0] * len(xd)
            for id in range(len(xd)):
                id+=1
                if x_position[-id]:
                    if now == -1: #若為第一個ground truth
                        now = xd[-id]
                        if ground[-1] < self.opt["source_item_num"]: #若為x domain
                            x_ground[-id] = ground[-1]
                            x_ground_mask[-id] = 1
                        else:
                            xd[-id] = self.opt["source_item_num"] + self.opt["target_item_num"]
                            ts_xd[-id] = 0
                            augment_xd = augment_xd[:-1]
                            x_position[-id] = 0
                    else:
                        x_ground[-id] = now
                        x_ground_mask[-id] = 1
                        now = xd[-id]
            if sum(x_ground_mask) == 0:
                # print("pass sequence x")
                if gender ==1:
                    male_delete_num+=1
                else:
                    female_delete_num+=1
                continue

            now = -1
            y_ground = [self.opt["target_item_num"]] * len(yd) # caution!
            y_ground_mask = [0] * len(yd)
            for id in range(len(yd)):
                id+=1
                if y_position[-id]:
                    if now == -1:
                        now = yd[-id] - self.opt["source_item_num"]
                        if ground[-1] > self.opt["source_item_num"]:
                            y_ground[-id] = ground[-1] - self.opt["source_item_num"]
                            y_ground_mask[-id] = 1
                        else:
                            yd[-id] = self.opt["source_item_num"] + self.opt["target_item_num"]
                            y_position[-id] = 0
                            ts_yd[-id] = 0
                            augment_yd = augment_yd[:-1]
                    else:
                        y_ground[-id] = now
                        y_ground_mask[-id] = 1
                        now = yd[-id] - self.opt["source_item_num"]
                        
            if sum(y_ground_mask) == 0:
                # print("pass sequence y")
                if gender ==1:
                    male_delete_num+=1
                else:
                    female_delete_num+=1
                continue
            
            if len(d) < max_len:
                position = [0] * (max_len - len(d)) + position
                x_position = [0] * (max_len - len(d)) + x_position
                y_position = [0] * (max_len - len(d)) + y_position

                ground = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (max_len - len(d)) + ground
                share_x_ground = [self.opt["source_item_num"]] * (max_len - len(d)) + share_x_ground
                share_y_ground = [self.opt["target_item_num"]] * (max_len - len(d)) + share_y_ground
                x_ground = [self.opt["source_item_num"]] * (max_len - len(d)) + x_ground
                y_ground = [self.opt["target_item_num"]] * (max_len - len(d)) + y_ground

                ground_mask = [0] * (max_len - len(d)) + ground_mask
                share_x_ground_mask = [0] * (max_len - len(d)) + share_x_ground_mask
                share_y_ground_mask = [0] * (max_len - len(d)) + share_y_ground_mask
                x_ground_mask = [0] * (max_len - len(d)) + x_ground_mask
                y_ground_mask = [0] * (max_len - len(d)) + y_ground_mask

                corru_x = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (max_len - len(d)) + corru_x
                corru_y = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (max_len - len(d)) + corru_y  
                xd = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (max_len - len(d)) + xd             #x domain sequence
                yd = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (max_len - len(d)) + yd             #y domain sequence
                d = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (max_len - len(d)) + d               #mixed domain sequence
                # timestamp padding
                ts_xd = [0] * (max_len - len(ts_xd)) + ts_xd
                ts_yd = [0] * (max_len - len(ts_yd)) + ts_yd
                ts_d = [0]*(max_len - len(ts_d)) + ts_d
                gender = [gender]*max_len
                augment_xd = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (max_len - len(augment_xd)) + augment_xd
                augment_yd = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (max_len - len(augment_yd)) + augment_yd
                if self.opt['ssl']=='mask_prediction':
                    masked_xd = [self.opt["source_item_num"] + self.opt["target_item_num"]]*(max_len - len(masked_xd)) + masked_xd
                    neg_xd = [self.opt["source_item_num"] + self.opt["target_item_num"]]*(max_len - len(neg_xd)) + neg_xd
                    masked_yd = [self.opt["source_item_num"] + self.opt["target_item_num"]]*(max_len - len(masked_yd)) + masked_yd
                    neg_yd = [self.opt["source_item_num"] + self.opt["target_item_num"]]*(max_len - len(neg_yd)) + neg_yd
                # i_t_xd = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (max_len - len(i_t_xd)) + i_t_xd
                # i_t_yd = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (max_len - len(i_t_yd)) + i_t_yd
            else:
                print("pass")
            # ts_xd = self.subtract_min(ts_xd)
            # ts_yd = self.subtract_min(ts_yd)
            # ts_d = self.subtract_min(ts_d)
            #create masked item sequence
            # if self.opt['ssl'] =="mask_prediction":
            #     masked_xd = []
            #     neg_xd = []
            #     masked_yd = []
            #     neg_yd = []
            #     masked_d = []
            #     neg_d = []
            #     for i in xd[:-1]:
            #         if (i!= self.opt["source_item_num"] + self.opt["target_item_num"]) and (random.random() < self.opt['mask_prob']):
            #             masked_xd.append(self.opt["source_item_num"] + self.opt["target_item_num"])
            #             neg = random.sample(set(range(self.opt['source_item_num'])) - set(xd),1)[0]
            #             neg_xd.append(neg)
            #         else:
            #             masked_xd.append(i)
            #             neg_xd.append(i)
            #     masked_xd.append(self.opt["source_item_num"] + self.opt["target_item_num"])
            #     neg_xd.append(random.sample(set(range(self.opt['source_item_num'])) - set(xd),1)[0])
            #     for i in yd[:-1]:
            #         if (i!= self.opt["source_item_num"] + self.opt["target_item_num"]) and (random.random() < self.opt['mask_prob']):
            #             masked_yd.append(self.opt["source_item_num"] + self.opt["target_item_num"])
            #             neg = random.sample(set(range(self.opt['source_item_num'],self.opt['source_item_num']+self.opt['target_item_num'])) - set(yd),1)[0]
            #             neg_yd.append(neg)
            #         else:
            #             masked_yd.append(i)
            #             neg_yd.append(i)
            #     masked_yd.append(self.opt["source_item_num"] + self.opt["target_item_num"])
            #     neg_yd.append(random.sample(set(range(self.opt['source_item_num'],self.opt['source_item_num']+self.opt['target_item_num'])) - set(yd),1)[0])
            #     for i in d[:-1]:
            #         if (i!= self.opt["source_item_num"] + self.opt["target_item_num"]) and (random.random() < self.opt['mask_prob']):
            #             masked_d.append(self.opt["source_item_num"] + self.opt["target_item_num"])
            #             neg = random.sample(set(range(self.opt['source_item_num']+self.opt['target_item_num'])) - set(d),1)[0]
            #             neg_d.append(neg)
            #         else:
            #             masked_d.append(i)
            #             neg_d.append(i)
            #     masked_d.append(self.opt["source_item_num"] + self.opt["target_item_num"])
            #     neg_d.append(random.sample(set(range(self.opt['source_item_num']+self.opt['target_item_num'])) - set(d),1)[0])
            # ssl data preprocessing
            # find the nearest target
            # t_xd = torch.tensor([x[1] for x in i_t_xd])
            # t_yd = torch.tensor([y[1] for y in i_t_yd])
            # i_xd = torch.tensor([x[0] for x in i_t_xd])
            # i_yd = torch.tensor([y[0] for y in i_t_yd])
            # # Reshape tensor1 to a column vector and tensor2 to a row vector
            # t_dx_reshaped1 = t_xd.view(-1, 1)
            # t_dy_reshaped1 = t_yd.view(1, -1)
            # t_dy_reshaped2 = t_yd.view(-1, 1)
            # t_dx_reshaped2 = t_xd.view(1, -1)
            # # Calculate differences using matrix subtraction
            # time_diff_xd = t_dx_reshaped1 - t_dy_reshaped1
            # time_diff_yd = t_dy_reshaped2 - t_dx_reshaped2
            # m_xd = (torch.abs(time_diff_xd)//86400)<=self.opt["window_size"]
            # m_yd = (torch.abs(time_diff_yd)//86400)<=self.opt["window_size"]

            # near_xd = i_yd.unsqueeze(0).masked_fill(m_xd.logical_not(), self.opt["source_item_num"] + self.opt["target_item_num"])
            # near_yd = i_xd.unsqueeze(0).masked_fill(m_yd.logical_not(), self.opt["source_item_num"] + self.opt["target_item_num"])            
            
            # #padding
            # i_xd = torch.cat((i_xd, torch.tensor([self.opt["source_item_num"] + self.opt["target_item_num"]] * (max_len - len(i_xd)))), 0)
            # i_yd = torch.cat((i_yd, torch.tensor([self.opt["source_item_num"] + self.opt["target_item_num"]] * (max_len - len(i_yd)))), 0)
            # # print(torch.tensor([self.opt["source_item_num"] + self.opt["target_item_num"]]).repeat(near_dx.shape[0],max_len-near_dx.shape[1]).shape)
            # near_xd = torch.cat((near_xd, torch.tensor([self.opt["source_item_num"] + self.opt["target_item_num"]]).repeat(near_xd.shape[0], max_len-near_xd.shape[1])), 1)
            # near_xd = torch.cat((near_xd, torch.tensor([self.opt["source_item_num"] + self.opt["target_item_num"]]).repeat(max_len-near_xd.shape[0], near_xd.shape[1])), 0)
            # near_yd = torch.cat((near_yd, torch.tensor([self.opt["source_item_num"] + self.opt["target_item_num"]]).repeat(near_yd.shape[0], max_len-near_yd.shape[1])), 1)
            # near_yd = torch.cat((near_yd, torch.tensor([self.opt["source_item_num"] + self.opt["target_item_num"]]).repeat(max_len-near_yd.shape[0], near_yd.shape[1])), 0)
            # ipdb.set_trace()
            processed.append([index, d, xd, yd, position, x_position, y_position, ts_d, ts_xd, ts_yd, ground, share_x_ground, share_y_ground, x_ground, y_ground, ground_mask, 
                                  share_x_ground_mask, share_y_ground_mask, x_ground_mask, y_ground_mask, corru_x, corru_y, masked_xd, neg_xd, masked_yd, neg_yd, augment_xd, augment_yd, gender])
        print(f"number of male deleted: {male_delete_num}")
        print(f"number of female deleted: {female_delete_num}")
        return processed
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch) #[B,20,50]
        batch = list(zip(*batch))
        if self.eval!=-1:
            if self.collate_fn:
                batch = self.collate_fn(batch)
                return batch
            else:   
                
                return (torch.LongTensor(batch[0]), torch.LongTensor(batch[1]), torch.LongTensor(batch[2]), torch.LongTensor(batch[3]),torch.LongTensor(batch[4]), torch.LongTensor(batch[5]), torch.tensor(batch[6]), torch.tensor(batch[7]), torch.tensor(batch[8]), torch.tensor(batch[9]),\
                    torch.LongTensor(batch[10]),torch.LongTensor(batch[11]),torch.LongTensor(batch[12]),torch.LongTensor(batch[13]),torch.LongTensor(batch[14]), torch.LongTensor(batch[15]), torch.LongTensor(batch[16]), torch.LongTensor(batch[17]),torch.LongTensor(batch[18]),torch.LongTensor(batch[19]),torch.LongTensor(batch[20]),torch.LongTensor(batch[21]))
        else :
            if self.collate_fn:
                batch = self.collate_fn(batch)
                return batch
            else:
                return (torch.LongTensor(batch[0]), torch.LongTensor(batch[1]), torch.LongTensor(batch[2]), torch.LongTensor(batch[3]),torch.LongTensor(batch[4]), torch.LongTensor(batch[5]), torch.LongTensor(batch[6]), torch.LongTensor(batch[7]), torch.LongTensor(batch[8]), torch.LongTensor(batch[9]), torch.LongTensor(batch[10]), torch.LongTensor(batch[11]),\
                        torch.LongTensor(batch[12]), torch.LongTensor(batch[13]), torch.LongTensor(batch[14]), torch.LongTensor(batch[15]), torch.LongTensor(batch[16]), torch.LongTensor(batch[17]),\
                        torch.LongTensor(batch[18]),torch.LongTensor(batch[19]),torch.LongTensor(batch[20]),torch.LongTensor(batch[21]),torch.LongTensor(batch[22]),torch.LongTensor(batch[23]),torch.LongTensor(batch[24]),torch.LongTensor(batch[25]),torch.LongTensor(batch[26]),torch.LongTensor(batch[27]),torch.LongTensor(batch[28]))
    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

class GDataLoader(DataLoader):
    
    #mask_id = opt["source_item_num"]+ opt["target_item_num"] +1
    
    def __init__(self, filename, batch_size, opt, eval, collate_fn=None):
        self.batch_size = batch_size
        self.opt = opt
        self.filename  = filename
        self.collate_fn = collate_fn
        self.eval = eval
        # ************* item_id *****************
        opt["source_item_num"], opt["target_item_num"] = self.read_item("./fairness_dataset/Movie_lens_time/" + filename + "/train.txt")
        opt['itemnum'] = opt['source_item_num'] + opt['target_item_num'] + 1
       
        if "Enter" in self.filename:
            max_len = 30
            self.opt["maxlen"] = 30
        else:
            max_len = 50
            self.opt["maxlen"] = 50
        self.mask_id = opt["source_item_num"]+ opt["target_item_num"] +1
        # ************* sequential data *****************
        # if self.opt['domain'] =="cross":
        source_train_data = "./fairness_dataset/Movie_lens_time/" + filename + f"/train.txt"
        source_valid_data = "./fairness_dataset/Movie_lens_time/" + filename + f"/valid.txt"
        source_test_data = "./fairness_dataset/Movie_lens_time/" + filename + f"/test.txt"
        if self.eval == -1: 
            self.train_data = self.read_train_data(source_train_data)
        else:
            self.train_data = self.read_train_data(source_valid_data)
        self.part_sequence = []
        self.split_sequence()
        data = self.preprocess()
        print("pretrain_data length:",len(data))
        
        # shuffle for training
      
        indices = list(range(len(data)))
        random.shuffle(indices)
        data = [data[i] for i in indices]
        if batch_size > len(data):
            batch_size = len(data)
            self.batch_size = batch_size
        if len(data)%batch_size != 0:
            data += data[:batch_size]
        data = data[: (len(data)//batch_size) * batch_size]
        self.num_examples = len(data)

        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data

    def read_item(self, fname):
        with codecs.open(fname, "r", encoding="utf-8") as fr:
            item_num = [int(d.strip()) for d in fr.readlines()[:2]]
        return item_num

    def read_train_data(self, train_file):
        with codecs.open(train_file, "r", encoding="utf-8") as infile:
            train_data = []
            for id, line in enumerate(infile.readlines()):
                if id<2:
                    continue
                res = []
                line = line.strip()
                gender = list(map(int, line.split()[1]))[0]
                i_t = [list(map(int,tmp.split("|"))) for tmp in line.split()[2:]]
                # data = (line[0],line[1:])
                data = (gender,i_t)
                train_data.append(data)
        return train_data
    
    def subtract_min(self,ts):
        min_value = min(filter(lambda x: x != 0, ts))
        result_list = [x - min_value if x != 0 else 0 for x in ts]
        return result_list
    def encode_time_features(self, timestamps):
      
        datetimes = pd.to_datetime(timestamps, unit='s')

        times_of_day = datetimes.hour + datetimes.minute / 60
        days_of_week = datetimes.weekday
        days_of_year = datetimes.dayofyear
        times_of_day_sin = np.sin(2 * np.pi * times_of_day / 24)
        times_of_day_cos = np.cos(2 * np.pi * times_of_day / 24)
        days_of_week_sin = np.sin(2 * np.pi * days_of_week / 7)
        days_of_week_cos = np.cos(2 * np.pi * days_of_week / 7)
        days_of_year_sin = np.sin(2 * np.pi * days_of_year / 365)
        days_of_year_cos = np.cos(2 * np.pi * days_of_year / 365)

        time_features = np.vstack((times_of_day_sin, times_of_day_cos,
                                days_of_week_sin, days_of_week_cos,
                                days_of_year_sin, days_of_year_cos)).T

        scaler = StandardScaler()
        time_features = scaler.fit_transform(time_features)
        return time_features
    def split_sequence(self):
        for seq in self.train_data:
            input_ids = seq[1][-(self.opt["maxlen"]+1):-1] # keeping same as train set
            for i in range(4,len(input_ids)): # 5 is the minimum length of sequence
                self.part_sequence.append((seq[0],input_ids[:i+1]))
                
    def preprocess(self):

        processed = []
        max_len =self.opt["maxlen"]
        for index, d in enumerate(self.part_sequence): # the pad is needed! but to be careful.
            gender = d[0]
            d = d[1]
            # d = [[tmp,0] for tmp in d] # add redundant timestamp
            i_t = copy.deepcopy(d)
            d = [i[0] for i in d]
            ground = copy.deepcopy(d)[1:]
            position = list(range(len(d)+1))[1:]
            ground_mask = [1] * len(d)
            
            i_t = i_t[:-1]
  
            masked_d = []
            neg_d = []
            ts_d = [t for i,t in i_t]
            
            if not [i for i in d if i < self.opt["source_item_num"]]:
                # print("No X domain item in sequence")
                continue
            if not [i for i in d if i >= self.opt["source_item_num"]]:

                # print("No Y domain item in sequence")
                continue
            
            # ts_d = self.subtract_min(ts_d)
            #create masked item sequence
            target_sentence = []
            masked_d = []
            neg_d = []
            for i in d[:-1]:
                if (i!= self.opt["source_item_num"] + self.opt["target_item_num"]) and (random.random() < self.opt['mask_prob']):
                    masked_d.append(self.mask_id)
                    neg = random.sample(set(range(self.opt['source_item_num']+self.opt['target_item_num'])) - set(d),1)[0]
                    neg_d.append(neg)
                    target_sentence.append(i)
                else:
                    masked_d.append(i)
                    neg_d.append(i)
                    target_sentence.append(self.opt["source_item_num"] + self.opt["target_item_num"])
            masked_d.append(self.mask_id)
            neg_d.append(random.sample(set(range(self.opt['source_item_num']+self.opt['target_item_num'])) - set(d),1)[0])
            target_sentence.append(d[-1])
            if len(d) < max_len:
                position = [0] * (max_len - len(d)) + position
                ground = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (max_len - len(d)) + ground
                ground_mask = [0] * (max_len - len(d)) + ground_mask
                # d = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (max_len - len(d)) + d
                # masked_d = [self.opt["source_item_num"] + self.opt["target_item_num"]]*(max_len - len(masked_d)) + masked_d
                # neg_d = [self.opt["source_item_num"] + self.opt["target_item_num"]]*(max_len - len(neg_d)) + neg_d
                ts_d = [0]*(max_len - len(ts_d)) + ts_d
                gender = [gender]*max_len
                # target_sentence = [self.opt["source_item_num"] + self.opt["target_item_num"]]*(max_len - len(target_sentence)) + target_sentence
            else:
                print("pass")
            # ipdb.set_trace()
            processed.append([index, d, position , ts_d, ground, ground_mask, gender])
        return processed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))
        if self.collate_fn:
            batch = self.collate_fn(batch)
            return batch
        return (torch.LongTensor(batch[0]), torch.LongTensor(batch[1]), torch.LongTensor(batch[2]), torch.LongTensor(batch[3]),torch.LongTensor(batch[4]), torch.LongTensor(batch[5]),\
                torch.tensor(batch[6]), torch.tensor(batch[7]), torch.tensor(batch[8]),torch.tensor(batch[9]))
    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

class NonoverlapDataLoader(DataLoader):
    def __init__(self, filename, batch_size, opt, collate_fn=None):
        self.batch_size = batch_size
        self.opt = opt
        self.filename  = filename
        self.collate_fn = collate_fn
        # ************* item_id *****************
        data1,data2 = filename.split("_")   
        # A_data_name = f"cate_id_{data1}"
        # B_data_name = f"cate_id_{data2}"
        if 'sci-fi' == data1:
            data1 = 'Sci-Fi' 
        if 'sci-fi' == data2:
            data2 = 'Sci-Fi'
        if 'film-noir' == data1:
            data1 = 'Film-Noir'
        if 'film-noir' == data2:
            data2 = 'Film-Noir'
        # folder_name = f"{data1.lower()}_{data2.lower()}"
        A_data_name = data1.capitalize() if data1!="Sci-Fi" and data1!="Film-Noir" else data1
        B_data_name = data2.capitalize() if data2!="Sci-Fi" and data2!="Film-Noir" else data2
        opt["source_item_num"], opt["target_item_num"] = self.read_item("./fairness_dataset/Movie_lens_time/" + filename + f"/nonoverlap_Y_female_{B_data_name}.txt")
        opt['itemnum'] = opt['source_item_num'] + opt['target_item_num'] + 1
       
        if "Enter" in self.filename:
            max_len = 30
            self.opt["maxlen"] = 30
        else:
            max_len = 50
            self.opt["maxlen"] = 50
        self.mask_id = opt["source_item_num"]+ opt["target_item_num"] +1
        # ************* sequential data *****************
      
        source_train_data = "./fairness_dataset/Movie_lens_time/" + filename + f"/nonoverlap_Y_female_{B_data_name}.txt"
        self.train_data = self.read_train_data(source_train_data)


        data = self.preprocess()
        self.all_data = data
        print("number of Nonoverlap user:",len(data))
        indices = list(range(len(data)))
        random.shuffle(indices)
        data = [data[i] for i in indices]
        if batch_size > len(data):
            batch_size = len(data)
            self.batch_size = batch_size
        if len(data)%batch_size != 0:
            data += data[:batch_size]
        data = data[: (len(data)//batch_size) * batch_size]
        self.num_examples = len(data)

        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data

    def read_item(self, fname):
        with codecs.open(fname, "r", encoding="utf-8") as fr:
            item_num = [int(d.strip()) for d in fr.readlines()[:2]]
        return item_num

    def read_train_data(self, train_file):
        with codecs.open(train_file, "r", encoding="utf-8") as infile:
            train_data = []
            for id, line in enumerate(infile.readlines()):
                if id<2:
                    continue
                res = []
                line = line.strip()
                gender = list(map(int, line.split()[1]))[0]
                i_t = [list(map(int,tmp.split("|"))) for tmp in line.split()[2:]]
                # data = (line[0],line[1:])
                data = (gender,i_t)
                train_data.append(data)
        return train_data
    
    def preprocess(self):

        processed = []
        max_len =self.opt["maxlen"]
        for index, d in enumerate(self.train_data): # the pad is needed! but to be careful.
            gender = d[0]
            d = d[1]
            # d = [[tmp,0] for tmp in d] # add redundant timestamp
            i_t = copy.deepcopy(d)
            d = [i[0] for i in d]
            ground = copy.deepcopy(d)[1:]
            d = d[:-1]  # delete the ground truth
            ts_d = [t for i,t in i_t]
            position = list(range(len(d)+1))[1:]
            ground_mask = [1] * len(d)
            
            if len(d) < max_len:
                position = [0] * (max_len - len(d)) + position
                ground = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (max_len - len(d)) + ground
                ground_mask = [0] * (max_len - len(d)) + ground_mask
                d = [self.opt["source_item_num"] + self.opt["target_item_num"]] * (max_len - len(d)) + d               #mixed domain sequence
                # timestamp padding
                ts_d = [0]*(max_len - len(ts_d)) + ts_d
                gender = [gender]*max_len
                
            else:
                print("pass")
            # ipdb.set_trace()
            processed.append([index, d, position , ground, ground_mask, gender, ts_d])
        return processed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))
        return (torch.LongTensor(batch[0]), torch.LongTensor(batch[1]), torch.LongTensor(batch[2]), torch.LongTensor(batch[3]),torch.LongTensor(batch[4]),\
                torch.LongTensor(batch[5]),torch.LongTensor(batch[6]))
    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

