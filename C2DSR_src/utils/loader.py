"""
Data loader for TACRED json files.
"""

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
import pandas as pd 
class DataLoader(DataLoader):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, filename, batch_size, opt, evaluation, data_type=None, collate_fn=None):
        self.batch_size = batch_size
        self.opt = opt
        self.eval = evaluation
        self.filename  = filename
        self.collate_fn = collate_fn
        
        # ************* item_id *****************
        opt["source_item_num"],opt["target_item_num"] = self.read_item("./fairness_dataset/" + filename + "/train.txt")
        opt['itemnum'] = opt['source_item_num'] + opt['target_item_num'] + 1
        # print(opt["source_item_num"] )
        # print(opt["target_item_num"] )

        # ************* sequential data *****************

        source_train_data = "./fairness_dataset/" + filename + "/train.txt"
        source_valid_data = "./fairness_dataset/" + filename + "/valid.txt"
        if evaluation==1:
            if data_type == "mixed":
                source_test_data = "./fairness_dataset/" + filename + "/mixed_test.txt"
            # elif data_type == "male":
            #     source_test_data = "./fairness_dataset/" + filename + "/male_test.txt"
            # elif data_type == "female":
            #     source_test_data = "./fairness_dataset/" + filename + "/female_test.txt"

        if evaluation < 0:
            self.train_data = self.read_train_data(source_train_data)
            data = self.preprocess()
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
        self.num_examples = len(data)

        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data

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
                line = list(map(int, line.split()))[1:]
                data = (line[0],line[1:])
                train_data.append(data)
                # for w in line:
                #     w = w.split(" ")
                #     res.append((int(w[0]), int(w[1])))
                # # res.sort(key=takeSecond)
                # # res_2 = [] 
                # # for r in res:
                # #     res_2.append((int(r[0]),int(r[1])))   
                # train_data.append(res)
                
                # for w in line:
                #     w = w.split("|")
                #     res.append((int(w[0]), int(w[1])))
                # res.sort(key=takeSecond)
                # res_2 = []
                # for r in res:
                # res_2.append(r[0])
                # train_data.append(res_2)
                
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
                line = list(map(int, line.split()))[1:]
                res = (line[0],line[1:])
                res_2 = []
                for r in res[1][:-1]:
                    res_2.append(r)

                if res[1][-1] >= self.opt["source_item_num"]: # denoted the corresponding validation/test entry , 若為y domain的item_id
                    test_data.append([res_2, 1, res[1][-1], res[0]]) #[整個test sequence, 1, 最後一個item_id]
                else :
                    test_data.append([res_2, 0, res[1][-1], res[0]])
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
            d[0] = [[tmp,0] for tmp in d[0]]
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
            # if self.opt['time_encode']:
            #     ts_d = self.time_transformation(ts_d)
            #     ts_xd = self.time_transformation(ts_xd)
            #     ts_yd = self.time_transformation(ts_yd)
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

            y_last = -1
            for id in range(len(y_position)):
                id += 1
                if y_position[-id]:
                    y_last = -id
                    break

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
            # print("-"*100)
            # print("seq:",seq)
            # print("xd:",xd)
            # print("yd:",yd)
            # print("position:",position)
            # print("x_position:",x_position)
            # print("y_position:",y_position)
            # print("ts_d:",ts_d)
            # print("ts_xd:",ts_xd)
            # print("ts_yd:",ts_yd)
            # print("x_last:",x_last)
            # print("y_last:",y_last)
            # print("d[1]:",d[1])
            # print("d[2]:",d[2])
            # print("negative_sample:",negative_sample)
            # print("-"*100)
            # if self.opt['time_encode']:
            #     ts_d = self.encode_time_features(ts_d)
            #     ts_xd = self.encode_time_features(ts_xd)
            #     ts_yd = self.encode_time_features(ts_yd)
            if d[1]:
                processed.append([seq, xd, yd, position, x_position, y_position, ts_d, ts_xd, ts_yd, x_last, y_last, d[1],
                                  d[2]-self.opt["source_item_num"], negative_sample, masked_xd, neg_xd, masked_yd, neg_yd, index,gender])
            else:
                processed.append([seq, xd, yd, position, x_position, y_position, ts_d, ts_xd, ts_yd, x_last, y_last, d[1],
                                  d[2], negative_sample, masked_xd, neg_xd, masked_yd, neg_yd, index, gender])
        return processed

    def preprocess(self):

        def myprint(a):
            for i in a:
                print("%6d" % i, end="")
            print("")
        """ Preprocess the data and convert to ids. """
        processed = []


        if "Enter" in self.filename:
            max_len = 30
            self.opt["maxlen"] = 30
        else:
            max_len = 50
            self.opt["maxlen"] = 50

        for index, d in enumerate(self.train_data): # the pad is needed! but to be careful.
            gender = d[0]
            
            d = d[1]
            d = [[tmp,0] for tmp in d]
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
            #source_item_num = 29207
            #target_item_num = 34886
            #print("ground:",ground) #[49361, 34672, 4342, 60259, 54344, 24905, 5258]
            #print("share_x_ground:",share_x_ground) #[29207, 29207, 4342, 29207, 29207, 24905, 5258]
            #print("share_y_ground:",share_y_ground) #[20154, 5465, 34886, 31052, 25137, 34886, 34886]
            

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
            
            # print("xd:",xd)           #[64093, 64093, 64093, 612, 67, 79, 64093]
            # print("corru_x:",corru_x) #[63544, 43638, 55554, 612, 67, 79, 31056]
            # print("yd:",yd)           #[32261, 35358, 30600, 64093, 64093, 64093, 29472]
            # print("corru_y:",corru_y) #[32261, 35358, 30600, 4814, 23044, 18957, 29472]    
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
                print("pass sequence x")
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
                print("pass sequence y")
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
            # if self.opt['time_encode']:
            #     ts_d = self.encode_time_features(ts_d)
            #     ts_xd = self.encode_time_features(ts_xd)
            #     ts_yd = self.encode_time_features(ts_yd)
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
            processed.append([index, d, xd, yd, position, x_position, y_position, ts_d, ts_xd, ts_yd, ground, share_x_ground, share_y_ground, x_ground, y_ground, ground_mask, 
                                  share_x_ground_mask, share_y_ground_mask, x_ground_mask, y_ground_mask, corru_x, corru_y, masked_xd, neg_xd, masked_yd, neg_yd, augment_xd, augment_yd, gender])
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
        if self.eval!=-1:
            if self.collate_fn:
                batch = self.collate_fn(batch)
                return batch
            else:   
                
                return (torch.LongTensor(batch[0]), torch.LongTensor(batch[1]), torch.LongTensor(batch[2]), torch.LongTensor(batch[3]),torch.LongTensor(batch[4]), torch.LongTensor(batch[5]), torch.tensor(batch[6]), torch.tensor(batch[7]), torch.tensor(batch[8]), torch.tensor(batch[9]),\
                    torch.LongTensor(batch[10]),torch.LongTensor(batch[11]),torch.LongTensor(batch[12]),torch.LongTensor(batch[13]),torch.LongTensor(batch[14]), torch.LongTensor(batch[15]), torch.LongTensor(batch[16]), torch.LongTensor(batch[17]),torch.LongTensor(batch[18]),torch.LongTensor(batch[19]))
        else :
            if self.collate_fn:
                batch = self.collate_fn(batch)
                return batch
            else:
                # print(torch.LongTensor(batch[1]).shape)
                # return (torch.LongTensor(batch[0]), torch.LongTensor(batch[1]))
                return (torch.LongTensor(batch[0]), torch.LongTensor(batch[1]), torch.LongTensor(batch[2]), torch.LongTensor(batch[3]),torch.LongTensor(batch[4]), torch.LongTensor(batch[5]), torch.LongTensor(batch[6]), torch.LongTensor(batch[7]), torch.LongTensor(batch[8]), torch.LongTensor(batch[9]), torch.LongTensor(batch[10]), torch.LongTensor(batch[11]),\
                        torch.LongTensor(batch[12]), torch.LongTensor(batch[13]), torch.LongTensor(batch[14]), torch.LongTensor(batch[15]), torch.LongTensor(batch[16]), torch.LongTensor(batch[17]),\
                        torch.LongTensor(batch[18]),torch.LongTensor(batch[19]),torch.LongTensor(batch[20]),torch.LongTensor(batch[21]),torch.LongTensor(batch[22]),torch.LongTensor(batch[23]),torch.LongTensor(batch[24]),torch.LongTensor(batch[25]),torch.LongTensor(batch[26]),torch.LongTensor(batch[27]),torch.LongTensor(batch[28]))
    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)


