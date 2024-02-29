import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler
import pandas as pd
from datetime import datetime
import numpy as np
import ipdb
import random
import copy
import time
import json
from pathlib import Path
from utils import torch_utils
from model.C2DSR import C2DSR
from model.MoCo import MoCo
from model.NNCL import NNCL
from utils.MoCo_utils import compute_features, compute_embedding_for_target_user,compute_features_for_I2C
from utils.cluster import run_kmeans
from utils.time_transformation import TimeTransformation
from scipy.spatial import distance
from model.C2DSR import *
from utils.loader import DataLoader,NonoverlapDataLoader
from utils.torch_utils import *
from utils.collator import CLDataCollator
class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)
    def forward(self, x, y):
        return self.cos(x, y) / self.temp
class Trainer(object):
    def __init__(self, opt):
        self.opt =opt
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    def update_lr(self, new_lr):  # here should change
        torch_utils.change_lr(self.optimizer, new_lr)

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        # self.model.load_state_dict(checkpoint['model'])
        # self.opt = checkpoint['config']
        # self.model.load_state_dict(checkpoint)
        state_dict = checkpoint['model']
        # print(state_dict['lin_X.weight'])
        # if self.opt['main_task'] == "X":
        #     state_dict.pop('lin_X.weight', None)
        #     state_dict.pop('lin_X.bias', None)
        # elif self.opt['main_task'] == "Y":
        #     state_dict.pop('lin_y.weight', None)
        #     state_dict.pop('lin_y.bias', None)
        # state_dict.pop('lin_PAD.weight', None)
        # state_dict.pop('lin_PAD.bias', None)
        self.model.load_state_dict(state_dict, strict=False)
    def save(self, filename):
        params = {
            'model': self.model.state_dict(),
            'config': self.opt,
        }
        # try:
        # torch.save(self.model.state_dict(), filename)
        torch.save(params, filename)
        print("model saved to {}".format(filename))
        # except BaseException:
            
        #     print("[Warning: Saving failed... continuing anyway.]")
    def unpack_batch_valid(self, batch):
        if self.opt["cuda"]:
            inputs = [Variable(b.cuda()) for b in batch]
            seq = inputs[0]
            x_seq = inputs[1]
            y_seq = inputs[2]
            position = inputs[3]
            x_position = inputs[4]
            y_position = inputs[5]
            ts_d = inputs[6].to(torch.float32)
            ts_xd = inputs[7].to(torch.float32)
            ts_yd = inputs[8].to(torch.float32)
            X_last = inputs[9]
            Y_last = inputs[10]
            XorY = inputs[11]
            ground_truth = inputs[12]
            neg_list = inputs[13]
            masked_xd= inputs[14]
            neg_xd= inputs[15]
            masked_yd= inputs[16]
            neg_yd= inputs[17]
            index = inputs[18]
            gender = inputs[19]
        else:
            inputs = [Variable(b) for b in batch]
            seq = inputs[0]
            x_seq = inputs[1]
            y_seq = inputs[2]
            position = inputs[3]
            x_position = inputs[4]
            y_position = inputs[5]
            ts_d = inputs[6]
            ts_xd = inputs[7]
            ts_yd = inputs[8]
            X_last = inputs[9]
            Y_last = inputs[10]
            XorY = inputs[11]
            ground_truth = inputs[12]
            neg_list = inputs[13]
            masked_xd= inputs[14]
            neg_xd= inputs[15]
            masked_yd= inputs[16]
            neg_yd= inputs[17]
            index = inputs[18]
            gender = inputs[19]
        return index, seq, x_seq, y_seq, position, x_position, y_position, ts_d, ts_xd, ts_yd, X_last, Y_last, XorY, ground_truth, neg_list,masked_xd,neg_xd, masked_yd, neg_yd, gender
    def unpack_batch(self, batch):
        if self.opt["cuda"]:
            try:
                inputs = [Variable(b.cuda()) for b in batch]
            except:
                print("error")
                ipdb.set_trace()
            index = inputs[0]
            seq = inputs[1]
            x_seq = inputs[2]
            y_seq = inputs[3]
            position = inputs[4]
            x_position = inputs[5]
            y_position = inputs[6]
            ts_d = inputs[7].to(torch.float32)
            ts_xd = inputs[8].to(torch.float32)
            ts_yd = inputs[9].to(torch.float32)
            ground = inputs[10]
            share_x_ground = inputs[11]
            share_y_ground = inputs[12]
            x_ground = inputs[13]
            y_ground = inputs[14]
            ground_mask = inputs[15]
            share_x_ground_mask = inputs[16]
            share_y_ground_mask = inputs[17]
            x_ground_mask = inputs[18]
            y_ground_mask = inputs[19]
            corru_x = inputs[20]
            corru_y = inputs[21]
            masked_xd = inputs[22]
            neg_xd = inputs[23]
            masked_yd = inputs[24]
            neg_yd = inputs[25]
            augmented_d = inputs[26]
            augmented_xd = inputs[27]
            augmented_yd = inputs[28]
            gender = inputs[29]
        else:
            inputs = [Variable(b) for b in batch]
            index = inputs[0]
            seq = inputs[1]
            x_seq = inputs[2]
            y_seq = inputs[3]
            position = inputs[4]
            x_position = inputs[5]
            y_position = inputs[6]
            ts_d = inputs[7]
            ts_xd = inputs[8]
            ts_yd = inputs[9]
            ground = inputs[10]
            share_x_ground = inputs[11]
            share_y_ground = inputs[12]
            x_ground = inputs[13]
            y_ground = inputs[14]
            ground_mask = inputs[15]
            share_x_ground_mask = inputs[16]
            share_y_ground_mask = inputs[17]
            x_ground_mask = inputs[18]
            y_ground_mask = inputs[19]
            corru_x = inputs[20]
            corru_y = inputs[21]
            masked_xd = inputs[22]
            neg_xd = inputs[23]
            masked_yd = inputs[24]
            neg_yd = inputs[25]
            augmented_d = inputs[26]
            augmented_xd = inputs[27]
            augmented_yd = inputs[28]
            gender = inputs[29]
        return index, seq, x_seq, y_seq, position, x_position, y_position, ts_d, ts_xd, ts_yd, ground, share_x_ground, share_y_ground, x_ground, y_ground, ground_mask, share_x_ground_mask, share_y_ground_mask, x_ground_mask, y_ground_mask, corru_x, corru_y,masked_xd, neg_xd, masked_yd, neg_yd, augmented_d, augmented_xd, augmented_yd,gender
    def unpack_batch_for_gen(self, batch):
        if self.opt["cuda"]:
            inputs = [Variable(b.cuda()) for b in batch]
            index = inputs[0]
            seq = inputs[1]
            position = inputs[2]
            ts_d = inputs[3]
            ground = inputs[4]
            ground_mask = inputs[5]
            masked_d = inputs[6]
            neg_d = inputs[7]
            target_sentence = inputs[8]
            gender = inputs[9]
        else:
            inputs = [Variable(b) for b in batch]
            index = inputs[0]
            seq = inputs[1]
            position = inputs[2]
            ts_d = inputs[3]
            ground = inputs[4]
            ground_mask = inputs[5]
            masked_d = inputs[6]
            neg_d = inputs[7]
            target_sentence = inputs[8]
            gender = inputs[9]
        return index,seq, position, ts_d, ground,ground_mask ,masked_d, neg_d,target_sentence, gender
class GTrainer(Trainer):
    def __init__(self, opt):
        self.opt = opt
        self.model = Generator(opt,type = self.opt["generate_type"])
        self.CE_criterion = nn.CrossEntropyLoss(ignore_index =self.opt["source_item_num"] + self.opt["target_item_num"] )
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.model.parameters(), opt['lr'])
        if opt['cuda']:
            self.model.cuda()
    def train_batch(self, batch):
        index,seq, position, ts_d, ground, ground_mask ,masked_d, neg_d, target_sentence, gender = self.unpack_batch_for_gen(batch)
        if self.opt['time_encode']:
            mip_pred = self.model(masked_d, position, ts_d)
        else:
            mip_pred = self.model(masked_d, position)
        batch_size, seq_len, item_num = mip_pred.shape
        mlm_predictions = mip_pred.argmax(dim=2)
        flatten_mip_pred = mip_pred.view(-1,item_num)
        flatten_target_sentence = target_sentence.view(batch_size * seq_len)
        if self.opt["generate_type"] == "Y":
            mask = flatten_target_sentence != self.opt["source_item_num"] + self.opt["target_item_num"]
            flatten_target_sentence[mask] = flatten_target_sentence[mask] - self.opt["source_item_num"]
        mip_loss = self.CE_criterion(flatten_mip_pred, flatten_target_sentence)
        self.optimizer.zero_grad()
        mip_loss.backward()
        self.optimizer.step()
        
        return mip_loss.item()
    def train(self ,train_dataloader, val_dataloader):
        global_step = 0
        
        current_lr = self.opt["lr"]
        format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/epoch), lr: {:.6f}'
        num_batch = len(train_dataloader)
        max_steps =  self.opt['pretrain_epoch'] * num_batch

        print("Start training:")
        
        begin_time = time.time()
        
        patience =  self.opt["pretrain_patience"]
        train_pred_loss = []
        val_pred_loss = []
        # start training
        for epoch in range(1, self.opt['pretrain_epoch'] + 1):
            epoch_start_time = time.time()
            self.prediction_loss = 0
            
            self.model.train()
            for i,batch in enumerate(train_dataloader):
                
                global_step += 1
                loss = self.train_batch(batch)
                self.prediction_loss += loss
                
            duration = time.time() - epoch_start_time
            print(format_str.format(datetime.now(), global_step, max_steps, epoch, \
                                            self.opt['pretrain_epoch'],  self.prediction_loss/num_batch, duration, current_lr))
            print("train mip loss:", self.prediction_loss/num_batch)
            train_pred_loss.append(self.prediction_loss/num_batch)
            if epoch%20==0:
                folder = self.opt['model_save_dir'] +f'{epoch}'
                Path(folder).mkdir(parents=True, exist_ok=True)
                self.save(folder + '/model.pt')
            if epoch%500==0:
                self.model.eval()
                with torch.no_grad():
                    self.val_prediction_loss = 0
                    for i,batch in enumerate(val_dataloader):
                        index, seq, position, ts_d, ground, ground_mask ,masked_d, neg_d, target_sentence, gender = self.unpack_batch_for_gen(batch)
                        mip_pred = self.model(seq,position)
                        batch_size, seq_len, item_num = mip_pred.shape
                        flatten_mip_res = mip_pred.view(-1,item_num)
                        flatten_target_sentence = target_sentence.view(batch_size * seq_len)
                        mip_loss = self.CE_criterion(flatten_mip_res,flatten_target_sentence)
                        self.val_prediction_loss += mip_loss.item()
                        # ipdb.set_trace()
                    print("-"*50)
                    print(f"Start validation at epoch {epoch}:")
                    print("validation mip loss:", self.val_prediction_loss/len(val_dataloader))
                    print("-"*50)
                    val_pred_loss.append(self.val_prediction_loss/len(val_dataloader))
                    if self.val_prediction_loss/len(val_dataloader) < min(val_pred_loss):
                        patience =  self.opt["pretrain_patience"]
                    else:
                        patience -= 1
                        print("Early stop counter:", 5-patience)
                        if patience == 0:
                            print("Early stop at epoch", epoch)
                            self.save(self.opt['model_save_dir'] + '/best_model.pt')
                            break
class Pretrainer(Trainer):
    def __init__(self, opt, adj = None, adj_single = None):
        self.opt = opt
        if opt["model"] == "C2DSR":
            self.model = C2DSR(opt, adj, adj_single)
        else:
            print("please select a valid model")
            exit(0)
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.model.parameters(), opt['lr'])
        self.CE_criterion = nn.CrossEntropyLoss()
        self.pred_criterion = nn.CrossEntropyLoss(reduction='none')
        self.BCE_criterion = nn.BCELoss(reduction='none')
        self.BCE_criterion_reduction = nn.BCELoss()
        self.pretrain_loss = 0
        self.pred_loss = 0
        self.MoCo_X = MoCo(opt, self.model, dim = opt['hidden_units'], r = opt['r'], m = opt['m'], T = opt['temp'], mlp = opt['mlp'], domain = "X")
        self.MoCo_Y = MoCo(opt, self.model, dim = opt['hidden_units'], r = opt['r'], m = opt['m'], T = opt['temp'], mlp = opt['mlp'], domain = "Y")
        self.MoCo_mixed = MoCo(opt, self.model, dim = opt['hidden_units'], r = opt['r'], m = opt['m'], T = opt['temp'], mlp = opt['mlp'], domain = "mixed")
        
        self.mip_norm = nn.Linear(opt['hidden_units'], opt['hidden_units'])
        
        if opt['cuda']:
            self.model.cuda()
            self.MoCo_X.cuda()
            self.MoCo_Y.cuda()
            self.MoCo_mixed.cuda()
            self.CE_criterion.cuda()
            self.pred_criterion.cuda()
            self.mip_norm.cuda()
            self.BCE_criterion.cuda()
        self.pooling = opt['pooling']
        self.format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/epoch)'
        
    def MoCo_train(self, MoCo_model, augmented_seq, cluster_result = None, index = None, task = "in-domain", train_mode ="train", ts =None):
        if self.opt['augment_type'] == "dropout":
            logits, labels, proto_logits, proto_labels, equivariance_loss, time_ssl_task = MoCo_model(seq_q = augmented_seq, seq_k=augmented_seq, is_eval=False, cluster_result= cluster_result, index=index, task=task, train_mode = train_mode, ts = ts)
        else:
            logits, labels, proto_logits, proto_labels, equivariance_loss, time_ssl_task = MoCo_model(seq_q = augmented_seq[:,0,:], seq_k=augmented_seq[:,1,:], is_eval=False, cluster_result= cluster_result, index=index, train_mode = train_mode, task=task, ts =ts)
        if task == "in-domain":
            cl_loss = self.CE_criterion(logits, labels)
            if self.opt['time_encode']:
                time_ssl = self.CE_criterion(time_ssl_task['speed_classification'][0], time_ssl_task['speed_classification'][1]) + \
                        self.BCE_criterion_reduction(time_ssl_task['direction_classification'][0], time_ssl_task['direction_classification'][1])
                loss = cl_loss + self.opt['equivariance_weight']*equivariance_loss + self.opt['time_weight']*time_ssl
            else:
                loss = cl_loss
        else:
            loss = 0
        if proto_logits is not None:
            loss_proto = 0
            if task == "in-domain":
                for proto_out, proto_target in zip(proto_logits, proto_labels):
                    loss_proto += self.CE_criterion(proto_out, proto_target)  
            elif task == "cross-domain":
                for proto_out in proto_logits:
                    proto_out = proto_out + 1e-8 # avoid value equal to 0
                    loss_proto += -torch.sum(proto_out*torch.log(proto_out))
            loss_proto /= len(self.opt['num_cluster']) 
            loss += loss_proto
        if self.opt['time_encode']:
            return loss, time_ssl_task['speed_classification'][2], time_ssl_task['direction_classification'][2]
        else:
            return loss
    def get_sequence_embedding(self, data, encoder, item_embed):

        non_zero_mask = (data != (self.opt["source_item_num"] +self.opt["target_item_num"])).long()
        position_id = non_zero_mask.cumsum(dim=1) * non_zero_mask
        seqs = item_embed(data)
        seq_feature = encoder(data, seqs, position_id,  causality_mask = False)
        # if self.pooling == "bert":
        #     out = seq_feature[:,0,:]
        # elif self.pooling == "ave":
        #     out = torch.sum(seq_feature, dim=1)/torch.sum(non_zero_mask, dim=1).unsqueeze(-1)
        return seq_feature
    def get_item_score(self, sequence_output, target_item):
        sequence_output = self.mip_norm(sequence_output.view([-1, self.opt['hidden_units']])) # [B*L H]
        target_item = target_item.view([-1, self.opt['hidden_units']]) # [B*L H]
        score = torch.mul(sequence_output, target_item) # [B*L H]
        return torch.sigmoid(torch.sum(score, -1)) # [B*L]
    def masked_item_prediction(self, pos_item, neg_item, masked_seq, enocoder, item_embed):
        sequence_output = self.get_sequence_embedding(masked_seq, enocoder, item_embed) 
        pos_item_embs = item_embed(pos_item)
        neg_item_embs = item_embed(neg_item)
        pos_score = self.get_item_score(sequence_output, pos_item_embs)
        neg_score = self.get_item_score(sequence_output, neg_item_embs)
        distance = torch.sigmoid(pos_score - neg_score)
        mip_loss = self.BCE_criterion(distance, torch.ones_like(distance, dtype=torch.float32))
        mip_mask = (masked_seq ==self.opt['itemnum']-1).float()
        mip_loss = torch.sum(mip_loss * mip_mask.flatten())
        return mip_loss
    def train_batch(self, epoch, batch, i, cluster_result, mode = "train"):
        self.model.train()
        self.optimizer.zero_grad()
        cluster_result_X, cluster_result_Y, cluster_result_cross = cluster_result[0], cluster_result[1], cluster_result[2]
        index, seq, x_seq, y_seq, position, x_position, y_position, ts_d, ts_xd, ts_yd, ground, share_x_ground, share_y_ground, x_ground, y_ground, ground_mask, share_x_ground_mask, share_y_ground_mask, x_ground_mask, y_ground_mask, corru_x, corru_y, masked_xd, neg_xd, masked_yd, neg_yd,augmented_d, augmented_xd,augmented_yd,gender = self.unpack_batch(batch)
        
        
        ssl_loss = torch.tensor(0, dtype = torch.float32).cuda()
        if self.opt["ssl"]=="proto_CL":
            if self.opt['time_encode']:
                MoCo_loss_xd, speed_acc_xd, direction_acc_xd = self.MoCo_train(self.MoCo_X, augmented_xd, cluster_result = cluster_result_X, index = index, task = "in-domain",ts =ts_xd)
                MoCo_loss_yd, speed_acc_yd, direction_acc_yd = self.MoCo_train(self.MoCo_Y, augmented_yd, cluster_result = cluster_result_Y, index = index, task = "in-domain",ts =ts_yd)
                # MoCo_loss_mixed = self.MoCo_train(epoch,self.MoCo_mixed, augmented_d, cluster_result = cluster_result_cross, index=index, task = "in-domain",ts = ts_d)
            else:
                MoCo_loss_xd = self.MoCo_train(self.MoCo_X, augmented_xd, cluster_result = cluster_result_X, index = index, task = "in-domain")
                MoCo_loss_yd = self.MoCo_train(self.MoCo_Y, augmented_yd, cluster_result = cluster_result_Y, index = index, task = "in-domain")
                # MoCo_loss_mixed = self.MoCo_train(epoch,self.MoCo_mixed, augmented_d, cluster_result = None, index=index, task = "in-domain")
            
            if cluster_result_cross is not None:
                if self.opt['time_encode']:
                    MoCo_loss_xd_cross = self.MoCo_train(self.MoCo_X, augmented_xd, cluster_result = cluster_result_cross, index=None, task = "cross-domain",ts =ts_xd)
                    MoCo_loss_yd_cross = self.MoCo_train(self.MoCo_Y, augmented_yd, cluster_result = cluster_result_cross, index=None, task = "cross-domain",ts =ts_xd)
                else:
                    MoCo_loss_xd_cross = self.MoCo_train(self.MoCo_X, augmented_xd, cluster_result = cluster_result_cross, index=None, task = "cross-domain")
                    MoCo_loss_yd_cross = self.MoCo_train(self.MoCo_Y, augmented_yd, cluster_result = cluster_result_cross, index=None, task = "cross-domain")
                ssl_loss += self.opt['cross_weight']*(MoCo_loss_xd_cross + MoCo_loss_yd_cross) + MoCo_loss_xd + MoCo_loss_yd
            else:
                ssl_loss += MoCo_loss_xd + MoCo_loss_yd
            
        if self.opt['ssl']=='mask_prediction':
            mip_loss_xd = self.masked_item_prediction(x_seq, neg_xd, masked_xd, self.model.encoder_X, self.model.item_emb_X)
            mip_loss_yd = self.masked_item_prediction(y_seq, neg_yd, masked_yd, self.model.encoder_Y, self.model.item_emb_Y)
            ssl_loss += mip_loss_xd + mip_loss_yd
            
        if self.opt['training_mode'] == 'joint_pretrain':
            if self.opt['time_encode']:
                
                seqs_fea, x_seqs_fea, y_seqs_fea = self.model(seq, x_seq, y_seq, position, x_position, y_position, ts_d, ts_xd, ts_yd)
            else:
                
                seqs_fea, x_seqs_fea, y_seqs_fea = self.model(seq, x_seq, y_seq, position, x_position, y_position)
            used = 10
            ground = ground[:,-used:]
            ground_mask = ground_mask[:, -used:]
            x_ground = x_ground[:, -used:]
            x_ground_mask = x_ground_mask[:, -used:]
            y_ground = y_ground[:, -used:]
            y_ground_mask = y_ground_mask[:, -used:]

            specific_x_result = self.model.lin_X(x_seqs_fea[:, -used:])  # b * seq * X_num
            specific_x_pad_result = self.model.lin_PAD(x_seqs_fea[:, -used:])  # b * seq * 1
            specific_x_result = torch.cat((specific_x_result, specific_x_pad_result), dim=-1)
            specific_y_result = self.model.lin_Y( y_seqs_fea[:, -used:])       # b * seq * Y_num
            specific_y_pad_result = self.model.lin_PAD(y_seqs_fea[:, -used:])  # b * seq * 1
            specific_y_result = torch.cat((specific_y_result, specific_y_pad_result), dim=-1)
            x_loss = self.pred_criterion(
                specific_x_result.reshape(-1, self.opt["source_item_num"] + 1),
                x_ground.reshape(-1))  # b * seq
            y_loss = self.pred_criterion(
                specific_y_result.reshape(-1, self.opt["target_item_num"] + 1),
                y_ground.reshape(-1))  # b * seq
            x_loss = (x_loss * (x_ground_mask.reshape(-1))).mean()
            y_loss = (y_loss * (y_ground_mask.reshape(-1))).mean()
            
            #cross domain
            share_x_ground = share_x_ground[:, -used:]
            share_x_ground_mask = share_x_ground_mask[:, -used:]
            share_y_ground = share_y_ground[:, -used:]
            share_y_ground_mask = share_y_ground_mask[:, -used:]
            share_x_result =  self.model.lin_X(seqs_fea[:,-used:]) # b * seq * X_num
            share_y_result = self.model.lin_Y(seqs_fea[:, -used:])  # b * seq * Y_num
            share_pad_result = self.model.lin_PAD(seqs_fea[:, -used:])  # b * seq * 1 #最後一維，即padding，score要是零
            share_trans_x_result = torch.cat((share_x_result, share_pad_result), dim=-1)
            share_trans_y_result = torch.cat((share_y_result, share_pad_result), dim=-1)
            x_share_loss = self.pred_criterion(
                share_trans_x_result.reshape(-1, self.opt["source_item_num"] + 1),
                share_x_ground.reshape(-1))  # b * seq
            y_share_loss = self.pred_criterion(
                share_trans_y_result.reshape(-1, self.opt["target_item_num"] + 1),
                share_y_ground.reshape(-1))
            x_share_loss = (x_share_loss * (share_x_ground_mask.reshape(-1))).mean() #只取預測x的部分
            y_share_loss = (y_share_loss * (share_y_ground_mask.reshape(-1))).mean() #只取預測y的部分
            mixed_loss = x_share_loss + y_share_loss
            loss = ssl_loss + x_loss + y_loss + mixed_loss
        else:
            loss = ssl_loss
        
        loss.backward()
        self.optimizer.step()
        if self.opt["time_encode"]:
            accuracy = {"x":[speed_acc_xd,direction_acc_xd],"y":[speed_acc_yd,direction_acc_yd]}
        else:
            accuracy = None
        if self.opt['training_mode']=='joint_pretrain':
            return ssl_loss.item(), x_loss.item() , y_loss.item(), accuracy
        else:
            return ssl_loss.item(), 0, 0, accuracy
    def valid_batch(self, epoch, batch, i, cluster_result):
        self.model.eval()
        with torch.no_grad():
            cluster_result_X, cluster_result_Y, cluster_result_cross = cluster_result[0], cluster_result[1], cluster_result[2]
            index, seq, x_seq, y_seq, position, x_position, y_position, ts_d, ts_xd, ts_yd, X_last, Y_last, XorY, ground_truth, neg_list, masked_xd, neg_xd, masked_yd, neg_yd, gender  = self.unpack_batch_valid(batch)
            ssl_loss = torch.tensor(0, dtype = torch.float32).cuda()
            acc = None
            if self.opt["ssl"]=="proto_CL":
                if self.opt['time_encode']:
                    MoCo_loss_xd, speed_acc_xd, direction_acc_xd = self.MoCo_train(self.MoCo_X, augmented_xd, cluster_result = cluster_result_X, index = index, task = "in-domain", train_mode = "valid",ts =ts_xd)
                    MoCo_loss_yd, speed_acc_yd, direction_acc_yd = self.MoCo_train(self.MoCo_Y, augmented_yd, cluster_result = cluster_result_Y, index = index, task = "in-domain", train_mode = "valid",ts =ts_yd)
                    # MoCo_loss_mixed = self.MoCo_train(epoch,self.MoCo_mixed, augmented_d, cluster_result = cluster_result_cross, index=index, task = "in-domain",ts = ts_d)
                    acc = {"x":[speed_acc_xd,direction_acc_xd],"y":[speed_acc_yd,direction_acc_yd]}
                else:
                    MoCo_loss_xd = self.MoCo_train(self.MoCo_X, augmented_xd, cluster_result = None, index = index, task = "in-domain", train_mode = "valid")
                    MoCo_loss_yd = self.MoCo_train(self.MoCo_Y, augmented_yd, cluster_result = None, index = index, task = "in-domain", train_mode = "valid")
                    # MoCo_loss_mixed = self.MoCo_train(epoch,self.MoCo_mixed, augmented_d, cluster_result = None, index=index, task = "in-domain")
                if cluster_result_cross is not None:
                    if self.opt['time_encode']:
                        MoCo_loss_xd_cross = self.MoCo_train(self.MoCo_X, augmented_xd, cluster_result = cluster_result_cross, index=None, task = "cross-domain", train_mode = "valid",ts =ts_xd)
                        MoCo_loss_yd_cross = self.MoCo_train(self.MoCo_Y, augmented_yd, cluster_result = cluster_result_cross, index=None, task = "cross-domain", train_mode = "valid",ts =ts_xd)
                    else:
                        MoCo_loss_xd_cross = self.MoCo_train(self.MoCo_X, augmented_xd, cluster_result = cluster_result_cross, index=None, task = "cross-domain", train_mode = "valid")
                        MoCo_loss_yd_cross = self.MoCo_train(self.MoCo_Y, augmented_yd, cluster_result = cluster_result_cross, index=None, task = "cross-domain", train_mode = "valid")
                    ssl_loss += self.opt['cross_weight']*(MoCo_loss_xd_cross + MoCo_loss_yd_cross) + MoCo_loss_xd + MoCo_loss_yd
                else:
                    ssl_loss +=  MoCo_loss_xd + MoCo_loss_yd
            
            if self.opt['ssl']=='mask_prediction':
                mip_loss_xd = self.masked_item_prediction(x_seq, neg_xd, masked_xd, self.model.encoder_X, self.model.item_emb_X)
                mip_loss_yd = self.masked_item_prediction(y_seq, neg_yd, masked_yd, self.model.encoder_Y, self.model.item_emb_Y)
                ssl_loss += mip_loss_xd + mip_loss_yd
            return ssl_loss.item(),acc
    def generate_cluster(self,dataloader):
        cluster_result_X, cluster_result_Y, cluster_result_cross = None, None, None
        # x_domain
        features_X = compute_features(self.opt, dataloader, self.MoCo_X, domain = 'X')
        features_X[torch.norm(features_X,dim=1)>1.5] /= 2 #account for the few samples that are computed twice  
        features_X = features_X.numpy()
        cluster_result_X = run_kmeans(features_X, self.opt) 
        # y_domain
        features_Y = compute_features(self.opt, dataloader, self.MoCo_Y, domain = 'Y')
        features_Y[torch.norm(features_Y,dim=1)>1.5] /= 2 
        features_Y = features_Y.numpy()
        cluster_result_Y = run_kmeans(features_Y, self.opt)
        # mixed domain
        if self.opt['mixed_included']:
            features_cross = compute_features(self.opt, dataloader, self.model, domain = 'mixed')
            features_cross[torch.norm(features_cross,dim=1)>1.5] /= 2 
            features_cross = features_cross.numpy()
            cluster_result_cross = run_kmeans(features_cross, self.opt)
        return cluster_result_X, cluster_result_Y, cluster_result_cross
    def train(self, num_epoch, train_dataloader, val_dataloader = None):
        global_step = 0
        ssl_loss_list = []
        X_prediction_loss_list = []
        Y_prediction_loss_list = []
        val_ssl_loss_list = []
        last_val_loss = float("inf")
        patience = 0
        for epoch in range(1, num_epoch+ 1):
            pretrain_loss = 0
            X_prediction_loss = 0
            Y_prediction_loss = 0
            speed_acc_xd = 0
            direction_acc_xd = 0
            speed_acc_yd = 0
            direction_acc_yd = 0
            epoch_start_time = time.time()
            cluster_result_X = None
            cluster_result_Y = None
            cluster_result_cross = None          
            if self.opt['ssl'] == "proto_CL":
                if epoch >= self.opt['warmup_epoch']:
                    cluster_result_X, cluster_result_Y, cluster_result_cross = self.generate_cluster(train_dataloader)
            for i,batch in enumerate(train_dataloader):
                global_step += 1
                ssl_loss, X_pred_loss, Y_pred_loss ,accuracy = self.train_batch(epoch, batch, i, cluster_result = (cluster_result_X, cluster_result_Y, cluster_result_cross))
                pretrain_loss += ssl_loss
                X_prediction_loss += X_pred_loss 
                Y_prediction_loss += Y_pred_loss
                if accuracy is not None:
                    speed_acc_xd += accuracy['x'][0]
                    direction_acc_xd += accuracy['x'][1]
                    speed_acc_yd += accuracy['y'][0]
                    direction_acc_yd += accuracy['y'][1]
            if epoch%10==0:
                save_path = "pretrain_models/" +f"{str(self.opt['data_dir'])}/"  + str(self.opt['id']) +f"/{str(epoch)}"
                Path(save_path).mkdir(parents=True, exist_ok=True)
                self.save(save_path + "/pretrain_model.pt")
            ssl_loss_list.append(pretrain_loss/len(train_dataloader))
            X_prediction_loss_list.append(X_prediction_loss/len(train_dataloader))   
            Y_prediction_loss_list.append(Y_prediction_loss/len(train_dataloader))
            duration = time.time() - epoch_start_time
            num_batch = len(train_dataloader)
            print(self.format_str.format(datetime.now(), global_step, self.opt['num_epoch'] * num_batch, epoch, \
                                            self.opt['num_epoch'], (pretrain_loss+X_prediction_loss+Y_prediction_loss)/num_batch, duration))
            print("SSL loss:", pretrain_loss/num_batch)
            print("X Prediction loss:", X_prediction_loss/num_batch)
            print("Y Prediction loss:", Y_prediction_loss/num_batch)
            print("speed_acc_xd:", speed_acc_xd/num_batch)
            print("direction_acc_xd:", direction_acc_xd/num_batch)
            print("speed_acc_yd:", speed_acc_yd/num_batch)
            print("direction_acc_yd:", direction_acc_yd/num_batch)
            # do validation for ssl
            if epoch%self.opt['valid_epoch']==0:
                if val_dataloader:
                    val_pretrain_loss = 0
                    val_speed_acc_xd = 0
                    val_direction_acc_xd = 0
                    val_speed_acc_yd = 0
                    val_direction_acc_yd = 0
                    cluster_result_X = None
                    cluster_result_Y = None
                    cluster_result_cross = None
                    for i,batch in enumerate(val_dataloader):
                        ssl_loss, accuracy = self.valid_batch(epoch, batch, i, cluster_result = (cluster_result_X, cluster_result_Y, cluster_result_cross))
                        val_pretrain_loss += ssl_loss
                        if accuracy is not None:
                            val_speed_acc_xd += accuracy['x'][0]
                            val_direction_acc_xd += accuracy['x'][1]
                            val_speed_acc_yd += accuracy['y'][0]
                            val_direction_acc_yd += accuracy['y'][1]
                    print("----------------------------------------")
                    print("validation SSL loss:", val_pretrain_loss/len(val_dataloader))
                    if accuracy is not None:
                        print("validation speed_acc_xd:", val_speed_acc_xd/len(val_dataloader))
                        print("validation direction_acc_xd:", val_direction_acc_xd/len(val_dataloader))
                        print("validation speed_acc_yd:", val_speed_acc_yd/len(val_dataloader)) 
                        print("validation direction_acc_yd:", val_direction_acc_yd/len(val_dataloader))
                    print("----------------------------------------")
                    val_ssl_loss_list.append(val_pretrain_loss/len(val_dataloader))
                    if last_val_loss>(val_pretrain_loss/len(val_dataloader)):
                        last_val_loss = val_pretrain_loss/len(val_dataloader)
                        patience = 0
                    else:
                        patience += 1
                        print("patience:",patience)
                        if patience>=self.opt['pretrain_patience']:
                            print("\033[01;32m early stop for SSL !!!!\033[0m\n")
                            save_path = "pretrain_models/" +f"{str(self.opt['data_dir'])}/"  + str(self.opt['id']) +f"/best_pretrain_model_epoch{str(epoch)}"
                            # print("save_path:",save_path)
                            Path(save_path).mkdir(parents=True, exist_ok=True)
                            self.save(save_path + "/pretrain_model.pt")
                            break
        loss_save_path = f"./loss/{self.opt['data_dir']}/{self.opt['id']}"
        print(f"write loss into path {loss_save_path}")
        Path(loss_save_path).mkdir(parents=True, exist_ok=True)
        np.save(f"{loss_save_path}/ssl_loss.npy", np.array(ssl_loss_list))
        np.save(f"{loss_save_path}/X_prediction_loss.npy", np.array(X_prediction_loss_list))
        np.save(f"{loss_save_path}/Y_prediction_loss.npy", np.array(Y_prediction_loss_list))
        np.save(f"{loss_save_path}/val_ssl_loss.npy", np.array(val_ssl_loss_list))
class CDSRTrainer(Trainer):
    def __init__(self, opt, adj = None, adj_single = None):
        self.opt = opt
        if opt["model"] == "C2DSR":
            self.model = C2DSR(opt, adj, adj_single)
        else:
            print("please select a valid model")
            exit(0)

        self.mi_loss = 0
        self.time_CL_loss = 0
        self.augmentation_based_CL_loss = 0
        self.proto_CL_loss = 0
        self.prediction_loss = 0
        self.pull_loss = 0
        self.NNCL_loss = 0
        self.group_CL_loss = 0
        self.I2C_loss = 0
        self.BCE_criterion = nn.BCELoss(reduction='none')
        self.CS_criterion = nn.CrossEntropyLoss(reduction='none')
        self.CL_criterion = nn.CrossEntropyLoss()
        self.proto_criterion = nn.CrossEntropyLoss()
        self.val_criterion = nn.CrossEntropyLoss(reduction='none')
        
        self.MoCo_X = MoCo(opt, self.model, dim = opt['hidden_units'], r = opt['r'], m = opt['m'], T = opt['temp'], mlp = opt['mlp'], domain = "X")
        self.MoCo_Y = MoCo(opt, self.model, dim = opt['hidden_units'], r = opt['r'], m = opt['m'], T = opt['temp'], mlp = opt['mlp'], domain = "Y")
        self.MoCo_mixed =MoCo(opt, self.model, dim = opt['hidden_units'], r = opt['r'], m = opt['m'], T = opt['temp'], mlp = opt['mlp'], domain = "mixed")
        self.NNCL = NNCL(opt,self.model, dim = opt['hidden_units'], r = opt['r'], m = opt['m'], T = opt['temp'], mlp = opt['mlp'])
        ###adversarial learning###
        self.gender_discriminator = GenderDiscriminator(self.opt)
        self.gender_discriminator_pretrained =False
        self.lambda_ = 0.1
        
        ###Istance to cluster Contrastive Learning###
        if self.opt['ssl'] in ['interest_cluster','both']:
            self.male_cluster = ClusterRepresentation(opt,opt['hidden_units'], opt['num_cluster'][0], topk = opt['topk_cluster'])
            self.female_cluster = ClusterRepresentation(opt,opt['hidden_units'], opt['num_cluster'][1], topk = opt['topk_cluster'])
            # self.cluster = ClusterRepresentation(opt,opt['hidden_units'], opt['num_cluster'][2], topk = opt['topk_cluster'])
        if opt['cuda']:
            self.model = self.model.cuda()
            self.BCE_criterion = self.BCE_criterion.cuda()
            self.CS_criterion = self.CS_criterion.cuda()
            self.CL_criterion = self.CL_criterion.cuda()
            self.proto_criterion = self.proto_criterion.cuda()
            self.MoCo_X = self.MoCo_X.cuda()
            self.MoCo_Y = self.MoCo_Y.cuda()
            self.MoCo_mixed = self.MoCo_mixed.cuda()
            self.NNCL = self.NNCL.cuda()
            self.gender_discriminator = self.gender_discriminator.cuda()
           
            if self.opt['ssl'] in ['interest_cluster','both']:
                self.male_cluster = self.male_cluster.cuda()
                self.female_cluster = self.female_cluster.cuda()
                # self.cluster = self.cluster.cuda()
        if self.opt['param_group'] : 
            param_name = []
            for name, param in self.model.named_parameters():
                param_name.append(name)
            target_param_name = [s for s in param_name if "encoder_X" in s]
            print("target_param_name:",target_param_name)
            group1 =[p for n, p in self.model.named_parameters() if n not in target_param_name and p.requires_grad]
            group2 =[p for n, p in self.model.named_parameters() if n in target_param_name and p.requires_grad]
            self.optimizer = torch_utils.get_optimizer(opt['optim'],
                                                    [{'params': group1, 'lr': opt['lr']},
                                                        {'params': group2, 'lr': opt['lr']*0.01}],
                                                    opt['lr'])
        else:
            if self.opt['training_mode'] == 'joint_learn' and self.opt['ssl'] in ['interest_cluster','both']: 
                self.optimizer = torch_utils.get_optimizer(opt['optim'], list(self.model.parameters())+list(self.male_cluster.parameters())+list(self.female_cluster.parameters()), opt['lr'])
                # self.optimizer = torch_utils.get_optimizer(opt['optim'], list(self.model.parameters())+list(self.cluster.parameters()), opt['lr'])

            else:
                self.optimizer = torch_utils.get_optimizer(opt['optim'],self.model.parameters(), opt['lr'])
        self.d_optimizer = torch_utils.get_optimizer(opt['optim'], self.gender_discriminator.parameters(), opt['lr'])
        self.pooling = opt['pooling']
        self.sim = Similarity(opt['temp'])
        
        
        
    def get_dot_score(self, A_embedding, B_embedding):
        output = (A_embedding * B_embedding).sum(dim=-1)
        return output

    def unpack_batch_predict(self, batch):
        if self.opt["cuda"]:
            inputs = [Variable(b.cuda()) for b in batch]
            seq = inputs[0]
            x_seq = inputs[1]
            y_seq = inputs[2]
            position = inputs[3]
            x_position = inputs[4]
            y_position = inputs[5]
            ts_d = inputs[6].to(torch.float32)
            ts_xd = inputs[7].to(torch.float32)
            ts_yd = inputs[8].to(torch.float32)
            X_last = inputs[9]
            Y_last = inputs[10]
            XorY = inputs[11]
            ground_truth = inputs[12]
            neg_list = inputs[13]
            index = inputs[14]
            masked_xd= inputs[15]
            neg_xd= inputs[16]
            masked_yd= inputs[17]
            neg_yd= inputs[18]
            gender = inputs[19]
            x_last_3 = inputs[20]
            y_last_3 = inputs[21]
        else:
            inputs = [Variable(b) for b in batch]
            seq = inputs[0]
            x_seq = inputs[1]
            y_seq = inputs[2]
            position = inputs[3]
            x_position = inputs[4]
            y_position = inputs[5]
            ts_d = inputs[6]
            ts_xd = inputs[7]
            ts_yd = inputs[8]
            X_last = inputs[9]
            Y_last = inputs[10]
            XorY = inputs[11]
            ground_truth = inputs[12]
            neg_list = inputs[13]
            index = inputs[14]
            masked_xd= inputs[15]
            neg_xd= inputs[16]
            masked_yd= inputs[17]
            neg_yd= inputs[18]
            gender = inputs[19]
            x_last_3 = inputs[20]
            y_last_3 = inputs[21]
        return index, seq, x_seq, y_seq, position, x_position, y_position, ts_d, ts_xd, ts_yd, X_last, Y_last, XorY, ground_truth, neg_list, masked_xd, neg_xd, masked_yd, neg_yd,gender, x_last_3,y_last_3

    
    def HingeLoss(self, pos, neg):
        gamma = torch.tensor(self.opt["margin"])
        if self.opt["cuda"]:
            gamma = gamma.cuda()
        return F.relu(gamma - pos + neg).mean()
    def isnan(self,x):
        return torch.any(torch.isnan(x))
    def my_index_select(self, memory, index):#從memory中選擇index的item
        tmp = list(index.size()) + [-1]
        index = index.reshape(-1)
        ans = torch.index_select(memory, 0, index)
        ans = ans.reshape(tmp)
        return ans
    def mask_correlated_samples(self, seq_len):
        N =seq_len.sum()
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        # print("seq_len:",seq_len.shape)
        # print(mask.shape)
        # cum_seq_len = torch.cat([torch.tensor([0]).cuda(), seq_len.cumsum(dim=0)])
        # print("cum_seq_len:",cum_seq_len.shape)
        # print(batch_size)
        # for  i in range(batch_size):
        #     mask[cum_seq_len[i]:cum_seq_len[i+1], cum_seq_len[i]:cum_seq_len[i+1]] = 0
        return mask
    def ssl_loss(self, item, near_item, seq_len, temp, batch_size):
        N = seq_len.sum()
        # print('0', self.isnan(item), self.isnan(near_item))
        item = F.normalize(item) #[B*15, 64] =>every x item 
        near_item = F.normalize(near_item) #[B*15, 64] =>every near item sum embedding with respect to x
        # print('1', self.isnan(item), self.isnan(near_item))
        sim = torch.mm(item, near_item.T)/temp
        pos = torch.diag(sim, 0) 
        if batch_size != self.opt['batch_size']:
            mask = self.mask_correlated_samples(seq_len)
        else:
            mask = self.mask_correlated_samples(seq_len)
        neg = sim[mask].reshape(N,-1)
        labels = torch.zeros(N).to(pos.device).long()
        logits = torch.cat([pos.unsqueeze(1), neg], dim=1)
        # print('2', self.isnan(logits))
        loss = F.cross_entropy(logits, labels)
        return loss
    def time_CL(self,i_xd,near_xd ,i_yd,near_yd ):
        # get embedding
        item_embed_xd = self.model.item_emb(i_xd)    #[B, 15 , 64]
        near_embed_xd = self.model.item_emb(near_xd) #[B, 15 , 15, 64]
        item_embed_yd = self.model.item_emb(i_yd)    #[B, 15 , 15, 64]
        near_embed_yd = self.model.item_emb(near_yd) #[B, 15 , 15, 64]
        #get mask
        item_mask_xd = i_xd!= self.opt["itemnum"] - 1    #[B, 15]
        near_mask_xd = near_xd!= self.opt["itemnum"] - 1 #[B, 15 , 15]
        item_mask_yd = i_yd!= self.opt["itemnum"] - 1    #[B, 15]
        near_mask_yd = near_yd!= self.opt["itemnum"] - 1 #[B, 15 , 15]
        # X domain : get real item embedding
        item_embed_xd = item_embed_xd[item_mask_xd].reshape(-1,self.opt['hidden_units']) #[X, 64]
        sum_near_embed_xd =torch.sum(near_embed_xd*near_mask_xd.float().unsqueeze(-1), dim=2) #[B, 15, 64]
        mean_near_embed_xd = sum_near_embed_xd/torch.sum(near_mask_xd+1e-5,dim=2).unsqueeze(-1) #[B, 15, 64]
        mean_near_embed_xd = mean_near_embed_xd[item_mask_xd].reshape(-1,self.opt['hidden_units']) #[X, 64]
        ssl_loss_xd = self.ssl_loss(item = item_embed_xd, near_item = mean_near_embed_xd, seq_len =item_mask_xd.sum(-1),  temp = self.opt['temp'], batch_size = item_mask_xd.shape[0])
        # Y domain : get real item embedding
        item_embed_yd = item_embed_yd[item_mask_yd].reshape(-1,self.opt['hidden_units']) #[X, 64]
        sum_near_embed_yd =torch.sum(near_embed_yd*near_mask_yd.float().unsqueeze(-1), dim=2) #[B, 15, 64]
        mean_near_embed_yd = sum_near_embed_yd/torch.sum(near_mask_yd+1e-5,dim=2).unsqueeze(-1) #[B, 15, 64]
        mean_near_embed_yd = mean_near_embed_yd[item_mask_yd].reshape(-1,self.opt['hidden_units']) #[X, 64]
        # print('mean_near_embed_yd', self.isnan(mean_near_embed_yd))
        ssl_loss_yd = self.ssl_loss(item = item_embed_yd, near_item = mean_near_embed_yd, seq_len =item_mask_yd.sum(-1),  temp = self.opt['temp'], batch_size = item_mask_yd.shape[0])
        # print(ssl_loss_yd)
        return ssl_loss_xd, ssl_loss_yd
    def get_embedding_for_ssl(self, data, encoder, item_embed, CL_projector, 
                          encoder_causality_mask = False, add_graph_node=False, graph_embedding = None):
        batch_size = data.shape[0]
        device = data.device
        non_zero_mask = (data != (self.opt["source_item_num"] +self.opt["target_item_num"])).long()
        position_id = non_zero_mask.cumsum(dim=1) * non_zero_mask
        if add_graph_node:
            if self.pooling == "bert":
                node_feature = torch.cat((torch.zeros(data.size(0),1,self.opt['hidden_units'],device = device),self.my_index_select(graph_embedding, data[:,1:])),1)
                seqs = item_embed(data) + node_feature #[B*2, seq_len, hidden_units]
            elif self.pooling == "ave":
                seqs = item_embed(data) + self.my_index_select(graph_embedding, data)
            if self.opt['augment_type'] == "dropout":
                seq_feature1 = encoder(data, seqs, position_id, causality_mask = encoder_causality_mask).unsqueeze(1)
                seq_feature2 = encoder(data, seqs, position_id, causality_mask = encoder_causality_mask).unsqueeze(1)
                seq_feature = torch.cat([seq_feature1, seq_feature2], dim=1).view(batch_size*2 ,seq_feature1.size(2),-1)
            else:
                seq_feature = encoder(data, seqs, position_id,  causality_mask = encoder_causality_mask)
        else:
            seqs = item_embed(data)
            seq_feature = encoder(data, seqs, position_id,  causality_mask = encoder_causality_mask)
        if self.pooling == "bert":
            out = seq_feature[:,0,:]
        elif self.pooling == "ave":
            if self.opt['augment_type']=="dropout":
            #     non_zero_mask = non_zero_mask.unsqueeze(1).expand(-1,2,-1).reshape(batch_size*2,-1)
            #     out = torch.sum(seq_feature, dim=1)/torch.sum(non_zero_mask, dim=1).unsqueeze(-1)
            # else:
                out = torch.sum(seq_feature, dim=1)/torch.sum(non_zero_mask, dim=1).unsqueeze(-1)
            
        out = CL_projector(out)
        return out

    def augmentation_based_CL(self, augmented_data, domain = "X"):
        # print(f"domain {domain}:",augmented_data[0][0],augmented_data[0][1])
        batch_size  = augmented_data.shape[0]
        if self.opt['augment_type'] != "dropout":
            augmented_data = augmented_data.view(batch_size*2 ,-1) #[B*2, seq_len]
        # print("augmented_data:",augmented_data.shape)
        if domain == "X":
            out = self.get_embedding_for_ssl(data = augmented_data, encoder = self.model.encoder_X, item_embed = self.model.item_emb_X, CL_projector = self.model.CL_projector_X, 
                          encoder_causality_mask = False, add_graph_node=True, graph_embedding = self.model.single_emb_X)
        elif domain == "Y":
            out = self.get_embedding_for_ssl(data = augmented_data, encoder = self.model.encoder_Y, item_embed = self.model.item_emb_Y, CL_projector = self.model.CL_projector_Y, 
                          encoder_causality_mask = False, add_graph_node=True, graph_embedding = self.model.single_emb_Y)
        elif domain == "mixed":
            out = self.get_embedding_for_ssl(data = augmented_data, encoder = self.model.encoder, item_embed = self.model.item_emb, CL_projector = self.model.CL_projector, 
                          encoder_causality_mask = False, add_graph_node=True, graph_embedding = self.model.cross_emb)
        
        out = out.view(batch_size,2,-1)
        z1, z2 = out[:, 0, :], out[:, 1, :]
        cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))
        labels = torch.arange(cos_sim.size(0)).long().to(cos_sim.device)
        cl_loss = self.CL_criterion(cos_sim, labels)
        return cl_loss
        
    def MoCo_train(self, MoCo_model, augmented_seq, cluster_result = None, index = None, task = "in-domain",ts =None):
        if self.opt['augment_type'] == "dropout":
            logits, labels, proto_logits, proto_labels,_,_ = MoCo_model(seq_q = augmented_seq, seq_k=augmented_seq, is_eval=False, cluster_result= cluster_result, index=index, task=task,ts = ts)
        else:
            logits, labels, proto_logits, proto_labels,_,_ = MoCo_model(seq_q = augmented_seq[:,0,:], seq_k=augmented_seq[:,1,:], is_eval=False, cluster_result= cluster_result, index=index, task=task, ts =ts)
        if task == "in-domain":
            loss = self.CL_criterion(logits, labels)
            # loss = 0
        else:
            loss = 0
        if proto_logits is not None:
            loss_proto = 0
            if task == "in-domain":
                for proto_out, proto_target in zip(proto_logits, proto_labels):
                    loss_proto += self.proto_criterion(proto_out, proto_target)  
            elif task == "cross-domain":
                for proto_out in proto_logits:
                    entropy = -torch.sum(proto_out*torch.log(proto_out+ 1e-9),dim=1)# avoid value equal to 0
                    mean_entropy = torch.mean(entropy)
                    loss_proto += mean_entropy
            loss_proto /= len(self.opt['num_cluster']) 
            loss += loss_proto
        return loss
    def pull_xy_embedding(self,seq, x_seq, y_seq):
        feat = self.get_embedding_for_ssl(data = seq, encoder = self.model.encoder, item_embed = self.model.item_emb, CL_projector = self.model.CL_projector, 
                          encoder_causality_mask = False, add_graph_node=False)
        # feat_X = self.get_embedding_for_ssl(data = x_seq, encoder = self.model.encoder_X, item_embed = self.model.item_emb_X, CL_projector = self.model.CL_projector_X)
        feat_Y = self.get_embedding_for_ssl(data = y_seq, encoder = self.model.encoder_Y, item_embed = self.model.item_emb_Y, CL_projector = self.model.CL_projector_Y)
        # sim_X_Y = torch.matmul(feat_X, feat_Y.T)
        # labels_X_Y = torch.arange(sim_X_Y.shape[0]).to(sim_X_Y.device).long()
        # sim_mixed_Y = torch.matmul(feat, feat_Y.T)
        # labels_mixed_X = torch.arange(sim_mixed_X.shape[0]).to(sim_mixed_X.device).long()
        sim_mixed_Y = torch.matmul(feat, feat_Y.T)
        labels_mixed_Y = torch.arange(sim_mixed_Y.shape[0]).to(sim_mixed_Y.device).long()
        # logits = torch.cat([sim_X_Y, sim_mixed_X, sim_mixed_Y], dim=0)
        # labels = torch.cat([labels_X_Y, labels_mixed_X, labels_mixed_Y], dim=0)
        logits = sim_mixed_Y
        labels = labels_mixed_Y
        loss = self.CL_criterion(logits, labels)
        return loss
    def nonoverlap_user_generation(self):
        
        # get non-overlap data
        nonoverlap_dataLoader = NonoverlapDataLoader(self.opt['data_dir'], self.opt['batch_size'],  self.opt)
        nonoverlap_seq_feat, nonoverlap_feat = compute_embedding_for_target_user(self.opt, nonoverlap_dataLoader, self.model, name = 'non-overlap')
        target_gt = [data[3] for data in nonoverlap_dataLoader.all_data]
        target_gt_mask = [data[4] for data in nonoverlap_dataLoader.all_data]        
        # get overlap data
        overlap_dataLoader = DataLoader(self.opt['data_dir'], self.opt['batch_size'], self.opt, -1)
        overlap_seq_feat, _ = compute_embedding_for_target_user(self.opt, overlap_dataLoader, self.model, name = 'overlap')
        # get topk similar user for nonoverlap female data
        sim = nonoverlap_seq_feat@overlap_seq_feat.T
        top_k_values, top_k_indice = torch.topk(sim, 20, dim=1)
        try:
            lookup_dict = {item[0]: item[1:] for item in overlap_dataLoader.all_data}
            lookup_keys = set(lookup_dict.keys())
            top_k_data = [[lookup_dict[idx] for idx in sorted(random.sample(indices, 10))] for indices in top_k_indice.tolist() if all([x in lookup_keys for x in indices])] #從20個中隨機選10個
        except:
            ipdb.set_trace()
        
        #select [required_num] augmented user to be added in each batch 
        required_num = 1000 #number of aumented user to be added in each batch
        if len(top_k_data)<required_num:
            required_num = len(top_k_data)
        selected_idx = random.sample(list(range(len(top_k_data))), required_num)
        selected_top_k_data = [top_k_data[idx] for idx in selected_idx]
        
        augmented_seq_fea = []
        augmented_share_y_ground = []
        augmented_share_y_ground_mask = []
        for query, topk in zip(nonoverlap_seq_feat, selected_top_k_data):
            mixed_seq = [seq[0] for seq in topk] # 1,4,12
            target_seq = [seq[2] for seq in topk]
            position = [seq[3] for seq in topk] 
            share_y_grounds = [seq[11] for seq in topk] 
            # share_y_ground_masks = [seq[16] for seq in topk] 
            value = torch.tensor(mixed_seq)
            key = torch.tensor(target_seq)
            query = query.unsqueeze(0)
            
            if self.opt['cuda']:
                query, key, value = query.cuda(), key.cuda(), value.cuda()
            ts_key = [seq[9] for seq in topk] if self.opt['time_encode'] else None
            ts_value = [seq[7] for seq in topk] if self.opt['time_encode'] else None

            key, _ = get_sequence_embedding(self.opt,key,self.model.encoder_Y,self.model.item_emb_Y, encoder_causality_mask = False, ts = ts_key) #[10, 128] (sequence_embedding, item embedding)
            _, value = get_sequence_embedding(self.opt,value,self.model.encoder,self.model.item_emb, encoder_causality_mask = True, ts = ts_value) #[10, 50, 128]
            weight = torch.softmax(torch.matmul(query, key.T)/math.sqrt(query.size(-1)), dim=-1) #[1, 10]
            aug_mixed_seq = torch.einsum('ij,jkl->ikl', weight, value).squeeze()#[1, 50, 128]
            max_position_id = max([max(s) for s in position])
            aug_position = [0]*(50-max_position_id) + list(range(0,max_position_id+1))[1:]
            
            share_y_grounds = torch.tensor(share_y_grounds)
            indices = torch.randint(share_y_grounds.size(0), (share_y_grounds.size(-1),)) 
            aug_share_y_ground = share_y_grounds[indices, torch.arange(share_y_grounds.size(-1))]    
            aug_share_y_ground_mask =  (aug_share_y_ground != self.opt['target_item_num']).to(torch.int)                          
            augmented_share_y_ground.append(aug_share_y_ground)
            augmented_share_y_ground_mask.append(copy.deepcopy(aug_share_y_ground_mask))
            augmented_seq_fea.append(aug_mixed_seq.clone())
        # nonoverlap_feat = nonoverlap_feat.repeat_interleave(target_num,dim=0)
        # target_gt = torch.tensor(target_gt).repeat_interleave(target_num,dim=0)
        # target_gt_mask = torch.tensor(target_gt_mask).repeat_interleave(target_num,dim=0)
        
        def add_noise(representation): # do perturbation
            noise = torch.randn_like(representation)
            scaled_noise = torch.nn.functional.normalize(noise,dim=2)
            return representation + scaled_noise
        
        nonoverlap_feat = nonoverlap_feat[selected_idx]
        nonoverlap_feat = add_noise(nonoverlap_feat)
        target_gt = torch.tensor(target_gt)[selected_idx]
        target_gt_mask = torch.tensor(target_gt_mask)[selected_idx]
        augmented_seq_fea = add_noise(torch.stack(augmented_seq_fea))        
        augmented_share_y_ground = torch.stack(augmented_share_y_ground)
        augmented_share_y_ground_mask = torch.stack(augmented_share_y_ground_mask)
        if self.opt['cuda']:
            augmented_seq_fea = augmented_seq_fea.cuda()
            augmented_share_y_ground = augmented_share_y_ground.cuda()
            augmented_share_y_ground_mask = augmented_share_y_ground_mask.cuda()
            nonoverlap_feat = nonoverlap_feat.cuda()
            target_gt = target_gt.cuda()
            target_gt_mask = target_gt_mask.cuda()
        
        return augmented_seq_fea, augmented_share_y_ground, augmented_share_y_ground_mask, nonoverlap_feat, target_gt - self.opt['source_item_num'], target_gt_mask
    def mask_correlated_samples(self, batch_size):
        N =batch_size*2
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for  i in range(batch_size):
            mask[i,i+batch_size] = 0
            mask[i+batch_size,i] = 0
        return mask
    def group_CL(self, seq, seq_xd, seq_yd):
        seq = seq.reshape(-1,seq.size(-1))
        seq_xd = seq_xd.reshape(-1,seq_xd.size(-1))
        seq_yd = seq_yd.reshape(-1,seq_yd.size(-1))
        # ipdb.set_trace()
        out = get_embedding_for_ssl(self.opt, seq, self.model.encoder, self.model.item_emb, projector = None)
        out_yd = get_embedding_for_ssl(self.opt, seq_yd, self.model.encoder_Y, self.model.item_emb_Y, projector = None)
        out = self.model.CL_projector(out+out_yd)
        # out = out+out_yd
        out = out.reshape(-1,2,out.size(-1))
        z1,z2 = out[:,0,:], out[:,1,:]
        batch_size = z1.size(0)
        N = batch_size*2
        z = torch.cat([z1,z2],dim=0)
        sim = torch.mm(z, z.T)/self.opt['temp']
        pos = torch.cat([torch.diag(sim, z1.size(0)), torch.diag(sim, -z1.size(0))])
        
        if batch_size != self.opt['batch_size']:
            mask = self.mask_correlated_samples(batch_size)
        else:
            mask = self.mask_correlated_samples(self.opt['batch_size'])
        neg = sim[mask].reshape(batch_size*2,-1)
        labels = torch.zeros(N).to(pos.device).long()
        logits = torch.cat([pos.unsqueeze(1), neg], dim=1)
        loss = self.CL_criterion(logits, labels)
        return loss
        # seq_len = seq[seq!=self.opt["source_item_num"] + self.opt["target_item_num"]].sum(dim=1)
        # mask = seq!=0
        # cumulative_sum = mask.cumsum(dim=1)
        # first_non_zero_mask = (cumulative_sum == 1) & mask
        # idx = first_non_zero_mask.nonzero()
        # pred, weight = self.gender_discriminator(out)
        # pred, weight = pred.squeeze(), weight.squeeze()
        # pred_female_seq = seq[torch.round(pred)==0]
        # sorted_idx = weight.argsort(dim=1)[:,-10:] #ascending order
        # pred_female_seq[torch.arange(len(pred_female_seq))[:, None], sorted_idx]
    def I2C_CL(self, index, mixed_seq, target_seq, ts_d, ts_yd, gender):
        ts_d = ts_d if self.opt['time_encode'] else None
        ts_yd = ts_yd if self.opt['time_encode'] else None
        # mixed_feature = get_embedding_for_ssl(self.opt , mixed_seq, self.model.encoder, self.model.item_emb,ts=ts_d, projector = self.model.interest_projector)
        # target_feature = get_embedding_for_ssl(self.opt , target_seq, self.model.encoder_Y, self.model.item_emb_Y,ts=ts_yd,projector=self.model.interest_projector_Y)
        # new_cluster, multi_interest = self.cluster(mixed_feature)
        # pos = torch.sum(multi_interest*target_feature,dim=1,keepdim=True)
        # neg = target_feature@new_cluster.T
        # logits = torch.cat([pos,neg],dim=1)

        gender = gender[:,0]
        target_male_seq = target_seq[gender==1]
        target_female_seq = target_seq[gender==0]
        mixed_male_seq = mixed_seq[gender==1]
        mixed_female_seq = mixed_seq[gender==0]
        ts_d = ts_d if self.opt['time_encode'] else None
        ts_yd = ts_yd if self.opt['time_encode'] else None
        mixed_male_feature = get_embedding_for_ssl(self.opt , mixed_male_seq, self.model.encoder, self.model.item_emb,ts=ts_d, projector = self.model.interest_projector)
        mixed_female_feature = get_embedding_for_ssl(self.opt , mixed_female_seq, self.model.encoder, self.model.item_emb,ts=ts_d, projector = self.model.interest_projector)
        target_male_feature = get_embedding_for_ssl(self.opt , target_male_seq, self.model.encoder_Y, self.model.item_emb_Y,ts=ts_yd,projector=self.model.interest_projector_Y)
        target_female_feature = get_embedding_for_ssl(self.opt , target_female_seq, self.model.encoder_Y, self.model.item_emb_Y,ts=ts_yd,projector=self.model.interest_projector_Y)
        new_male_cluster, male_multi_interest = self.male_cluster(mixed_male_feature)
        new_female_cluster, female_multi_interest = self.female_cluster(mixed_female_feature)
        new_male_cluster =torch.nn.functional.normalize(new_male_cluster, dim=1)
        male_multi_interest = torch.nn.functional.normalize(male_multi_interest, dim=1)
        new_female_cluster =torch.nn.functional.normalize(new_female_cluster, dim=1)
        female_multi_interest = torch.nn.functional.normalize(female_multi_interest, dim=1)
        male_pos = torch.sum(male_multi_interest*target_male_feature,dim=1,keepdim=True)
        female_pos = torch.sum(female_multi_interest*target_female_feature, dim=1,keepdim=True)
        male_neg = target_male_feature@new_male_cluster.T
        female_neg  = target_female_feature@new_female_cluster.T
        male_logits = torch.cat([male_pos, male_neg],dim=1)
        female_logits = torch.cat([female_pos, female_neg],dim=1)
        logits = torch.cat([male_logits,female_logits],dim=0)
        
        logits = logits/self.opt['temp']
        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        if self.opt['cuda']:
            labels = labels.cuda() 
        loss = self.CL_criterion(logits, labels)
        return loss
    def train_batch(self, epoch, batch, i, cluster_result):
        self.model.train()
        self.optimizer.zero_grad()
        self.d_optimizer.zero_grad()
        self.model.graph_convolution()
        cluster_result_X, cluster_result_Y, cluster_result_cross = cluster_result[0], cluster_result[1], cluster_result[2]
        index, seq, x_seq, y_seq, position, x_position, y_position, ts_d, ts_xd, ts_yd, ground, share_x_ground, share_y_ground, x_ground, y_ground, ground_mask, share_x_ground_mask, share_y_ground_mask, x_ground_mask, y_ground_mask, corru_x, corru_y,masked_xd, neg_xd, masked_yd, neg_yd, augmented_d, augmented_xd,augmented_yd,gender = self.unpack_batch(batch)
        if self.opt['training_mode'] =="joint_learn":
            # time_CL
            if self.opt["ssl"]=="time_CL":
                ssl_loss_xd, ssl_loss_yd = self.time_CL(i_xd, near_xd , i_yd, near_yd)
                time_CL_loss = ssl_loss_xd + ssl_loss_yd
            # augmentation_based_CL
            if self.opt["ssl"]=="augmentation_based_CL":
                augmentation_based_CL_loss_xd = self.augmentation_based_CL(augmented_xd, domain = "X")
                augmentation_based_CL_loss_yd = self.augmentation_based_CL(augmented_yd, domain = "Y")
                # augmentation_based_CL_loss_d = self.augmentation_based_CL(augmented_d, domain = "mixed")
                augmentation_based_CL_loss =  augmentation_based_CL_loss_xd + augmentation_based_CL_loss_yd 
            
            #MoCo : proto_CL
            if self.opt["ssl"]=="proto_CL":
                # MoCo_loss_xd = self.MoCo_train(self.MoCo_X, augmented_xd, cluster_result = cluster_result_X, index=index, task = "in-domain", ts = ts_xd)
                # MoCo_loss_yd = self.MoCo_train(self.MoCo_Y, augmented_yd, cluster_result = cluster_result_Y, index=index, task = "in-domain", ts = ts_yd)
                MoCo_loss_mixed = self.MoCo_train(epoch,self.MoCo_mixed, seq, cluster_result = None, index=index, task = "in-domain")

                if cluster_result_cross is not None:
                    MoCo_loss_xd_cross = self.MoCo_train(self.MoCo_X, augmented_xd, cluster_result = cluster_result_cross, index=None, task = "cross-domain")
                    MoCo_loss_yd_cross = self.MoCo_train(self.MoCo_Y, augmented_yd, cluster_result = cluster_result_cross, index=None, task = "cross-domain")
                    proto_CL_loss = MoCo_loss_xd_cross + MoCo_loss_yd_cross
                else:
                    # proto_CL_loss =  MoCo_loss_xd + MoCo_loss_yd
                    proto_CL_loss = torch.Tensor([0]).cuda()
            if self.opt['ssl'] == "triple_pull":
                pull_loss = self.pull_xy_embedding(seq, x_seq, y_seq)
            if self.opt['ssl'] == "NNCL":
                NNCL_loss = self.NNCL(x_seq, y_seq)
            if self.opt['ssl'] in ['interest_cluster','both']:
                if epoch+1>=self.opt['warmup_epoch']:
                    I2C_loss = self.I2C_CL(index,seq, y_seq,ts_d,ts_yd,gender)
                    # print(f"\033[34mI2C_loss:{I2C_loss}\033[0m")
                else:
                    I2C_loss = torch.Tensor([0]).cuda()
            if self.opt['ssl'] in ["both","group_CL"]:
                group_CL_loss = self.group_CL(augmented_d,augmented_xd,augmented_yd)
        if self.opt['domain'] =="single":
            seq = None 
            if self.opt['main_task'] == "X":
                y_seq = None
            elif self.opt['main_task'] == "Y":
                x_seq = None  
                
        if self.opt['time_encode']:
            seqs_fea, x_seqs_fea, y_seqs_fea = self.model(seq, x_seq, y_seq, position, x_position, y_position, ts_d, ts_xd, ts_yd)
        else: 
            seqs_fea, x_seqs_fea, y_seqs_fea = self.model(seq, x_seq, y_seq, position, x_position, y_position)
        
       
        #取倒數10個
        used = 50
        ground = ground[:,-used:]
        ground_mask = ground_mask[:, -used:]
        share_x_ground = share_x_ground[:, -used:]
        share_x_ground_mask = share_x_ground_mask[:, -used:]
        share_y_ground = share_y_ground[:, -used:]
        share_y_ground_mask = share_y_ground_mask[:, -used:]
        x_ground = x_ground[:, -used:]
        x_ground_mask = x_ground_mask[:, -used:]
        y_ground = y_ground[:, -used:]
        y_ground_mask = y_ground_mask[:, -used:]
        
        aug_data = None
        if epoch > 10 and i==0 and self.opt['data_augmentation']=="user_generation":
            aug_data = self.nonoverlap_user_generation()
            # aug_data = None
        if aug_data is not None:
            augmented_seqs_fea, augmented_share_y_ground, augmented_share_y_ground_mask, target_feat, target_gt, target_gt_mask = aug_data
            print(f"\033[34m{len(augmented_seqs_fea)} Non-overlap user generation\033[0m")
            seqs_fea = torch.cat([seqs_fea,augmented_seqs_fea])
            share_y_ground = torch.cat([share_y_ground,augmented_share_y_ground[:, -used:]],dim=0)
            share_y_ground_mask = torch.cat([share_y_ground_mask,augmented_share_y_ground_mask[:, -used:]],dim=0)
            y_seqs_fea = torch.cat([y_seqs_fea, target_feat])
            y_ground = torch.cat([y_ground, target_gt[:, -used:]],dim=0)
            y_ground_mask = torch.cat([y_ground_mask, target_gt_mask[:, -used:]],dim=0)
        if self.opt['main_task'] == "dual":
            share_x_result =  self.model.lin_X(seqs_fea[:,-used:]) # b * seq * X_num
            share_y_result = self.model.lin_Y(seqs_fea[:, -used:])  # b * seq * Y_num
            share_pad_result = self.model.lin_PAD(seqs_fea[:, -used:])  # b * seq * 1 #最後一維，即padding，score要是零
            share_trans_x_result = torch.cat((share_x_result, share_pad_result), dim=-1)
            share_trans_y_result = torch.cat((share_y_result, share_pad_result), dim=-1)

            specific_x_result = self.model.lin_X(seqs_fea[:,-used:] + x_seqs_fea[:, -used:])  # b * seq * X_num
            # specific_x_result = self.model.lin_X(x_seqs_fea[:, -used:])
            specific_x_pad_result = self.model.lin_PAD(x_seqs_fea[:, -used:])  # b * seq * 1
            specific_x_result = torch.cat((specific_x_result, specific_x_pad_result), dim=-1)
            
            specific_y_result = self.model.lin_Y(seqs_fea[:,-used:] + y_seqs_fea[:, -used:])  # b * seq * Y_num
            # specific_y_result = self.model.lin_Y(y_seqs_fea[:, -used:])
            specific_y_pad_result = self.model.lin_PAD(y_seqs_fea[:, -used:])  # b * seq * 1
            specific_y_result = torch.cat((specific_y_result, specific_y_pad_result), dim=-1)

            x_share_loss = self.CS_criterion(
                share_trans_x_result.reshape(-1, self.opt["source_item_num"] + 1),
                share_x_ground.reshape(-1))  # b * seq
            y_share_loss = self.CS_criterion(
                share_trans_y_result.reshape(-1, self.opt["target_item_num"] + 1),
                share_y_ground.reshape(-1))  # b * seq
            x_loss = self.CS_criterion(
                specific_x_result.reshape(-1, self.opt["source_item_num"] + 1),
                x_ground.reshape(-1))  # b * seq
            y_loss = self.CS_criterion(
                specific_y_result.reshape(-1, self.opt["target_item_num"] + 1),
                y_ground.reshape(-1))  # b * seq

            x_share_loss = (x_share_loss * (share_x_ground_mask.reshape(-1))).mean() #只取預測x的部分
            y_share_loss = (y_share_loss * (share_y_ground_mask.reshape(-1))).mean() #只取預測y的部分
            x_loss = (x_loss * (x_ground_mask.reshape(-1))).mean()
            y_loss = (y_loss * (y_ground_mask.reshape(-1))).mean()
            
            if self.opt['training_mode'] =="joint_learn":
                if self.opt["ssl"]=="time_CL":
                    loss = x_share_loss + y_share_loss + x_loss + y_loss + time_CL_loss
                    self.time_CL_loss += time_CL_loss.item()
                elif self.opt["ssl"]=="augmentation_based_CL":
                    loss = x_share_loss + y_share_loss + x_loss + y_loss + augmentation_based_CL_loss
                    self.augmentation_based_CL_loss += augmentation_based_CL_loss.item()
                elif self.opt["ssl"]=="proto_CL":
                    loss = x_share_loss + y_share_loss + x_loss + y_loss + proto_CL_loss
                    if torch.is_tensor(proto_CL_loss):
                        self.proto_CL_loss += proto_CL_loss.item()
                elif self.opt['ssl'] == "triple_pull":
                    loss = x_share_loss + y_share_loss + x_loss + y_loss + pull_loss
                    self.pull_loss += pull_loss.item()
                elif self.opt['ssl'] == "NNCL":
                    loss = x_share_loss + y_share_loss + x_loss + y_loss + NNCL_loss*0.1
                    self.NNCL_loss += NNCL_loss.item()
                elif self.opt['ssl'] in ['interest_cluster']:
                    loss = x_share_loss + y_share_loss + x_loss + y_loss + I2C_loss
                    self.I2C_loss += I2C_loss.item()
                elif self.opt['ssl'] in ['group_CL']:
                    loss = x_share_loss + y_share_loss + x_loss + y_loss + group_CL_loss
                    self.group_CL_loss += group_CL_loss.item()
                elif self.opt['ssl'] in ['both']:
                    loss = x_share_loss + y_share_loss + x_loss + y_loss + group_CL_loss*self.opt['lambda_'][0] + I2C_loss*self.opt['lambda_'][1] 
                    self.I2C_loss += I2C_loss.item()
                    self.group_CL_loss += group_CL_loss.item()
            else:
                loss = x_share_loss + y_share_loss + x_loss + y_loss
                # loss = x_share_loss + x_loss + y_loss
                # loss =  x_loss + y_loss
            # loss = self.opt["lambda"]*(x_share_loss + y_share_loss + x_loss + y_loss) + (1 - self.opt["lambda"]) * (x_mi_loss + y_mi_loss)
            
            self.prediction_loss += (x_share_loss.item() + y_share_loss.item() + x_loss.item() + y_loss.item())
            # self.prediction_loss += (x_share_loss.item() + x_loss.item() + y_loss.item())
            # self.prediction_loss +=  (x_loss.item() + y_loss.item())
           
        elif self.opt['main_task'] == "X":
            if seqs_fea is not None:
                share_x_result =  self.model.lin_X(seqs_fea[:,-used:]) # b * seq * X_num
                share_pad_result = self.model.lin_PAD(seqs_fea[:, -used:])  # b * seq * 1 #最後一維，即padding，score要是零
                share_trans_x_result = torch.cat((share_x_result, share_pad_result), dim=-1)
                
                specific_x_result = self.model.lin_X(seqs_fea[:,-used:] + x_seqs_fea[:, -used:])  # b * seq * X_num
                # specific_x_result = self.model.lin_X(x_seqs_fea[:, -used:])
                specific_x_pad_result = self.model.lin_PAD(x_seqs_fea[:, -used:])  # b * seq * 1
                specific_x_result = torch.cat((specific_x_result, specific_x_pad_result), dim=-1)
                x_share_loss = self.CS_criterion(
                share_trans_x_result.reshape(-1, self.opt["source_item_num"] + 1),
                share_x_ground.reshape(-1))  # b * seq
                x_share_loss = (x_share_loss * (share_x_ground_mask.reshape(-1))).mean() #只取預測x的部分
                # ipdb.set_trace()
            else:
                # print("seqs_fea is None")
                specific_x_result = self.model.lin_X(x_seqs_fea[:, -used:])
                specific_x_pad_result = self.model.lin_PAD(x_seqs_fea[:, -used:])  # b * seq * 1
                specific_x_result = torch.cat((specific_x_result, specific_x_pad_result), dim=-1)
            
            x_loss = self.CS_criterion(
                specific_x_result.reshape(-1, self.opt["source_item_num"] + 1),
                x_ground.reshape(-1))  # b * seq
            x_loss = (x_loss * (x_ground_mask.reshape(-1))).mean()
            
            if self.opt['training_mode'] =="joint_learn":
                if self.opt["ssl"]=="time_CL":
                    loss = x_share_loss + x_loss  + time_CL_loss
                    self.time_CL_loss += time_CL_loss.item()
                elif self.opt["ssl"]=="augmentation_based_CL":
                    loss = x_share_loss +  x_loss + augmentation_based_CL_loss
                    self.augmentation_based_CL_loss += augmentation_based_CL_loss.item()
                elif self.opt["ssl"]=="proto_CL":
                    loss = x_share_loss + x_loss + proto_CL_loss
                    if torch.is_tensor(proto_CL_loss):
                        self.proto_CL_loss += proto_CL_loss.item()
                elif self.opt['ssl'] == "triple_pull":
                    loss = x_share_loss + x_loss + pull_loss
                    self.pull_loss += pull_loss.item()
                elif self.opt['ssl'] == "NNCL":
                    loss = x_share_loss + x_loss + NNCL_loss
                    self.NNCL_loss += NNCL_loss.item()
                elif self.opt['ssl'] in ['interest_cluster']:
                    loss = y_share_loss + y_loss + I2C_loss
                    self.I2C_loss += I2C_loss.item()
                elif self.opt['ssl'] in ['group_CL']:
                    loss = x_share_loss + x_loss + group_CL_loss
                    self.group_CL_loss += group_CL_loss.item()
                elif self.opt['ssl'] in ['both']:
                    loss = x_share_loss + x_loss + I2C_loss*self.opt['lambda_'][1] + group_CL_loss*self.opt['lambda_'][0]
                    self.I2C_loss += I2C_loss.item()
                    self.group_CL_loss += group_CL_loss.item()
            else:
                if seqs_fea is not None:
                    loss = x_share_loss + x_loss
                    self.prediction_loss += (x_share_loss.item() + x_loss.item())
                else:
                    loss = x_loss
                    self.prediction_loss += x_loss.item()
           
        elif self.opt['main_task'] == "Y":
            if seqs_fea is not None:                                  
                share_y_result =  self.model.lin_Y(seqs_fea[:,-used:]) # b * seq * X_num
                share_pad_result = self.model.lin_PAD(seqs_fea[:, -used:])  # b * seq * 1 #最後一維，即padding，score要是零
                share_trans_y_result = torch.cat((share_y_result, share_pad_result), dim=-1)

                specific_y_result = self.model.lin_Y(seqs_fea[:,-used:] + y_seqs_fea[:, -used:])  # b * seq * X_num
                # specific_y_result = self.model.lin_Y(y_seqs_fea[:, -used:])
                specific_y_pad_result = self.model.lin_PAD(y_seqs_fea[:, -used:])  # b * seq * 1
                specific_y_result = torch.cat((specific_y_result, specific_y_pad_result), dim=-1)
                y_share_loss = self.CS_criterion(
                share_trans_y_result.reshape(-1, self.opt["target_item_num"] + 1),share_y_ground.reshape(-1))
                y_share_loss = (y_share_loss * (share_y_ground_mask.reshape(-1))).mean() # 只取預測y的部分
            else:
                specific_y_result = self.model.lin_Y(y_seqs_fea[:, -used:])
                specific_y_pad_result = self.model.lin_PAD(y_seqs_fea[:, -used:])
                specific_y_result = torch.cat((specific_y_result, specific_y_pad_result), dim=-1)
            # ipdb.set_trace()
            # weight = []    
            # for g in gender:
            #     g = g[0]
            #     if g==0:
            #         weight.append(5)
            #     else:
            #         weight.append(1)
            # weight = torch.tensor(weight).cuda()
            # weight = weight.float().unsqueeze(-1).expand(256,10).reshape(-1)
            y_loss = self.CS_criterion(
                specific_y_result.reshape(-1, self.opt["target_item_num"] + 1),
                y_ground.reshape(-1))  # b * seq
            # weighted_y_loss = y_loss*weight
            y_loss = (y_loss * (y_ground_mask.reshape(-1))).mean()
            
            if self.opt['training_mode'] =="joint_learn":
                if self.opt["ssl"]=="time_CL":
                    loss = y_share_loss + y_loss  + time_CL_loss
                    self.time_CL_loss += time_CL_loss.item()
                elif self.opt["ssl"] == "augmentation_based_CL":
                    loss = y_share_loss +  y_loss + augmentation_based_CL_loss
                    self.augmentation_based_CL_loss += augmentation_based_CL_loss.item()
                elif self.opt["ssl"]=="proto_CL":
                    loss = y_share_loss + y_loss + proto_CL_loss
                    if torch.is_tensor(proto_CL_loss):
                        self.proto_CL_loss += proto_CL_loss.item()
                elif self.opt['ssl'] == "triple_pull":
                    loss = y_share_loss + y_loss + pull_loss
                    self.pull_loss += pull_loss.item()
                elif self.opt['ssl'] == "NNCL":
                    loss = y_share_loss + y_loss + NNCL_loss
                    self.NNCL_loss += NNCL_loss.item()
                elif self.opt['ssl'] in ['interest_cluster']:
                    loss = y_share_loss + y_loss + I2C_loss
                    self.I2C_loss += I2C_loss.item()
                elif self.opt['ssl'] in ['group_CL']:
                    loss = y_share_loss + y_loss + group_CL_loss
                    self.group_CL_loss += group_CL_loss.item()
                elif self.opt['ssl'] in ['both']:
                    loss = y_share_loss + y_loss + group_CL_loss*self.opt['lambda_'][0] + I2C_loss*self.opt['lambda_'][1] 
                    self.I2C_loss += I2C_loss.item()
                    self.group_CL_loss += group_CL_loss.item()
            else:
                if seqs_fea is not None:
                    loss = y_share_loss + y_loss
                    self.prediction_loss += (y_share_loss.item() + y_loss.item())
                else:
                    loss = y_loss
                    self.prediction_loss += y_loss.item()
                    
        ### adversarial training for bias free embedding ###
        dis_loss, adv_loss = None, None
        if self.gender_discriminator_pretrained:
            ### update generator ###
            target = gender[:,0].float()
            # fake_target = (target.clone()==0).to(int).float()
            seqs_fea,_ = get_sequence_embedding(self.opt,seq,self.model.encoder,self.model.item_emb, encoder_causality_mask = True)
            if self.opt['main_task'] == "X":
                specific_seq_fea,_ = get_sequence_embedding(self.opt,x_seq,self.model.encoder_X,self.model.item_emb_X, encoder_causality_mask = True)
            elif self.opt['main_task'] == "Y":
                specific_seq_fea,_ = get_sequence_embedding(self.opt,y_seq,self.model.encoder_Y,self.model.item_emb_Y, encoder_causality_mask = True)
            pred = self.gender_discriminator(seqs_fea + specific_seq_fea).squeeze()
            adv_loss = self.BCE_criterion(pred, target)
            # adv_loss.backward()
            loss = loss - adv_loss
            # loss = loss - adv_loss*2
            
            ###update discriminator###
            pred = self.gender_discriminator(seqs_fea.detach() + specific_seq_fea.detach()).squeeze()
            dis_loss = self.BCE_criterion(pred, target)
            dis_loss.backward()
            self.d_optimizer.step()
        loss.backward()
        self.optimizer.step()
       
        # for j,(name, param) in enumerate(self.male_cluster.named_parameters()):
        #     if param.requires_grad and i==0 and j==0:
        #         print(name, param.data)
        return loss.item(), adv_loss.item() if adv_loss else None, dis_loss.item() if dis_loss else None
    def pretrain_discriminator(self,dataloader):
        self.gender_discriminator.train()
        for epoch in range(30):
            batch_dis_loss = 0
            batch_dis_acc = 0
            batch_predict_male_num = 0
            batch_gt_male_num = 0
            # dataloader = DataLoader(self.opt['data_dir'], self.opt['batch_size'], self.opt, evaluation = -1, collate_fn  = None, generator = None, model = self.model)
            for i,batch in enumerate(dataloader):  
                self.d_optimizer.zero_grad()
                index, seq, x_seq, y_seq, position, x_position, y_position, ts_d, ts_xd, ts_yd, ground, share_x_ground, share_y_ground, x_ground, y_ground, ground_mask, share_x_ground_mask, share_y_ground_mask, x_ground_mask, y_ground_mask, corru_x, corru_y,masked_xd, neg_xd, masked_yd, neg_yd,augmented_d, augmented_xd,augmented_yd,gender= self.unpack_batch(batch)
                # seqs_fea,_ = get_sequence_embedding(self.opt,seq,self.model.encoder,self.model.item_emb, encoder_causality_mask = True)
                # y_seqs_fea,_ = get_sequence_embedding(self.opt,y_seq,self.model.encoder_Y,self.model.item_emb_Y, encoder_causality_mask = True)
                # pred = self.gender_discriminator(seqs_fea + y_seqs_fea).squeeze()
                seqs_fea = get_item_embedding_for_sequence(self.opt,seq,self.model.encoder,self.model.item_emb, self.model.CL_projector,encoder_causality_mask = False,cl =False)
                # y_seqs_fea = get_item_embedding_for_sequence(self.opt,y_seq,self.model.encoder_Y,self.model.item_emb_Y, encoder_causality_mask = False,cl =False)
                pred, _ = self.gender_discriminator(seqs_fea)
                pred = pred.squeeze()
                weight = []    
                for g in gender:
                    g = g[0]
                    if g==0:
                        weight.append(3)
                    else:
                        weight.append(1)
                weight = torch.tensor(weight).cuda() if self.opt['cuda'] else torch.tensor(weight)
                weight = weight.float().unsqueeze(-1)
                dis_loss = self.BCE_criterion(pred,gender[:,0].float())
                dis_loss = (dis_loss*weight).sum()/weight.sum()
                
                dis_loss.backward()
                self.d_optimizer.step()
                acc = (torch.round(pred)== gender[:,0]).sum()/len(pred)
                batch_dis_loss += dis_loss.item()
                batch_dis_acc += acc.item()
                batch_predict_male_num += (torch.round(pred)==1).sum().item()
                batch_gt_male_num += gender[:,0].sum().item()
            print("-"*20)
            print(f"Epoch {epoch+1} discriminator pretrain loss:",batch_dis_loss/len(dataloader))
            print(f"Epoch {epoch+1} discriminator accuracy:",batch_dis_acc/len(dataloader))
            print(f"Number of real male:",batch_gt_male_num/len(dataloader))
            print(f"Number of predicted male:",batch_predict_male_num/len(dataloader))
            print("-"*20)
    def train(self, epoch, train_dataloader, valid_dataloader, mixed_test_dataloader, file_logger):
        global_step = 0
        
        current_lr = self.opt["lr"]
        format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/epoch), lr: {:.6f}'
        num_batch = len(train_dataloader)
        max_steps =  self.opt['num_epoch'] * num_batch

        print("Start training:")
        
        begin_time = time.time()
        X_dev_score_history=[0]
        Y_dev_score_history=[0]
        best_X_test = None
        best_Y_test = None
        
        patience =  self.opt["finetune_patience"]
        train_pred_loss = []
        train_in_domain_CL_loss = []
        val_pred_loss_X = []
        val_pred_loss_Y = []
        if self.opt['ssl'] in ['group_CL','both'] and self.opt['training_mode'] == "joint_learn" and self.opt['substitute_mode']=="attention_weight":
            self.pretrain_discriminator(train_dataloader)
            collator = CLDataCollator(self.opt,-1, self.generator[2],self.gender_discriminator,self.model)
            train_dataloader = DataLoader(self.opt['data_dir'], self.opt['batch_size'], self.opt, evaluation = -1, collate_fn  = collator, generator = None)
        for epoch in range(1, self.opt['num_epoch'] + 1):
            train_loss = 0
            epoch_start_time = time.time()
            self.mi_loss = 0
            self.time_CL_loss = 0
            self.augmentation_based_CL_loss = 0
            self.proto_CL_loss = 0
            self.prediction_loss = 0
            self.pull_loss = 0
            self.NNCL_loss = 0
            self.adv_loss = 0
            self.dis_loss = 0
            self.group_CL_loss = 0
            self.I2C_loss = 0
            cluster_result_X = None
            cluster_result_Y = None
            cluster_result_cross = None
            ### adversarial  learning : discriminator pretrain ###
            # if epoch ==15:
            #     self.pretrain_discriminator(train_dataloader)
                # self.gender_discriminator_pretrained = True
            
            # if self.opt['ssl'] == "proto_CL" and self.opt['training_mode'] == "joint_learn":
                # if epoch+1>=self.opt['warmup_epoch']:
                #     # x_domain
                #     # features_X = compute_features(self.opt, train_dataloader, self.MoCo_X, domain = 'X')
                #     # features_X[torch.norm(features_X,dim=1)>1.5] /= 2 #account for the few samples that are computed twice  
                #     # features_X = features_X.numpy()
                #     # cluster_result_X = run_kmeans(features_X, self.opt) 
                #     # # y_domain
                #     # features_Y = compute_features(self.opt, train_dataloader, self.MoCo_Y, domain = 'Y')
                #     # features_Y[torch.norm(features_Y,dim=1)>1.5] /= 2 
                #     # features_Y = features_Y.numpy()
                #     # cluster_result_Y = run_kmeans(features_Y, self.opt)
                #     # mixed domain
                #     if self.opt['mixed_included']:
                #         features_cross = compute_features(self.opt, train_dataloader, self.model, domain = 'mixed')
                #         features_cross[torch.norm(features_cross,dim=1)>1.5] /= 2 
                #         features_cross = features_cross.numpy()
                #         cluster_result_cross = run_kmeans(features_cross, self.opt)
            
            ### item-generation & user augmentation ###
            if self.opt['data_augmentation']=="item_augmentation":
                collator = CLDataCollator(self.opt, eval=-1, mixed_generator = self.generator[2]) if self.opt['ssl'] in ['group_CL','both'] else None
                train_dataloader = DataLoader(self.opt['data_dir'], self.opt['batch_size'], self.opt, evaluation = -1,collate_fn=collator, generator = self.generator)
            
            for i,batch in enumerate(train_dataloader):
                # print("batch:",i)
                global_step += 1
                loss, adv_loss, dis_loss = self.train_batch(epoch, batch, i, cluster_result = (cluster_result_X, cluster_result_Y, cluster_result_cross))
                self.dis_loss += dis_loss if dis_loss else 0
                self.adv_loss += adv_loss if adv_loss else 0
                train_loss+=loss
            duration = time.time() - epoch_start_time
            print(format_str.format(datetime.now(), global_step, max_steps, epoch, \
                                            self.opt['num_epoch'], train_loss/num_batch, duration, current_lr))
            print("mi:", self.mi_loss/num_batch)
            print("Adversarial loss:", self.adv_loss/num_batch)
            print("discrimination loss:", self.dis_loss/num_batch)
            print("time_CL_loss:", self.time_CL_loss/num_batch)
            print("augmentation_based_CL_loss:", self.augmentation_based_CL_loss/num_batch)
            print("proto_CL_loss:", self.proto_CL_loss/num_batch)
            print("pull_loss:", self.pull_loss/num_batch)
            print("NNCL_loss:", self.NNCL_loss/num_batch)
            print("I2C_loss:", self.I2C_loss/num_batch)
            print("group_CL_loss:", self.group_CL_loss/num_batch)
            train_in_domain_CL_loss.append(self.proto_CL_loss/num_batch)
            train_pred_loss.append(self.prediction_loss/num_batch)
            
            if epoch % 5:
                val_X_pred, val_Y_pred,val_loss_X, val_loss_Y = self.get_evaluation_result(valid_dataloader,mode = "valid")
                val_pred_loss_X.append(val_loss_X)
                val_pred_loss_Y.append(val_loss_Y)
                continue
            # eval model
            print("Evaluating on dev set...")

            self.model.eval()
            self.model.graph_convolution()
            val_X_pred, val_Y_pred,val_loss_X, val_loss_Y = self.get_evaluation_result(valid_dataloader, mode = "valid")
            val_pred_loss_X.append(val_loss_X)
            val_pred_loss_Y.append(val_loss_Y)
            val_X_MRR, val_X_NDCG_5, val_X_NDCG_10, val_X_HR_1, val_X_HR_5, val_X_HR_10 = self.cal_test_score(val_X_pred)
            val_Y_MRR, val_Y_NDCG_5, val_Y_NDCG_10, val_Y_HR_1, val_Y_HR_5, val_Y_HR_10 = self.cal_test_score(val_Y_pred)

            print("")
            print('val epoch:%d, time: %f(s), X (MRR: %.4f, NDCG@10: %.4f, HR@10: %.4f), Y (MRR: %.4f, NDCG@10: %.4f, HR@10: %.4f)'
                % (epoch, time.time() - begin_time, val_X_MRR, val_X_NDCG_10, val_X_HR_10, val_Y_MRR, val_Y_NDCG_10, val_Y_HR_10))

            if ((self.opt['main_task'] == "X" or self.opt['main_task'] == "dual") and (val_X_MRR > max(X_dev_score_history))) or ((self.opt['main_task'] == "Y" or self.opt['main_task'] == "dual")and(val_Y_MRR > max(Y_dev_score_history))):
                patience = self.opt["finetune_patience"]
                test_X_pred, test_Y_pred,test_loss_X,test_loss_Y = self.get_evaluation_result(mixed_test_dataloader, mode = "test")
                test_X_MRR, test_X_NDCG_5, test_X_NDCG_10, test_X_HR_1, test_X_HR_5, test_X_HR_10 = self.cal_test_score(test_X_pred)
                test_Y_MRR, test_Y_NDCG_5, test_Y_NDCG_10, test_Y_HR_1, test_Y_HR_5, test_Y_HR_10 = self.cal_test_score(test_Y_pred)
                # female_test_X_pred, female_test_Y_pred,female_test_loss_X,female_test_loss_Y = self.get_evaluation_result(female_test_dataloader, mode = "test")
                # female_test_X_MRR,female_test_X_NDCG_5,female_test_X_NDCG_10, female_test_X_HR_1, female_test_X_HR_5, test_X_HR_10 = self.cal_test_score(female_test_X_pred)
                # test_Y_MRR, test_Y_NDCG_5, test_Y_NDCG_10, test_Y_HR_1, test_Y_HR_5, test_Y_HR_10 = self.cal_test_score(test_Y_pred)

                # result_str = 'Epoch {}: \n'.format(epoch) 
                
                print("")
                if self.opt['main_task'] == "X" or self.opt['main_task'] == "dual":
                    if val_X_MRR > max(X_dev_score_history):
                        print("X best!")
                        print([test_X_MRR, test_X_NDCG_10, test_X_HR_10])
                        best_X_test = [test_X_MRR, test_X_NDCG_10, test_X_HR_10]
                        # result_str += "X domain:" + str([test_X_MRR, test_X_NDCG_10, test_X_HR_10])+"\n"
                        model_save_dir = f"./models/{self.opt['data_dir']}/{self.opt['id']}/{self.opt['seed']}"
                        print(f"write models into path {model_save_dir}")
                        Path(model_save_dir).mkdir(parents=True, exist_ok=True)
                        self.save(f"{model_save_dir}/X_model.pt")
                if self.opt['main_task'] == "Y" or self.opt['main_task'] == "dual":
                    if val_Y_MRR > max(Y_dev_score_history):
                        print("Y best!")
                        print([test_Y_MRR, test_Y_NDCG_10, test_Y_HR_10])
                        best_Y_test = [test_Y_MRR, test_Y_NDCG_10, test_Y_HR_10]
                        # result_str += "Y domain:" + str([test_Y_MRR, test_Y_NDCG_10, test_Y_HR_10])+"\n"
                        model_save_dir = f"./models/{self.opt['data_dir']}/{self.opt['id']}/{self.opt['seed']}"
                        print(f"write models into path {model_save_dir}")
                        Path(model_save_dir).mkdir(parents=True, exist_ok=True)
                        self.save(f"{model_save_dir}/Y_model.pt")
                # file_logger.log(result_str)
            else:
                patience -=1
                print("early stop counter:", self.opt["finetune_patience"]-patience)
                if patience == 0:
                    print("Early stop triggered!")
                    break
            X_dev_score_history.append(val_X_MRR)
            Y_dev_score_history.append(val_Y_MRR)
        final_str = str({'Best X domain':best_X_test}) + '\n' + str({'Best Y domain':best_Y_test}) + '\n'
        # file_logger.log(final_str)
        
        # write loss into file
        
        loss_save_path = f"./loss/{self.opt['data_dir']}/{self.opt['id']}/{self.opt['seed']}"
        print(f"write loss into path {loss_save_path}")
        Path(loss_save_path).mkdir(parents=True, exist_ok=True)
        np.save(f"{loss_save_path}/train_in_domain_CL_loss.npy", np.array(train_in_domain_CL_loss))
        np.save(f"{loss_save_path}/val_pred_loss_X.npy", np.array(val_pred_loss_X))
        np.save(f"{loss_save_path}/val_pred_loss_Y.npy", np.array(val_pred_loss_Y))
        np.save(f"{loss_save_path}/train_pred_loss.npy", np.array(train_pred_loss))
        
        # save X, Y domain Encoder weights for embedding plotting
        model_save_dir = f"./models/{self.opt['data_dir']}/{self.opt['id']}/{self.opt['seed']}"
        print(f"write models into path {model_save_dir} for embedding plotting")
        Path(model_save_dir).mkdir(parents=True, exist_ok=True)
        self.save(f"{model_save_dir}/model.pt")
    def cal_test_score(self, predictions):
        MRR=0.0
        HR_1 = 0.0
        HR_5 = 0.0
        HR_10 = 0.0
        NDCG_5 = 0.0
        NDCG_10 = 0.0
        valid_entity = 0.0
        for pred in predictions:
            valid_entity += 1
            MRR += 1 / pred
            if pred <= 1:
                HR_1 += 1
            if pred <= 5:
                NDCG_5 += 1 / np.log2(pred + 1)
                HR_5 += 1
            if pred <= 10:
                NDCG_10 += 1 / np.log2(pred + 1)
                HR_10 += 1
            if valid_entity % 100 == 0:
                print('.', end='')
        if valid_entity == 0:
            valid_entity = 1
        return MRR/valid_entity, NDCG_5 / valid_entity, NDCG_10 / valid_entity, HR_1 / valid_entity, HR_5 / valid_entity, HR_10 / valid_entity

    def get_evaluation_result_for_test(self, evaluation_batch, mode = "valid"):
        X_pred = []
        Y_pred = []
        X_pred_female = []
        X_pred_male = []
        Y_pred_female = []
        Y_pred_male = []
        val_loss_X = 0
        val_loss_Y = 0
        
        for i, batch in enumerate(evaluation_batch):
            X_predictions,  X_predictions_male, X_predictions_female, Y_predictions, Y_predictions_male, Y_predictions_female, batch_loss_X, batch_loss_Y = self.test_batch(batch, mode = mode)
            X_pred = X_pred + X_predictions
            Y_pred = Y_pred + Y_predictions
            X_pred_female = X_pred_female + X_predictions_female
            X_pred_male = X_pred_male + X_predictions_male
            Y_pred_female = Y_pred_female + Y_predictions_female
            Y_pred_male = Y_pred_male + Y_predictions_male
            val_loss_X += batch_loss_X
            val_loss_Y += batch_loss_Y
        return X_pred,X_pred_male,X_pred_female, Y_pred, Y_pred_male, Y_pred_female, val_loss_X / len(evaluation_batch), val_loss_Y / len(evaluation_batch)  
    def get_evaluation_result(self, evaluation_batch, mode = "valid"):
        X_pred = []
        Y_pred = []
        X_pred_female = []
        X_pred_male = []
        Y_pred_female = []
        Y_pred_male = []
        val_loss_X = 0
        val_loss_Y = 0
        for i, batch in enumerate(evaluation_batch):
            X_predictions, _, _, Y_predictions,_, _, batch_loss_X, batch_loss_Y = self.test_batch(batch, mode = mode)
            X_pred = X_pred + X_predictions
            Y_pred = Y_pred + Y_predictions
            val_loss_X += batch_loss_X
            val_loss_Y += batch_loss_Y
        return X_pred, Y_pred, val_loss_X / len(evaluation_batch), val_loss_Y / len(evaluation_batch)    
    def get_fairness_metric_for_test(self, evaluation_batch):
        with open(f"./fairness_dataset/{self.opt['dataset']}/{self.opt['data_dir']}/item_IF.json","r") as f:
            item_if = json.load(f)
        all_DIF = 0
        c =0
        # DP
        X_pred_female = []
        X_pred_male = []
        Y_pred_female = []
        Y_pred_male = []
        # EO
        EO_X_pred_female = []
        EO_Y_pred_female = []
        EO_X_pred_male = []
        EO_Y_pred_male = []
        
        X_pred_female_num =0
        X_pred_male_num =0
        Y_pred_female_num =0
        Y_pred_male_num =0
        
        for i, batch in enumerate(evaluation_batch):
            
            index, seq, x_seq, y_seq, position, x_position, y_position, ts_d, ts_xd, ts_yd, X_last, Y_last, XorY, ground_truth, neg_list, masked_xd, neg_xd, masked_yd, neg_yd,gender,x_last_3,y_last_3 = self.unpack_batch_predict(batch)
            if self.opt['domain'] =="single":
                seq = None
            seqs_fea, x_seqs_fea, y_seqs_fea = self.model(seq, x_seq, y_seq, position, x_position, y_position)
            if self.opt['domain'] =="single":
                seq = None
                if self.opt['main_task'] == "X":
                    tmp = x_seq
                elif self.opt['main_task'] == "Y":
                    tmp = y_seq
            else:
                tmp =seq
            for id, (fea, sex) in enumerate(zip(tmp,gender)): # b * s * f
                if XorY[id] == 0: #if x domain
                    # print("share_fea:", share_fea.shape)#[256]
                    specific_fea = x_seqs_fea[id, X_last[id]]
                    if seqs_fea is not None:
                        # print("seqs_fea:", seqs_fea.shape)
                        share_fea = seqs_fea[id, -1]
                        X_score = self.model.lin_X(share_fea + specific_fea).squeeze(0) #256-> self.opt["source_item_num"]
                    else:
                        X_score = self.model.lin_X(specific_fea).squeeze(0)
                    topk_item = torch.topk(X_score, self.opt['topk'])[1].detach().cpu().numpy()
                    try:
                        predicted_IF = [item_if[str(d)] for d in topk_item if str(d) in list(item_if.keys())]
                        gt_item_id = [x_seq[id][i].item() for i in x_last_3[id]]
                        gt_IF = [item_if[str(d)] for d in gt_item_id if str(d) in list(item_if.keys())]
                        DIF = np.sum(predicted_IF) - np.sum(gt_IF)
                        all_DIF += DIF
                        c+=1
                    except:
                        print("Something wrong with IF!")
                        ipdb.set_trace()
                    if sex[0]==0:
                        X_pred_female_num+=1
                        X_pred_female+=topk_item.tolist()
                        EO_X_pred_female+=[item for item in topk_item.tolist()if item in x_seq[id]] 
                    elif sex[0]==1:
                        X_pred_male_num +=1
                        X_pred_male+=topk_item.tolist()
                        EO_X_pred_male+=[item for item in topk_item.tolist()if item in x_seq[id]] 
                else :
                    specific_fea = y_seqs_fea[id, Y_last[id]]
                    if seqs_fea is not None:
                        share_fea = seqs_fea[id, -1]
                        Y_score = self.model.lin_Y(share_fea + specific_fea).squeeze(0)
                    else:
                        Y_score = self.model.lin_Y(specific_fea).squeeze(0)
                    topk_item = torch.topk(Y_score, self.opt['topk'])[1].detach().cpu().numpy()
                    try:
                        predicted_IF = [item_if[str(d)] for d in topk_item + self.opt['source_item_num'] if str(d) in list(item_if.keys())]
                        gt_item_id = [y_seq[id][i].item() for i in y_last_3[id]]
                        gt_IF = [item_if[str(d)] for d in gt_item_id if str(d) in list(item_if.keys())]
                        DIF = np.sum(predicted_IF) - np.sum(gt_IF)
                        all_DIF+=DIF
                        c+=1
                    except:
                        print("Something wrong with IF!")
                        ipdb.set_trace()
                    if sex[0]==0:
                        Y_pred_female_num+=1
                        Y_pred_female+=topk_item.tolist()
                        EO_Y_pred_female+=[item for item in topk_item.tolist() if item in y_seq[id]] 
                    elif sex[0]==1:
                        Y_pred_male_num+=1
                        Y_pred_male+=topk_item.tolist()
                        EO_Y_pred_male+=[item for item in topk_item.tolist() if item in y_seq[id]] 
        avg_DIF = all_DIF / c
        if self.opt['main_task'] == "X" or self.opt['main_task'] == "dual":
            X_pred_female = torch.tensor(X_pred_female)
            X_pred_male = torch.tensor(X_pred_male)
            EO_X_pred_female = torch.tensor(EO_X_pred_female)
            EO_X_pred_male = torch.tensor(EO_X_pred_male)
        if self.opt['main_task'] == "Y" or self.opt['main_task'] == "dual":
            Y_pred_female = torch.tensor(Y_pred_female)
            Y_pred_male = torch.tensor(Y_pred_male)
            EO_Y_pred_female = torch.tensor(EO_Y_pred_female)
            EO_Y_pred_male = torch.tensor(EO_Y_pred_male)
        X_DP, X_EO = None, None
        Y_DP, Y_EO = None, None
        
        if self.opt['main_task'] == "X" or self.opt['main_task'] == "dual":
            #DP calculation    
            X_pred_female_dist= torch.zeros(self.opt['source_item_num'],dtype=torch.int64)
            num, count = X_pred_female.unique(return_counts=True)
            X_pred_female_dist[num] = count
            X_pred_female_dist = X_pred_female_dist.to(torch.float32)/X_pred_female_num
            # max_count = X_pred_female_dist.max()
            # if max_count > 0:
            #     X_pred_female_dist /= max_count
            
            X_pred_male_dist= torch.zeros(self.opt['source_item_num'],dtype=torch.int64)
            num, count = X_pred_male.unique(return_counts=True)
            X_pred_male_dist[num] = count
            X_pred_male_dist = X_pred_male_dist.to(torch.float32)/X_pred_male_num
            X_DP = distance.jensenshannon(X_pred_female_dist+1e-12,X_pred_male_dist+1e-12).item() if X_pred_female_dist is not None else None
            
            #EO calculation
            EO_X_pred_female_dist= torch.zeros(self.opt['source_item_num'],dtype=torch.int64)
            num, count = EO_X_pred_female.unique(return_counts=True)
            EO_X_pred_female_dist[num] = count
            EO_X_pred_female_dist = EO_X_pred_female_dist.to(torch.float32)/X_pred_female_num
            EO_X_pred_male_dist= torch.zeros(self.opt['source_item_num'],dtype=torch.int64)
            num, count = EO_X_pred_male.unique(return_counts=True)
            EO_X_pred_male_dist[num] = count
            EO_X_pred_male_dist = EO_X_pred_male_dist.to(torch.float32)/X_pred_male_num
            X_EO = distance.jensenshannon(EO_X_pred_female_dist+1e-12,EO_X_pred_male_dist+1e-12).item() 
        elif self.opt['main_task'] == "Y" or self.opt['main_task'] == "dual":
            #DP calculation
            Y_pred_female_dist= torch.zeros(self.opt['target_item_num'],dtype=torch.int64)
            tmp_num, tmp_count = Y_pred_female.unique(return_counts=True)
            Y_pred_female_dist[tmp_num] = tmp_count
            Y_pred_female_dist = Y_pred_female_dist.to(torch.float32)/Y_pred_female_num
            Y_pred_male_dist= torch.zeros(self.opt['target_item_num'],dtype=torch.int64)
            num, count = Y_pred_male.unique(return_counts=True)
            Y_pred_male_dist[num] = count
            Y_pred_male_dist = Y_pred_male_dist.to(torch.float32)/Y_pred_male_num
            Y_DP = distance.jensenshannon(Y_pred_female_dist+1e-12, Y_pred_male_dist+1e-12).item() 
            #EO calculation
            # try:
            EO_Y_pred_female_dist= torch.zeros(self.opt['target_item_num'],dtype=torch.int64)
            num, count = EO_Y_pred_female.unique(return_counts=True)
            if num.tolist():
                EO_Y_pred_female_dist[num] = count
                EO_Y_pred_female_dist = EO_Y_pred_female_dist.to(torch.float32)/Y_pred_female_num
            EO_Y_pred_male_dist= torch.zeros(self.opt['target_item_num'],dtype=torch.int64)
            num, count = EO_Y_pred_male.unique(return_counts=True)
            if num.tolist():
                EO_Y_pred_male_dist[num] = count
                EO_Y_pred_male_dist = EO_Y_pred_male_dist.to(torch.float32)/Y_pred_male_num
                Y_EO = distance.jensenshannon(EO_Y_pred_female_dist+1e-12, EO_Y_pred_male_dist+1e-12).item() if Y_pred_female_dist is not None else None
                    
            # except:
            #     ipdb.set_trace()
            #     pass

        
        print("number of X tested male:",X_pred_male_num)
        print("number of X tested female:",X_pred_female_num)
        print("number of Y tested male:",Y_pred_male_num)
        print("number of Y tested female:",Y_pred_female_num)
        print("number of total tested user:",Y_pred_male_num+Y_pred_female_num)
        print("number of test data:",Y_pred_male_num+Y_pred_female_num)
        # return distance.jensenshannon(X_pred_female_dist+1e-12,X_pred_male_dist+1e-12).item() ,distance.jensenshannon(Y_pred_female_dist+1e-12,Y_pred_male_dist+1e-12).item()
        return avg_DIF, X_DP, Y_DP, X_EO, Y_EO
    def test_batch(self, batch, mode):
        if mode == "valid":
            index, seq, x_seq, y_seq, position, x_position, y_position, ts_d, ts_xd, ts_yd, X_last, Y_last, XorY, ground_truth, neg_list, masked_xd, neg_xd, masked_yd, neg_yd, gender = self.unpack_batch_valid(batch)
        elif mode == "test":
            index, seq, x_seq, y_seq, position, x_position, y_position, ts_d, ts_xd, ts_yd, X_last, Y_last, XorY, ground_truth, neg_list, masked_xd, neg_xd, masked_yd, neg_yd,gender,x_last_3,y_last_3 = self.unpack_batch_predict(batch)
        
        if self.opt['domain'] =="single":
            seq = None
        if self.opt['time_encode']:
            seqs_fea, x_seqs_fea, y_seqs_fea = self.model(seq, x_seq, y_seq, position, x_position, y_position, ts_d, ts_xd, ts_yd)
        else:
            seqs_fea, x_seqs_fea, y_seqs_fea = self.model(seq, x_seq, y_seq, position, x_position, y_position)
            
        X_pred = []
        X_pred_female = []
        X_pred_male = []
        Y_pred = []
        Y_pred_female = []
        Y_pred_male = []
        batch_loss_X = 0
        batch_loss_Y = 0
        if self.opt['domain'] =="single":
            seq = None
            if self.opt['main_task'] == "X":
                tmp = x_seq
            elif self.opt['main_task'] == "Y":
                tmp = y_seq
        else:
            tmp = seq
        for id, (fea, sex) in enumerate(zip(tmp,gender)): # b * s * f
            if XorY[id] == 0: #if x domain
                # print("share_fea:", share_fea.shape)#[256]
                specific_fea = x_seqs_fea[id, X_last[id]]
                if seqs_fea is not None:
                    # print("seqs_fea:", seqs_fea.shape)
                    share_fea = seqs_fea[id, -1]
                    X_score = self.model.lin_X(share_fea + specific_fea).squeeze(0) #256-> self.opt["source_item_num"]
                else:
                    X_score = self.model.lin_X(specific_fea).squeeze(0)
                cur = X_score[ground_truth[id]] 
                score_larger = (X_score[neg_list[id]] > (cur + 0.00001)).data.cpu().numpy() #1000
                true_item_rank = np.sum(score_larger) + 1
                if sex[0]==0:
                    # print("female:",x_seq[id])
                    X_pred_female.append(true_item_rank)
                elif sex[0]==1:
                    # print("male:",x_seq[id])
                    X_pred_male.append(true_item_rank)
                X_pred.append(true_item_rank)
                batch_loss_X+=self.val_criterion(X_score, ground_truth[id]).item()
            else :
                # share_fea = seqs_fea[id, -1]
                specific_fea = y_seqs_fea[id, Y_last[id]]
                if seqs_fea is not None:
                    share_fea = seqs_fea[id, -1]
                    Y_score = self.model.lin_Y(share_fea + specific_fea).squeeze(0)
                else:
                    Y_score = self.model.lin_Y(specific_fea).squeeze(0)
                cur = Y_score[ground_truth[id]]
                score_larger = (Y_score[neg_list[id]] > (cur + 0.00001)).data.cpu().numpy()
                true_item_rank = np.sum(score_larger) + 1
                if sex[0]==0:
                    Y_pred_female.append(true_item_rank)
                elif sex[0]==1:
                    Y_pred_male.append(true_item_rank)
                Y_pred.append(true_item_rank)
                batch_loss_Y+=self.val_criterion(Y_score, ground_truth[id]).item()
        # ipdb.set_trace()
        return X_pred, X_pred_male, X_pred_female, Y_pred, Y_pred_male, Y_pred_female, batch_loss_X, batch_loss_Y #[B,1]
    def evaluate(self,test_dataloader,file_logger):
        self.model.eval()
        with torch.no_grad():
            test_X_pred,test_X_pred_male,test_X_pred_female, test_Y_pred, test_Y_pred_male, test_Y_pred_female,test_loss_X,test_loss_Y = self.get_evaluation_result_for_test(test_dataloader, mode = "test")
            test_X_MRR, test_X_NDCG_5, test_X_NDCG_10, test_X_HR_1, test_X_HR_5, test_X_HR_10 = self.cal_test_score(test_X_pred)
            test_Y_MRR, test_Y_NDCG_5, test_Y_NDCG_10, test_Y_HR_1, test_Y_HR_5, test_Y_HR_10 = self.cal_test_score(test_Y_pred)
            test_X_MRR_male, test_X_NDCG_5_male, test_X_NDCG_10_male, test_X_HR_1_male, test_X_HR_5_male, test_X_HR_10_male = self.cal_test_score(test_X_pred_male)

            test_X_MRR_female, test_X_NDCG_5_female, test_X_NDCG_10_female, test_X_HR_1_female, test_X_HR_5_female, test_X_HR_10_female = self.cal_test_score(test_X_pred_female)
            test_Y_MRR_male, test_Y_NDCG_5_male, test_Y_NDCG_10_male, test_Y_HR_1_male, test_Y_HR_5_male, test_Y_HR_10_male = self.cal_test_score(test_Y_pred_male)
            test_Y_MRR_female, test_Y_NDCG_5_female, test_Y_NDCG_10_female, test_Y_HR_1_female, test_Y_HR_5_female, test_Y_HR_10_female = self.cal_test_score(test_Y_pred_female)
            DIF, X_DP, Y_DP, X_EO, Y_EO = self.get_fairness_metric_for_test(test_dataloader)
            # result_str = 'Epoch {}: \n'.format(epoch) 
            result_str =""
            print("")
            print([test_X_MRR, test_X_NDCG_5, test_X_NDCG_10, test_X_HR_5, test_X_HR_10])
            best_X_test = [test_X_MRR, test_X_NDCG_5, test_X_NDCG_10, test_X_HR_5, test_X_HR_10]
            best_X_test_male = [test_X_MRR_male, test_X_NDCG_5_male, test_X_NDCG_10_male, test_X_HR_5_male,test_X_HR_10_male]
            best_X_test_female = [test_X_MRR_female,test_X_NDCG_5_female, test_X_NDCG_10_female, test_X_HR_5_female,test_X_HR_10_female]
            result_str += str({"DIF":DIF, "X_DP":X_DP, "Y_DP":Y_DP, "X_EO":X_EO, "Y_EO":Y_EO}) + "\n" 
            # result_str+=str({"JS_divergence_X":js_X})+"\n"
            # result_str+=str({"JS_divergence_Y":js_Y})+"\n"
            result_str += str({"Best X domain":[test_X_MRR, test_X_NDCG_5, test_X_NDCG_10, test_X_HR_5, test_X_HR_10]})+"\n"
            result_str += str({"Best X domain male":best_X_test_male})+"\n"
            result_str += str({"Best X domain female":best_X_test_female})+"\n"
            print("test overall X:",[test_X_MRR, test_X_NDCG_5, test_X_NDCG_10, test_X_HR_5, test_X_HR_10])
            print("test male X:",best_X_test_male)
            print("test female X:",best_X_test_female)
            best_Y_test = [test_Y_MRR, test_Y_NDCG_5, test_Y_NDCG_10, test_Y_HR_5, test_Y_HR_10]
            best_Y_test_male = [test_Y_MRR_male, test_Y_NDCG_5_male, test_Y_NDCG_10_male, test_Y_HR_5_male, test_Y_HR_10_male]
            best_Y_test_female = [test_Y_MRR_female, test_Y_NDCG_5_female, test_Y_NDCG_10_female, test_Y_HR_5_female, test_Y_HR_10_female]
            result_str += str({"Best Y domain": best_Y_test})+"\n"
            result_str += str({"Best Y domain male":best_Y_test_male})+"\n"
            result_str += str({"Best Y domain female":best_Y_test_female})+"\n"
            print("test overall Y",best_Y_test)
            print("test male Y:",best_Y_test_male)
            print("test female Y:",best_Y_test_female)
            # {'Best X domain':best_X_test}
            file_logger.log(result_str)
        return best_Y_test, best_Y_test_male,best_Y_test_female