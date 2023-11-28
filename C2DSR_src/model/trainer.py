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
import time
from pathlib import Path
from utils import torch_utils
from model.C2DSR import C2DSR
from model.MoCo import MoCo
from model.NNCL import NNCL
from utils.MoCo_utils import compute_features
from utils.cluster import run_kmeans
from utils.time_transformation import TimeTransformation
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
        self.model.load_state_dict(checkpoint['model'])
        # self.opt = checkpoint['config']
        # self.model.load_state_dict(checkpoint)

    def save(self, filename):
        params = {
            'model': self.model.state_dict(),
            'config': self.opt,
        }
        try:
            # torch.save(self.model.state_dict(), filename)
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")
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
            inputs = [Variable(b.cuda()) for b in batch]
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
        return index, seq, x_seq, y_seq, position, x_position, y_position, ts_d, ts_xd, ts_yd, ground, share_x_ground, share_y_ground, x_ground, y_ground, ground_mask, share_x_ground_mask, share_y_ground_mask, x_ground_mask, y_ground_mask, corru_x, corru_y,masked_xd, neg_xd, masked_yd, neg_yd, augmented_d, augmented_xd, augmented_yd
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
        index, seq, x_seq, y_seq, position, x_position, y_position, ts_d, ts_xd, ts_yd, ground, share_x_ground, share_y_ground, x_ground, y_ground, ground_mask, share_x_ground_mask, share_y_ground_mask, x_ground_mask, y_ground_mask, corru_x, corru_y, masked_xd, neg_xd, masked_yd, neg_yd,augmented_d,augmented_xd,augmented_yd = self.unpack_batch(batch)
        
        
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
        self.BCE_criterion = nn.BCEWithLogitsLoss()
        self.CS_criterion = nn.CrossEntropyLoss(reduction='none')
        self.CL_criterion = nn.CrossEntropyLoss()
        self.proto_criterion = nn.CrossEntropyLoss()
        self.val_criterion = nn.CrossEntropyLoss(reduction='none')
        
        self.MoCo_X = MoCo(opt, self.model, dim = opt['hidden_units'], r = opt['r'], m = opt['m'], T = opt['temp'], mlp = opt['mlp'], domain = "X")
        self.MoCo_Y = MoCo(opt, self.model, dim = opt['hidden_units'], r = opt['r'], m = opt['m'], T = opt['temp'], mlp = opt['mlp'], domain = "Y")
        self.MoCo_mixed =MoCo(opt, self.model, dim = opt['hidden_units'], r = opt['r'], m = opt['m'], T = opt['temp'], mlp = opt['mlp'], domain = "mixed")
        self.NNCL = NNCL(opt,self.model, dim = opt['hidden_units'], r = opt['r'], m = opt['m'], T = opt['temp'], mlp = opt['mlp'])
        if opt['cuda']:
            self.model.cuda()
            self.BCE_criterion.cuda()
            self.CS_criterion.cuda()
            self.CL_criterion.cuda()
            self.MoCo_X.cuda()
            self.MoCo_Y.cuda()
            self.MoCo_mixed.cuda()
            self.NNCL.cuda()
            self.proto_criterion.cuda()
        if self.opt['param_group'] : 
            param_name = []
            for name, param in self.model.named_parameters():
                param_name.append(name)
            target_param_name = [s for s in param_name if "lin" in s]
            group1 =[p for n, p in self.model.named_parameters() if n not in target_param_name and p.requires_grad]
            group2 =[p for n, p in self.model.named_parameters() if n in target_param_name and p.requires_grad]
            self.optimizer = torch_utils.get_optimizer(opt['optim'],
                                                    [{'params': group1, 'lr': opt['lr']*0.1},
                                                        {'params': group2, 'lr': opt['lr']}],
                                                    opt['lr'])
        else:
            self.optimizer = torch_utils.get_optimizer(opt['optim'], self.model.parameters(), opt['lr'])
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
        return index, seq, x_seq, y_seq, position, x_position, y_position, ts_d, ts_xd, ts_yd, X_last, Y_last, XorY, ground_truth, neg_list, masked_xd, neg_xd, masked_yd, neg_yd,gender

    
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
        feat_X = self.get_embedding_for_ssl(data = x_seq, encoder = self.model.encoder_X, item_embed = self.model.item_emb_X, CL_projector = self.model.CL_projector_X)
        feat_Y = self.get_embedding_for_ssl(data = y_seq, encoder = self.model.encoder_Y, item_embed = self.model.item_emb_Y, CL_projector = self.model.CL_projector_Y)
        sim_X_Y = torch.matmul(feat_X, feat_Y.T)
        labels_X_Y = torch.arange(sim_X_Y.shape[0]).to(sim_X_Y.device).long()
        sim_mixed_X = torch.matmul(feat, feat_X.T)
        labels_mixed_X = torch.arange(sim_mixed_X.shape[0]).to(sim_mixed_X.device).long()
        sim_mixed_Y = torch.matmul(feat, feat_Y.T)
        labels_mixed_Y = torch.arange(sim_mixed_Y.shape[0]).to(sim_mixed_Y.device).long()
        logits = torch.cat([sim_X_Y, sim_mixed_X, sim_mixed_Y], dim=0)
        labels = torch.cat([labels_X_Y, labels_mixed_X, labels_mixed_Y], dim=0)
        loss = self.CL_criterion(logits, labels)
        return loss
    def train_batch(self, epoch, batch, i, cluster_result):
        self.model.train()
        self.optimizer.zero_grad()
        self.model.graph_convolution()
        cluster_result_X, cluster_result_Y, cluster_result_cross = cluster_result[0], cluster_result[1], cluster_result[2]
        index, seq, x_seq, y_seq, position, x_position, y_position, ts_d, ts_xd, ts_yd, ground, share_x_ground, share_y_ground, x_ground, y_ground, ground_mask, share_x_ground_mask, share_y_ground_mask, x_ground_mask, y_ground_mask, corru_x, corru_y,masked_xd, neg_xd, masked_yd, neg_yd,augmented_d,augmented_xd,augmented_yd = self.unpack_batch(batch)
        
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
                # MoCo_loss_mixed = self.MoCo_train(epoch,self.MoCo_mixed, augmented_d, cluster_result = None, index=index, task = "in-domain")

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
                
        if self.opt['time_encode']:
            seqs_fea, x_seqs_fea, y_seqs_fea = self.model(seq, x_seq, y_seq, position, x_position, y_position, ts_d, ts_xd, ts_yd)
        else:
            seqs_fea, x_seqs_fea, y_seqs_fea = self.model(seq, x_seq, y_seq, position, x_position, y_position)


        #取倒數10個
        used = 10
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
            else:
                loss = x_share_loss + y_share_loss + x_loss + y_loss
                # loss = x_share_loss + x_loss + y_loss
                # loss =  x_loss + y_loss
            # loss = self.opt["lambda"]*(x_share_loss + y_share_loss + x_loss + y_loss) + (1 - self.opt["lambda"]) * (x_mi_loss + y_mi_loss)
            
            self.prediction_loss += (x_share_loss.item() + y_share_loss.item() + x_loss.item() + y_loss.item())
            # self.prediction_loss += (x_share_loss.item() + x_loss.item() + y_loss.item())
            # self.prediction_loss +=  (x_loss.item() + y_loss.item())
            loss.backward()
            self.optimizer.step()
        elif self.opt['main_task'] == "X":
            share_x_result =  self.model.lin_X(seqs_fea[:,-used:]) # b * seq * X_num
            share_pad_result = self.model.lin_PAD(seqs_fea[:, -used:])  # b * seq * 1 #最後一維，即padding，score要是零
            share_trans_x_result = torch.cat((share_x_result, share_pad_result), dim=-1)

            # specific_x_result = self.model.lin_X(seqs_fea[:,-used:] + x_seqs_fea[:, -used:])  # b * seq * X_num
            # # specific_x_result = self.model.lin_X(x_seqs_fea[:, -used:])
            # specific_x_pad_result = self.model.lin_PAD(x_seqs_fea[:, -used:])  # b * seq * 1
            # specific_x_result = torch.cat((specific_x_result, specific_x_pad_result), dim=-1)


            x_share_loss = self.CS_criterion(
                share_trans_x_result.reshape(-1, self.opt["source_item_num"] + 1),
                share_x_ground.reshape(-1))  # b * seq
            # x_loss = self.CS_criterion(
            #     specific_x_result.reshape(-1, self.opt["source_item_num"] + 1),
            #     x_ground.reshape(-1))  # b * seq
            x_share_loss = (x_share_loss * (share_x_ground_mask.reshape(-1))).mean() #只取預測x的部分
            # x_loss = (x_loss * (x_ground_mask.reshape(-1))).mean()
            
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
            else:
                # loss = x_share_loss + x_loss
                loss = x_share_loss
            
            # self.prediction_loss += (x_share_loss.item() + x_loss.item())
            self.prediction_loss += x_share_loss.item()
            loss.backward()
            self.optimizer.step()
        elif self.opt['main_task'] == "Y":
            share_y_result =  self.model.lin_Y(seqs_fea[:,-used:]) # b * seq * X_num
            share_pad_result = self.model.lin_PAD(seqs_fea[:, -used:])  # b * seq * 1 #最後一維，即padding，score要是零
            share_trans_y_result = torch.cat((share_y_result, share_pad_result), dim=-1)

            specific_y_result = self.model.lin_Y(seqs_fea[:,-used:] + y_seqs_fea[:, -used:])  # b * seq * X_num
            specific_y_pad_result = self.model.lin_PAD(y_seqs_fea[:, -used:])  # b * seq * 1
            specific_y_result = torch.cat((specific_y_result, specific_y_pad_result), dim=-1)

            y_share_loss = self.CS_criterion(
                share_trans_y_result.reshape(-1, self.opt["target_item_num"] + 1),
                share_y_ground.reshape(-1))  # b * seq
            y_loss = self.CS_criterion(
                specific_y_result.reshape(-1, self.opt["target_item_num"] + 1),
                y_ground.reshape(-1))  # b * seq
            y_share_loss = (y_share_loss * (share_y_ground_mask.reshape(-1))).mean() #只取預測y的部分
            y_loss = (y_loss * (y_ground_mask.reshape(-1))).mean()
            
            if self.opt['training_mode'] =="joint_learn":
                if self.opt["ssl"]=="time_CL":
                    loss = y_share_loss + y_loss  + time_CL_loss
                    self.time_CL_loss += time_CL_loss.item()
                elif self.opt["ssl"]=="augmentation_based_CL":
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
            else:
                loss = y_share_loss + y_loss
            
            self.prediction_loss += (y_share_loss.item() + y_loss.item())
            loss.backward()
            self.optimizer.step()
        return loss.item()
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
        # start training
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
            cluster_result_X = None
            cluster_result_Y = None
            cluster_result_cross = None
            if self.opt['ssl'] == "proto_CL" and self.opt['training_mode'] == "joint_learn":
                if epoch+1>=self.opt['warmup_epoch']:
                    # x_domain
                    # features_X = compute_features(self.opt, train_dataloader, self.MoCo_X, domain = 'X')
                    # features_X[torch.norm(features_X,dim=1)>1.5] /= 2 #account for the few samples that are computed twice  
                    # features_X = features_X.numpy()
                    # cluster_result_X = run_kmeans(features_X, self.opt) 
                    # # y_domain
                    # features_Y = compute_features(self.opt, train_dataloader, self.MoCo_Y, domain = 'Y')
                    # features_Y[torch.norm(features_Y,dim=1)>1.5] /= 2 
                    # features_Y = features_Y.numpy()
                    # cluster_result_Y = run_kmeans(features_Y, self.opt)
                    # mixed domain
                    if self.opt['mixed_included']:
                        features_cross = compute_features(self.opt, train_dataloader, self.model, domain = 'mixed')
                        features_cross[torch.norm(features_cross,dim=1)>1.5] /= 2 
                        features_cross = features_cross.numpy()
                        cluster_result_cross = run_kmeans(features_cross, self.opt)
            for i,batch in enumerate(train_dataloader):
                global_step += 1
                loss = self.train_batch(epoch,batch, i, cluster_result = (cluster_result_X, cluster_result_Y, cluster_result_cross))
                train_loss += loss
                
            duration = time.time() - epoch_start_time
            print(format_str.format(datetime.now(), global_step, max_steps, epoch, \
                                            self.opt['num_epoch'], train_loss/num_batch, duration, current_lr))
            print("mi:", self.mi_loss/num_batch)
            print("time_CL_loss:", self.time_CL_loss/num_batch)
            print("augmentation_based_CL_loss:", self.augmentation_based_CL_loss/num_batch)
            print("proto_CL_loss:", self.proto_CL_loss/num_batch)
            print("pull_loss:", self.pull_loss/num_batch)
            print("NNCL_loss:", self.NNCL_loss/num_batch)
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

            if val_X_MRR > max(X_dev_score_history) or val_Y_MRR > max(Y_dev_score_history):
                patience = self.opt["finetune_patience"]
                test_X_pred, test_Y_pred,test_loss_X,test_loss_Y = self.get_evaluation_result(mixed_test_dataloader, mode = "test")
                test_X_MRR, test_X_NDCG_5, test_X_NDCG_10, test_X_HR_1, test_X_HR_5, test_X_HR_10 = self.cal_test_score(test_X_pred)
                test_Y_MRR, test_Y_NDCG_5, test_Y_NDCG_10, test_Y_HR_1, test_Y_HR_5, test_Y_HR_10 = self.cal_test_score(test_Y_pred)
                # female_test_X_pred, female_test_Y_pred,female_test_loss_X,female_test_loss_Y = self.get_evaluation_result(female_test_dataloader, mode = "test")
                # female_test_X_MRR,female_test_X_NDCG_5,female_test_X_NDCG_10, female_test_X_HR_1, female_test_X_HR_5, test_X_HR_10 = self.cal_test_score(female_test_X_pred)
                # test_Y_MRR, test_Y_NDCG_5, test_Y_NDCG_10, test_Y_HR_1, test_Y_HR_5, test_Y_HR_10 = self.cal_test_score(test_Y_pred)

                result_str = 'Epoch {}: \n'.format(epoch) 
                
                print("")
                if val_X_MRR > max(X_dev_score_history):
                    print("X best!")
                    print([test_X_MRR, test_X_NDCG_10, test_X_HR_10])
                    best_X_test = [test_X_MRR, test_X_NDCG_10, test_X_HR_10]
                    result_str += "X domain:" + str([test_X_MRR, test_X_NDCG_10, test_X_HR_10])+"\n"
                    model_save_dir = f"./models/{self.opt['data_dir']}/{self.opt['id']}/{self.opt['seed']}"
                    print(f"write models into path {model_save_dir} for embedding plotting")
                    Path(model_save_dir).mkdir(parents=True, exist_ok=True)
                    self.save(f"{model_save_dir}/X_model.pt")
                if val_Y_MRR > max(Y_dev_score_history):
                    print("Y best!")
                    print([test_Y_MRR, test_Y_NDCG_10, test_Y_HR_10])
                    best_Y_test = [test_Y_MRR, test_Y_NDCG_10, test_Y_HR_10]
                    result_str += "Y domain:" + str([test_Y_MRR, test_Y_NDCG_10, test_Y_HR_10])+"\n"
                    model_save_dir = f"./models/{self.opt['data_dir']}/{self.opt['id']}/{self.opt['seed']}"
                    print(f"write models into path {model_save_dir} for embedding plotting")
                    Path(model_save_dir).mkdir(parents=True, exist_ok=True)
                    self.save(f"{model_save_dir}/Y_model.pt")
                file_logger.log(result_str)
            else:
                patience -=1
                print("early stop counter:", self.opt["finetune_patience"]-patience)
                if patience == 0:
                    print("Early stop triggered!")
                    break
            X_dev_score_history.append(val_X_MRR)
            Y_dev_score_history.append(val_Y_MRR)
        final_str = str({'Best X domain':best_X_test}) + '\n' + str({'Best Y domain':best_Y_test}) + '\n'
        file_logger.log(final_str)
        
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
    def test_batch(self, batch, mode):
        if mode == "valid":
            index, seq, x_seq, y_seq, position, x_position, y_position, ts_d, ts_xd, ts_yd, X_last, Y_last, XorY, ground_truth, neg_list, masked_xd, neg_xd, masked_yd, neg_yd, gender = self.unpack_batch_valid(batch)
        elif mode == "test":
            index, seq, x_seq, y_seq, position, x_position, y_position, ts_d, ts_xd, ts_yd, X_last, Y_last, XorY, ground_truth, neg_list, masked_xd, neg_xd, masked_yd, neg_yd,gender = self.unpack_batch_predict(batch)
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
        for id, (fea,gender) in enumerate(zip(seqs_fea,gender)): # b * s * f
            # print("seqs_fea:", seqs_fea.shape) #[2029, 15, 256] 
            if XorY[id] == 0: #if x domain
                share_fea = seqs_fea[id, -1]
                # print("share_fea:", share_fea.shape)#[256]
                specific_fea = x_seqs_fea[id, X_last[id]]
                X_score = self.model.lin_X(share_fea + specific_fea).squeeze(0) #256-> self.opt["source_item_num"]
                # X_score = self.model.lin_X(share_fea).squeeze(0)
                cur = X_score[ground_truth[id]] 
                score_larger = (X_score[neg_list[id]] > (cur + 0.00001)).data.cpu().numpy() #1000
                true_item_rank = np.sum(score_larger) + 1
                if gender[0]==0:
                    X_pred_female.append(true_item_rank)
                elif gender[0]==1:
                    X_pred_male.append(true_item_rank)
                X_pred.append(true_item_rank)
                batch_loss_X+=self.val_criterion(X_score, ground_truth[id]).item()
            else :
                share_fea = seqs_fea[id, -1]
                specific_fea = y_seqs_fea[id, Y_last[id]]
                Y_score = self.model.lin_Y(share_fea + specific_fea).squeeze(0)
                # Y_score = self.model.lin_Y(share_fea).squeeze(0)
                cur = Y_score[ground_truth[id]]
                score_larger = (Y_score[neg_list[id]] > (cur + 0.00001)).data.cpu().numpy()
                true_item_rank = np.sum(score_larger) + 1
                if gender[0]==0:
                    Y_pred_female.append(true_item_rank)
                elif gender[0]==1:
                    Y_pred_male.append(true_item_rank)
                Y_pred.append(true_item_rank)
                batch_loss_Y+=self.val_criterion(Y_score, ground_truth[id]).item()
        return X_pred, X_pred_male, X_pred_female, Y_pred, Y_pred_male, Y_pred_female, batch_loss_X, batch_loss_Y #[B,1]
    def evaluate(self,test_dataloader,file_logger):
        test_X_pred,test_X_pred_male,test_X_pred_female, test_Y_pred, test_Y_pred_male, test_Y_pred_female,test_loss_X,test_loss_Y = self.get_evaluation_result_for_test(test_dataloader, mode = "test")
        test_X_MRR, test_X_NDCG_5, test_X_NDCG_10, test_X_HR_1, test_X_HR_5, test_X_HR_10 = self.cal_test_score(test_X_pred)
        test_Y_MRR, test_Y_NDCG_5, test_Y_NDCG_10, test_Y_HR_1, test_Y_HR_5, test_Y_HR_10 = self.cal_test_score(test_Y_pred)
        test_X_MRR_male, test_X_NDCG_5_male, test_X_NDCG_10_male, test_X_HR_1_male, test_X_HR_5_male, test_X_HR_10_male = self.cal_test_score(test_X_pred_male)
        test_X_MRR_female, test_X_NDCG_5_female, test_X_NDCG_10_female, test_X_HR_1_female, test_X_HR_5_female, test_X_HR_10_female = self.cal_test_score(test_X_pred_female)
        test_Y_MRR_male, test_Y_NDCG_5_male, test_Y_NDCG_10_male, test_Y_HR_1_male, test_Y_HR_5_male, test_Y_HR_10_male = self.cal_test_score(test_Y_pred_male)
        test_Y_MRR_female, test_Y_NDCG_5_female, test_Y_NDCG_10_female, test_Y_HR_1_female, test_Y_HR_5_female, test_Y_HR_10_female = self.cal_test_score(test_Y_pred_female)
        # result_str = 'Epoch {}: \n'.format(epoch) 
        result_str =""
        print("")
        print([test_X_MRR, test_X_NDCG_10, test_X_HR_10])
        best_X_test = [test_X_MRR, test_X_NDCG_10, test_X_HR_10]
        best_X_test_male = [test_X_MRR_male, test_X_NDCG_10_male, test_X_HR_10_male]
        best_X_test_female = [test_X_MRR_female, test_X_NDCG_10_female, test_X_HR_10_female]
        result_str += str({"Best X domain":[test_X_MRR, test_X_NDCG_10, test_X_HR_10]})+"\n"
        result_str += str({"Best X domain male":best_X_test_male})+"\n"
        result_str += str({"Best X domain female":best_X_test_female})+"\n"
        print([test_Y_MRR, test_Y_NDCG_10, test_Y_HR_10])
        print("test male X:",best_X_test_male)
        print("test female X:",best_X_test_female)
        best_Y_test = [test_Y_MRR, test_Y_NDCG_10, test_Y_HR_10]
        best_Y_test_male = [test_Y_MRR_male, test_Y_NDCG_10_male, test_Y_HR_10_male]
        best_Y_test_female = [test_Y_MRR_female, test_Y_NDCG_10_female, test_Y_HR_10_female]
        result_str += str({"Best Y domain": [test_Y_MRR, test_Y_NDCG_10, test_Y_HR_10]})+"\n"
        result_str += str({"Best Y domain male":best_Y_test_male})+"\n"
        result_str += str({"Best Y domain female":best_Y_test_female})+"\n"
        print(best_Y_test)
        print("test male Y:",best_Y_test_male)
        print("test female Y:",best_Y_test_female)
        # {'Best X domain':best_X_test}
        file_logger.log(result_str)
