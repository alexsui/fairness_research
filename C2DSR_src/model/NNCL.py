import torch
import torch.nn as nn
import copy
import ipdb
from random import sample

class NNCL(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, opt, model, dim=128, r=1280, m=0.999, T=0.1, mlp = True):
        """
        dim: feature dimension (default: 128)
        r: queue size; number of negative samples/prototypes (default: 16384)
        m: momentum for updating key encoder (default: 0.999)
        T: softmax temperature 
        mlp: whether to use mlp projection
        """
        super(NNCL, self).__init__()

        self.r = r
        self.m = m
        self.T = T
        self.mlp = mlp
        self.opt = opt
        self.pooling = opt["pooling"]
        self.encoder_X = model.encoder_X
        self.encoder_Y = model.encoder_Y
        self.item_embed_X = model.item_emb_X
        self.item_embed_Y = model.item_emb_Y
        self.CL_projector_X = model.CL_projector_X      
        self.CL_projector_Y = model.CL_projector_Y # for instance discrimination
        # create the queue
        self.register_buffer("queue_X", torch.randn(dim, r))
        self.register_buffer("queue_Y", torch.randn(dim, r))
        self.queue_X = nn.functional.normalize(self.queue_X, dim=0)
        self.queue_Y = nn.functional.normalize(self.queue_Y, dim=0)

        self.register_buffer("queue_ptr_X", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_ptr_Y", torch.zeros(1, dtype=torch.long))
        
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, domain):
        if domain=="X":
            batch_size = keys.shape[0]
            ptr = int(self.queue_ptr_X)
            assert self.r % batch_size == 0  # for simplicity
            # replace the keys at ptr (dequeue and enqueue)
            self.queue_X[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.r  # move pointer
            self.queue_ptr_X[0] = ptr
        elif domain=="Y":
            batch_size = keys.shape[0]
            ptr = int(self.queue_ptr_Y)
            assert self.r % batch_size == 0  # for simplicity
            # replace the keys at ptr (dequeue and enqueue)
            self.queue_Y[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.r  # move pointer
            self.queue_ptr_Y[0] = ptr
        
    def encode(self,encoder,item_embed, CL_projector, seq, mlp, ts):
        non_zero_mask = (seq != (self.opt["source_item_num"] + self.opt["target_item_num"])).long()
        position_id = non_zero_mask.cumsum(dim=1) * non_zero_mask
        seq_feat = item_embed(seq)
        if ts is not None:
            ts = torch.sort(ts,dim=-1)[0]
        feat = encoder(seq, seq_feat, position_id, ts, causality_mask = False)  # keys: NxC

        if self.pooling == "bert":
            feat = feat[:,0,:]
        elif self.pooling == "ave":
            feat= torch.sum(feat, dim=1)/torch.sum(non_zero_mask, dim=1).unsqueeze(-1)
        if mlp:
            feat = CL_projector(feat)
    
        feat = nn.functional.normalize(feat, dim=1)
        return feat
    def forward(self, seq_X, seq_Y, ts = None):
        batch_size = seq_X.size(0)
        n_neighbor = 10
        feat_X = self.encode(self.encoder_X, self.item_embed_X,self.CL_projector_X, seq_X, self.mlp, ts)
        feat_Y = self.encode(self.encoder_Y, self.item_embed_Y,self.CL_projector_Y, seq_Y, self.mlp, ts)
        idx = (feat_X @ self.queue_Y.clone().detach()).topk(n_neighbor,dim=1)[1]
        nn_X = self.queue_Y[:, idx].reshape(feat_X.size(-1),-1).T #[B*n_neighbor,dim]
        idx = (feat_Y @ self.queue_X.clone().detach()).topk(n_neighbor,dim=1)[1]
        nn_Y = self.queue_X[:, idx].reshape(feat_Y.size(-1),-1).T  #[B*n_neighbor,dim]
        sim_X = (feat_X@nn_X.T)/self.T
        sim_Y = (feat_Y@nn_Y.T)/self.T
        positive_mask = torch.zeros_like(sim_X)
        for i in range(batch_size):
            positive_mask[i, i*n_neighbor:(i+1)*n_neighbor] = 1
        log_prob_X = torch.nn.functional.log_softmax(sim_X, dim=1)
        loss_X = -torch.sum(log_prob_X * positive_mask) / batch_size
        log_prob_Y = torch.nn.functional.log_softmax(sim_Y, dim=1)
        loss_Y = -torch.sum(log_prob_Y * positive_mask) / batch_size
        self._dequeue_and_enqueue(feat_X,domain="X")
        self._dequeue_and_enqueue(feat_Y,domain="Y")
        loss = loss_X+loss_Y
        return loss