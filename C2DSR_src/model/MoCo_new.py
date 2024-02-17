import torch
import torch.nn as nn
import copy
import ipdb
from random import sample
from utils.time_transformation import TimeTransformation
from utils.augmentation import TimeSpeedUp, TimeSlowDown, TimeReverse,TimeShift,TimeReorder

class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, opt, model, dim=128, r=1280, m=0.999, T=0.1, mlp = False, domain = "X"):
        super(MoCo, self).__init__()
        self.r = r
        self.m = m
        self.T = T
        self.mlp = mlp
        self.opt = opt
        self.pooling = opt["pooling"]
        if domain == "X":
            self.encoder_q = model.encoder_X
            self.encoder_k = copy.deepcopy(model.encoder_X)
            self.item_embed = model.item_emb_X
            self.CL_projector_q = model.CL_projector_X                   # for instance discrimination
            self.CL_projector_k = copy.deepcopy(model.CL_projector_X)
            self.projector = model.projector_X                           #for cross-domain clustering : project x,y to joint space
        elif domain == "Y":
            self.encoder_q = model.encoder_Y
            self.encoder_k = copy.deepcopy(model.encoder_Y)
            self.item_embed = model.item_emb_Y
            self.CL_projector_q = model.CL_projector_Y
            self.CL_projector_k = copy.deepcopy(model.CL_projector_Y)
            self.projector = model.projector_Y                          
        elif domain == "mixed":
            self.encoder_q = model.encoder
            self.encoder_k = copy.deepcopy(model.encoder)
            self.item_embed = model.item_emb
            self.CL_projector_q = model.CL_projector
            self.CL_projector_k = copy.deepcopy(model.CL_projector)
            self.projector = model.projector 
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  #not update by gradient
        for param_q, param_k in zip(self.CL_projector_q.parameters(), self.CL_projector_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        # create the queue
        self.register_buffer("queue", torch.randn(dim, r))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.time_transformation = TimeTransformation(opt)
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        if self.mlp:
            for param_q, param_k in zip(self.CL_projector_q.parameters(), self.CL_projector_k.parameters()):
                param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.r % batch_size == 0  # for simplicity
        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.r  # move pointer
        self.queue_ptr[0] = ptr
        
    def encode(self, encoder, seq, mlp, ts, mode = "q"):
        non_zero_mask = (seq != (self.opt["source_item_num"] + self.opt["target_item_num"])).long()
        position_id = non_zero_mask.cumsum(dim=1) * non_zero_mask
        seq_feat = self.item_embed(seq)
        if ts is not None:
            ts = torch.sort(ts,dim=-1)[0]
        feat = encoder(seq, seq_feat, position_id, ts, causality_mask = False)  # keys: NxC

        if self.pooling == "bert":
            feat = feat[:,0,:]
        elif self.pooling == "ave":
            feat= torch.sum(feat, dim=1)/torch.sum(non_zero_mask, dim=1).unsqueeze(-1)
        if mlp:
            if mode == "q":
                feat = self.CL_projector_q(feat)
            elif mode == "k":
                feat = self.CL_projector_k(feat)
        feat = nn.functional.normalize(feat, dim=1)
        return feat
    def forward(self, seq_q, seq_k=None, is_eval=False, cluster_result=None, index=None, task="in-domain", train_mode = "train", ts = None):
        
        if is_eval:
            k = self.encode(self.encoder_k, seq_q, mlp = self.mlp, ts = ts, mode = "k")  #???           
            return k
        if task == "in-domain":
            # Instance Discrimination
            with torch.no_grad():
                if train_mode == "train":
                    # compute key features
                        self._momentum_update_key_encoder()  # update the key encoder
                        k = self.encode(self.encoder_k, seq_k, mlp = self.mlp, ts = ts, mode = "k")  # keys: NxC
                else:
                        k = self.encode(self.encoder_k, seq_k, mlp = self.mlp, ts = ts, mode = "k")  # keys: NxC
            q = self.encode(self.encoder_q, seq_q, mlp = self.mlp, ts = ts, mode = "q")
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
            logits = torch.cat([l_pos, l_neg], dim=1)
            logits = logits / self.T
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

            if train_mode == "train":
                self._dequeue_and_enqueue(k)
        elif task == "cross-domain":
            q = self.encode(self.encoder_q, seq_q, mlp = False, ts = ts, mode = "q")
            q = self.projector(q) # project to the joint space of the prototypes
        # prototypical contrast
        if task=="in-domain":
            pass
        elif task=="cross-domain":
            if cluster_result is not None:  
                proto_logits = []
                for n, (im2cluster,prototypes) in enumerate(zip(cluster_result['im2cluster'],cluster_result['centroids'])):
                    # compute prototypical logits
                    sim = torch.mm(q,prototypes.t())/self.opt['temp']
                    sim_max, _ = torch.max(sim, dim=1, keepdim=True)
                    sim = sim - sim_max
                    logits_proto = torch.nn.functional.softmax(sim, dim=1)
                    # logits_proto = torch.exp(sim)
                    # z = torch.sum(logits_proto, dim=-1) + 1e-5
                    # logits_proto = logits_proto/z.unsqueeze(1)
                    proto_logits.append(logits_proto)
                return None, None, proto_logits, None, None, None
            else:
                return None, None, None, None, None, None


# # utils
# @torch.no_grad()
# def concat_all_gather(tensor):
#     """
#     Performs all_gather operation on the provided tensors.
#     *** Warning ***: torch.distributed.all_gather has no gradient.
#     """
#     tensors_gather = [torch.ones_like(tensor)
#         for _ in range(torch.distributed.get_world_size())]
#     torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

#     output = torch.cat(tensors_gather, dim=0)
#     return output