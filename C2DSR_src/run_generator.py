import argparse
import time
import torch
from train_generator import main
import os
import glob
import ipdb
parser = argparse.ArgumentParser()
# dataset part
parser.add_argument('--data_dir', type=str, default='action_animation', help='Movie-Book, Entertainment-Education')

# model part
parser.add_argument('--model', type=str, default="C2DSR", help='model name')
parser.add_argument('--hidden_units', type=int, default=128, help='lantent dim.')

parser.add_argument('--num_blocks', type=int, default=2, help='lantent dim.')
parser.add_argument('--num_heads', type=int, default=1, help='lantent dim.')
parser.add_argument('--GNN', type=int, default=1, help='GNN depth.')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate.')
parser.add_argument('--optim', choices=['adamW','sgd', 'adagrad', 'adam', 'adamax'], default='adam',
                    help='Optimizer: sgd, adagrad, adam or adamax.')
parser.add_argument('--param_group', type = bool, default=False, help='param group')
parser.add_argument('--lr', type=float, default=0.001, help='Applies to sgd and adagrad.')
parser.add_argument('--lr_decay', type=float, default=1, help='Learning rate decay rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--decay_epoch', type=int, default=5, help='Decay learning rate after this epoch.')
parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
parser.add_argument('--leakey', type=float, default=0.1)
parser.add_argument('--maxlen', type=int, default=50)
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available(), help='Enables CUDA training.')
parser.add_argument('--lambda', type=float, default=0.7)       

# train part
parser.add_argument('--num_epoch', type=int, default=200, help='Number of total training epochs.')
parser.add_argument('--batch_size', type=int, default=256, help='Training batch size.')
parser.add_argument('--log_step', type=int, default=200, help='Print log every k steps.')
parser.add_argument('--log', type=str, default='log.txt', help='Write training log to file.')
parser.add_argument('--save_epoch', type=int, default=100, help='Save model checkpoints every k epochs.')
parser.add_argument('--save_dir', type=str, default='./generator_model', help='Root dir for saving models.')
parser.add_argument('--id', type=str, default=00, help='Model ID under which to save models.')
parser.add_argument('--seed', type=int, default=2023)
parser.add_argument('--load', dest='load', action='store_true', default=False,  help='Load pretrained model.')
parser.add_argument('--model_file', type=str, help='Filename of the pretrained model.')
parser.add_argument('--info', type=str, default='', help='Optional info for the experiment.')
parser.add_argument('--undebug', action='store_false', default=True)

# data augmentation
parser.add_argument('--augment_type', type=str, default="dropout", help='augment type: [crop,dropout]')
parser.add_argument('--crop_prob', type=float, default=0.7, help='crop probability')
parser.add_argument('--mask_prob', type=float, default=0.4, help='mask probability')
# time ssl
parser.add_argument('--window_size',type=int,default=3 ,help="window size for ssl")
parser.add_argument('--temp',type=float,default=0.05 ,help="temperature for ssl")
parser.add_argument('--ssl',type=str ,default="proto_CL" ,help="[mask_prediction,time_CL, augmentation_based_CL, no_ssl, proto_CL]")
#early stop
parser.add_argument("--pretrain_patience", type=int, default=5, help="early stop counter")
parser.add_argument("--finetune_patience", type=int, default=5, help="early stop counter")
parser.add_argument('--pooling',type=str,default="ave" ,help="pooling method")
parser.add_argument('--is_pooling',type=bool,default=True ,help="pooling or not")
#MoCo
parser.add_argument('--r',type=int,default=2048 ,help="queue size/negative sample") #warning : r must be divisible by batch_size
parser.add_argument('--m',type=float,default=0.999 ,help="momentum update ratio for moco")
parser.add_argument('--num_cluster', type=str, default= '2000,3000,4000' ,help="number of clusters for kmeans")
parser.add_argument('--warmup_epoch', type=int, default= 15 ,help="warmup epoch for cluster")
parser.add_argument('--mlp',type=bool,default=True ,help="use MoCo or not")
parser.add_argument('--cross_weight',type=float,default=0.001 ,help="cross domain weight for proto CL")
parser.add_argument('--num_proto_neg',type=int,default= 1280 ,help="intra domain weight for proto CL")

#pretrain
parser.add_argument('--training_mode',default= "finetune", type = str, help=["pretrain","joint_pretrain","finetune","joint_learn"])
parser.add_argument('--pretrain_model',type=str,default= None ,help="pretrain or not")
parser.add_argument('--pretrain_epoch',type=int,default= 30 ,help="pretrain epoch")
parser.add_argument('--load_pretrain_epoch',type=int,default= None ,help="pretrain epoch")

#time encoding
parser.add_argument('--time_encode',type=bool,default= False ,help="time encoding or not")
parser.add_argument('--time_embed', type=int, default=128, help='time dim.')
parser.add_argument('--speedup_scale', type=list, default = [0.3,0.5] , help='should be smaller than 1')
parser.add_argument('--slowdown_scale', type=int, default=[2,3], help='should be larger than 1') # 
parser.add_argument('--time_transformation', type=str, default="speedup", help="[speedup,slowdown]")

parser.add_argument('--valid_epoch', type=int, default=3, help='valid_epoch')
parser.add_argument('--mixed_included',type=bool,default= False ,help="mixed included or not")
parser.add_argument('--main_task',type=str,default="X" ,help="[dual, X, Y]")

parser.add_argument('--evaluation_model',type=str,default= None ,help="evaluation model")
parser.add_argument('--domain',type=str,default= "cross" ,help="target only or cross domain")
parser.add_argument('--topk',type=int,default= 5 ,help="topk item recommendation")

#what type of item to generate
parser.add_argument('--generate_type',type=str,default= "X" ,help="[X,Y,mixed]")
#how many item to generate
parser.add_argument('--generate_num',type=int,default= 5 ,help="number of item to generate")

args, unknown = parser.parse_known_args()

folder_list = glob.glob("./fairness_dataset/Movie_lens_time/*")
folder_list = [x.split("/")[-1] for x in folder_list]
data_dir = [x for x in folder_list if x not in ["data_preprocess.ipynb","data_preprocess.py","raw_data"]]
print(data_dir)
generate_types = ["X","Y","mixed"]
# generate_nums = [3, 5, 7, 10]
warmup_epoch = 1000 # prevent from doing clustering 
print("Config of Experiment:")
num_seeds = 1
for data_idx in range(len(data_dir)):
    data_name = data_dir[data_idx]
    for generate_type in generate_types:            
        args.data_dir = data_name
        args.generate_type = generate_type
        args.id = f"{generate_type}"
        args.warmup_epoch = warmup_epoch
        args.num_cluster = "2,3,4"
        main(args)

 