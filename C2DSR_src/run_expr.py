import argparse
import time
import torch
from train_rec import main
import os
import glob
import ipdb
parser = argparse.ArgumentParser()
# dataset part
parser.add_argument(
    "--data_dir",
    type=str,
    default="Food-Kitchen",
    help="Movie-Book, Entertainment-Education",
)

# model part
parser.add_argument("--model", type=str, default="C2DSR", help="model name")
parser.add_argument("--hidden_units", type=int, default=128, help="lantent dim.")
parser.add_argument("--num_blocks", type=int, default=2, help="lantent dim.")
parser.add_argument("--num_heads", type=int, default=1, help="lantent dim.")
parser.add_argument("--GNN", type=int, default=1, help="GNN depth.")
parser.add_argument("--dropout", type=float, default=0.2, help="dropout rate.")
parser.add_argument(
    "--optim",
    choices=["sgd", "adagrad", "adam", "adamax"],
    default="adam",
    help="Optimizer: sgd, adagrad, adam or adamax.",
)
parser.add_argument('--param_group', type = bool, default=False, help='param group')
parser.add_argument(
    "--lr", type=float, default=0.001, help="Applies to sgd and adagrad."
)
parser.add_argument(
    "--lr_decay", type=float, default=1, help="Learning rate decay rate."
)
parser.add_argument(
    "--weight_decay",
    type=float,
    default=5e-4,
    help="Weight decay (L2 loss on parameters).",
)
parser.add_argument(
    "--decay_epoch", type=int, default=5, help="Decay learning rate after this epoch."
)
parser.add_argument(
    "--max_grad_norm", type=float, default=5.0, help="Gradient clipping."
)
parser.add_argument("--leakey", type=float, default=0.1)
parser.add_argument("--maxlen", type=int, default=50)
parser.add_argument("--cpu", action="store_true", help="Ignore CUDA.")
parser.add_argument(
    "--cuda",
    type=bool,
    default=torch.cuda.is_available(),
    help="Enables CUDA training.",
)
parser.add_argument("--lambda", type=float, default=0.7)

# train part
parser.add_argument(
    "--num_epoch", type=int, default=200, help="Number of total training epochs."
)
parser.add_argument("--batch_size", type=int, default=256, help="Training batch size.")
parser.add_argument(
    "--log_step", type=int, default=200, help="Print log every k steps."
)
parser.add_argument(
    "--log", type=str, default="log.txt", help="Write training log to file."
)
parser.add_argument(
    "--save_epoch", type=int, default=100, help="Save model checkpoints every k epochs."
)
parser.add_argument(
    "--save_dir", type=str, default="./saved_models", help="Root dir for saving models."
)
parser.add_argument(
    "--id", type=str, default=00, help="Model ID under which to save models."
)
parser.add_argument("--seed", type=int, default=2023)
parser.add_argument(
    "--load",
    dest="load",
    action="store_true",
    default=False,
    help="Load pretrained model.",
)
parser.add_argument("--model_file", type=str, help="Filename of the pretrained model.")
parser.add_argument(
    "--info", type=str, default="", help="Optional info for the experiment."
)
parser.add_argument("--undebug", action="store_false", default=True)

# data augmentation
parser.add_argument("--augment_type", type=str, default="dropout", help="augment type")
parser.add_argument("--crop_prob", type=float, default=0.7, help="crop probability")
parser.add_argument('--mask_prob', type=float, default=0.2, help='mask probability')
# time ssl
parser.add_argument("--window_size", type=int, default=3, help="window size for ssl")
parser.add_argument("--temp", type=float, default=0.05, help="temperature for ssl")
parser.add_argument(
    "--ssl",
    type=str,
    default="proto_CL",
    help="[time_CL, augmentation_based_CL, no_ssl, proto_CL]",
)

# early stop
parser.add_argument("--pretrain_patience", type=int, default=1000, help="early stop counter")
parser.add_argument("--finetune_patience", type=int, default=5, help="early stop counter")

parser.add_argument("--pooling", type=str, default="ave", help="pooling method")
parser.add_argument("--is_pooling", type=bool, default=True, help="pooling or not")

# MoCo
parser.add_argument(
    "--r", type=int, default=2048, help="queue size/negative sample"
)  # warning : r must be divisible by batch_size
parser.add_argument(
    "--m", type=float, default=0.999, help="momentum update ratio for moco"
)
parser.add_argument(
    "--num_cluster",
    type=str,
    default="100,100,4",
    help="number of clusters for kmeans",
)
parser.add_argument(
    "--warmup_epoch", type=int, default=15, help="warmup epoch for cluster"
)
parser.add_argument("--mlp", type=bool, default=True, help="use pojector or not")
# args = parser.parse_args()

parser.add_argument('--cross_weight',type=float,default=0.001 ,help="cross domain weight for proto CL")
parser.add_argument('--num_proto_neg',type=int,default= 1280 ,help="intra domain weight for proto CL")

#pretrain
parser.add_argument('--training_mode',default= "finetune", type = str, help='["pretrain","joint_pretrain","finetune","joint_learn"]')
parser.add_argument('--pretrain_epoch',type=int,default= 70 ,help="pretrain epoch")
# parser.add_argument('--joint_learn',default= False , action='store_true', help="ssl + main task")
# parser.add_argument('--joint_pretrain', default= False, action='store_true', help="ssl + two encoder  loss task")

parser.add_argument('--load_pretrain_epoch',type=int,default= None ,help="pretrain epoch")
parser.add_argument('--pretrain_model',type=str,default= None ,help="pretrain or not")

parser.add_argument('--time_encode',type=bool,default= False ,help="=time encoding or not")
parser.add_argument('--time_embed', type=int, default= 128, help='time dim.')
parser.add_argument('--speedup_scale', type=list, default = [0.3, 0.5] , help='speedup_scale')
parser.add_argument('--slowdown_scale', type=int, default= 2, help='slowdown_scale')
parser.add_argument('--time_transformation', type=str, default="speedup", help="[speedup,slowdown]")

parser.add_argument('--valid_epoch', type=int, default= 3, help='valid_epoch')
parser.add_argument('--mixed_included',type=bool, default= False ,help="mixed included or not")
parser.add_argument('--main_task',type=str, default="Y" ,help="[dual, X, Y]")


parser.add_argument('--data_augmentation',type=str,default= None ,help="[item_generation, user_generation]")
parser.add_argument('--evaluation_model',type=str,default= None ,help="evaluation model")
parser.add_argument('--domain', type=str,default= "cross" ,help="target only or cross domain")
parser.add_argument('--topk', type=int,default= 10 ,help="topk item recommendation")
# item generation
parser.add_argument('--generate_type',type=str,default= "X" ,help="[X,Y,mixed]")
parser.add_argument('--generate_num',type=int,default= 5 ,help="number of item to generate")

# nonoverlap user augmentation
parser.add_argument('--augment_size',type=int,default= 30 ,help="nonoverlap_augment size")
parser.add_argument('--topk_cluster',type=str,default= 5 ,help="number of multi-view cluster")
args, unknown = parser.parse_known_args()

modes = [
    # "pretrain_mask_prediction","pretrain_mask_prediction_joint"
    # "fairness_baseline_Y_triple_pull"
    # "fairness_baseline_Y_time_eval","fairness_baseline_Y_single_domain_time_eval",
    # "fairness_maintaskY_nonoverlapAug_top20_random10_augmentSize40_eval","fairness_maintaskY_nonoverlapAug_top20_random10_augmentSize50_eval",
    # "fairness_maintaskY_nonoverlapAug_top20_random10_augmentSize40","fairness_maintaskY_nonoverlapAug_top20_random10_augmentSize50"
    # "fairness_maintaskY_usergen_biasFree_eval","fairness_maintaskY_usergen_sequentialSimTop5_repeat2_eval","fairness_maintaskY_usergen_sequentialSim_repeat2_eval",
    # "fairness_maintaskY_generatorAll_generatenumPoisson_20epoch_unbalance_max7_eval"
    # "fairness_baseline_Y_eval","fairness_baseline_Y_single_domain_eval"
    # "fairness_maintaskY_usergen_biasFree_fakelabel_eval","fairness_maintaskY_usergen_biasFree_eval"
    # "fairness_maintaskY_usergen_random_repeat2_eval"
    "fairness_maintaskY_interest_clusternum200_top10",
    # "fairness_maintaskY_generatorAll_generatenumPoisson_random_20epoch_eval"
    # "fairness_maintaskY_generatorAll_generatenumPoisson_20epoch_eval","fairness_maintaskY_generatorAll_generatenumPoisson_20epoch_unbalance_max7_eval"
    # "fairness_maintaskY_usergen_biasFree_lambda2_eval","fairness_maintaskY_none_biasFree_eval"
]

# evaluation_models = ["fairness_maintaskY_usergen_biasFree","fairness_maintaskY_usergen_sequentialSimTop5_repeat2","fairness_maintaskY_usergen_sequentialSim_repeat2"]

# evaluation_models = ["fairness_maintaskY_usergen_biasFree_lambda2","fairness_maintaskY_none_biasFree"]
# evaluation_models = ["fairness_baseline_Y","fairness_baseline_Y_single_domain"]
training_mode = "joint_learn"
domain = "cross"
main_task = "Y"
ssl = "interest_cluster"
data_augmentation = None
time_encode = False
topk_clusters = [5, 10]
num_clusters = ["100,100,100", "200,200,200","300,300,300"]
folder_list = glob.glob("./fairness_dataset/Movie_lens_time/*")
folder_list = [x.split("/")[-1] for x in folder_list]
data_dir = [x for x in folder_list]
print(data_dir)
warmup_epoch = 10 
print("Config of Experiment:")
print(f"Modes: {modes}")
print("training_mode:", training_mode)
print(f"Data: {data_dir}")
num_seeds = 5
for data_idx in range(len(data_dir)):
    data_name = data_dir[data_idx]
    for i, num_cluster in enumerate(num_clusters):
        for topk_cluster in topk_clusters:
            for seed in range(1, num_seeds+1):
                args.training_mode = training_mode
                args.domain = domain
                args.main_task = main_task
                args.time_encode = time_encode
                args.topk_cluster = topk_cluster
                args.num_cluster = num_cluster
                args.data_augmentation = data_augmentation     
                args.data_dir = data_name
                # args.evaluation_model = f"fairness_maintaskY_interest_clusternum{num_cluster.split(',')[-1]}_top{topk_cluster}_new_male_female"
                args.id =  f"fairness_maintaskY_interest_clusternum{num_cluster.split(',')[-1]}_top{topk_cluster}_new_mixed"
                args.seed = seed
                args.warmup_epoch = warmup_epoch
                args.ssl = ssl
                main(args)

# for finetune item generation
# folder_list = glob.glob("./fairness_dataset/Movie_lens_time/*")
# folder_list = [x.split("/")[-1] for x in folder_list]
# data_dir = [x for x in folder_list if x not in ["data_preprocess.ipynb","data_preprocess.py","raw_data"]]
# print(data_dir)
# # data_dir = ["comedy_drama",'adventure_thriller']
# # generator_models = ["X_time","mixed_time"]
# # generate_types = ["X","mixed"]
# # generate_nums = [3, 5, 7]
# data_augmentation = 'item_augmentation'
# load_pretrain_epochs = [20]
# training_mode = "finetune"
# main_task = "Y"
# domain = "cross"
# time_encode = False
# print("Config of Experiment:")
# print(f"load_pretrain_epochs: {load_pretrain_epochs}")
# print(f"Data: {data_dir}")
# num_seeds = 5
# for data_idx in range(len(data_dir)):
#     data_name = data_dir[data_idx]
#     for epoch in load_pretrain_epochs:
#         for seed in range(1, num_seeds+1):
#             args.load_pretrain_epoch = epoch           
#             args.training_mode = training_mode
#             args.data_augmentation = data_augmentation
#             args.domain = domain
#             args.main_task = main_task
#             args.time_encode = time_encode
#             args.data_dir = data_name
#             id =f"fairness_maintask{main_task}_generatorAll_generatenumPoisson_sample5_IFTop3_{epoch}epoch"
#             args.id = id
#             args.seed = seed
#             args.num_cluster = "2,3,4"
#             args.warmup_epoch = 10000
#             main(args)
                    
# # evaluation
# folder_list = glob.glob("./fairness_dataset/Movie_lens_time/*")
# folder_list = [x.split("/")[-1] for x in folder_list]
# data_dir = [x for x in folder_list if x not in ["data_preprocess.ipynb","data_preprocess.py","raw_data"]]
# print(data_dir)
# # mode_name = glob.glob("./saved_models/comedy_thriller/fairness_maintaskY_generatorAll*20epoch")
# # evaluation_models = [ x.split("/")[-1] for x in mode_name]
# evaluation_models = ["fairness_maintaskY_generatorAll_generatenumPoisson_sample10_IFTop5_20epoch"]
# print("evaluation_models:", evaluation_models)
# # generate_types = ["X","Y","mixed"]
# load_pretrain_epochs = [20]
# training_mode = "evaluation"
# time_encode = False
# main_task = "Y"
# domain = "cross"
# print("Config of Experiment:")
# print(f"load_pretrain_epochs: {load_pretrain_epochs}")
# print(f"Data: {data_dir}")
# num_seeds = 5
# for data_idx in range(len(data_dir)):
#     data_name = data_dir[data_idx]
#     for i, model in enumerate(evaluation_models):
#         for seed in range(1, num_seeds+1):
#             args.evaluation_model = model
#             args.training_mode = training_mode
#             args.domain = domain
#             args.main_task = main_task
#             args.data_dir = data_name
#             args.time_encode = time_encode
#             id =f"{model}_eval"
#             args.id = id
#             args.seed = seed
#             args.num_cluster = "2,3,4"
#             args.warmup_epoch = 10000
#             main(args)
