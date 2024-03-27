import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
# print(current_dir)
# print(parent_dir)

import optuna
import plotly as plot
import torch
import numpy as np
import argparse
import torch
from train_rec import main
import ipdb
import glob
import argparse
import time
import torch
import os
import glob
import ipdb

from train_rec import main
parser = argparse.ArgumentParser()
# dataset part
parser.add_argument(
    "--data_dir",
    type=str,
    default="Food-Kitchen",
    help="Movie-Book, Entertainment-Education",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="Movie_lens_final",
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
    # default=False,
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
parser.add_argument("--mlp", type=bool, default=True, help="use pojector or not")
# args = parser.parse_args()

parser.add_argument('--cross_weight',type=float,default=0.001 ,help="cross domain weight for proto CL")
parser.add_argument('--num_proto_neg',type=int,default= 1280 ,help="intra domain weight for proto CL")

#pretrain
parser.add_argument('--training_mode',default= "joint_learn", type = str, help='["pretrain","joint_pretrain","finetune","joint_learn"]')
parser.add_argument('--pretrain_epoch',type=int,default= 70 ,help="pretrain epoch")
# parser.add_argument('--joint_learn',default= False , action='store_true', help="ssl + main task")
# parser.add_argument('--joint_pretrain', default= False, action='store_true', help="ssl + two encoder  loss task")

parser.add_argument('--load_pretrain_epoch',type=int,default= 20 ,help="pretrain epoch")
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
parser.add_argument('--alpha',type=float,default= 0.5 ,help="insertion ration for DGSA")
# nonoverlap user augmentation
parser.add_argument('--augment_size',type=int,default= 30 ,help="nonoverlap_augment size")
#interest clustering
parser.add_argument('--topk_cluster',type=str,default= 5 ,help="number of multi-view cluster")
parser.add_argument('--num_cluster', type=str, default= '100,100,200' ,help="number of clusters for kmeans")
parser.add_argument('--cluster_mode',type=str,default= "separate" ,help="separate or joint")
parser.add_argument('--warmup_epoch', type=int, default= 0, help="warmup epoch for cluster")
parser.add_argument('--cluster_ratio',type=float, default= 0.5 ,help="cluster ratio")
#group CL
parser.add_argument('--substitute_ratio',type=float, default= 0.4 ,help="substitute ratio")
parser.add_argument('--substitute_mode',type=str, default= "AGIR" ,help="IR, attention_weight")
# loss weight
parser.add_argument('--lambda_',nargs='*', default= [1,1] ,help="loss weight")
args, unknown = parser.parse_known_args()

def objective(trial, data_dir):
    # Generate the hyperparameters to be tuned
    #item augmentation
    alpha = trial.suggest_float("alpha", 0.2, 0.6, step=0.1)
    #interest clustering
    topk_cluster = trial.suggest_categorical("topk_cluster", [3, 5, 7])
    num_clusters_option = trial.suggest_categorical("num_clusters_option", ["300,300,300","400,400,400","500,500,500","600,600,600"])
    # warmup_epoch = trial.suggest_categorical("warmup_epoch",[1,10])
    #group CL
    substitute_ratio = trial.suggest_float("substitute_ratio", 0.2, 0.7, step=0.1)
    # loss weight
    lambda_0 = trial.suggest_float("lambda_0", 0.4, 1, step =0.1)  # Adjust the range as needed
    lambda_1 = trial.suggest_float("lambda_1", 0.4, 1, step =0.1)  # Adjust the range as needed
    args.ssl = "both"
    args.main_task = "Y"
    args.domain = "cross"
    args.seed = 2024
    args.training_mode = "joint_learn"
    args.data_augmentation = "item_augmentation"
    args.dataset = "Movie_lens_main"
    args.substitute_mode = "AGIR"
    args.id = f"alpha{round(alpha,3)}_topk{topk_cluster}_num_cluster{num_clusters_option}_substitute{round(substitute_ratio,3)}_lambda_{round(lambda_0,3)}_{round(lambda_1,3)}"
    args.data_dir = data_dir
    args.substitute_ratio = round(substitute_ratio,3)
    args.topk_cluster = topk_cluster
    args.num_cluster = num_clusters_option
    args.alpha = round(alpha,3)
    # args.warmup_epoch = warmup_epoch
    args.lambda_ = [round(lambda_0,3), round(lambda_1,3)]
    best_Y_test, best_Y_test_male, best_Y_test_female  = main(args)
    # return best_Y_test_male[1] - best_Y_test_female[1] #NDCG@5
    trial.set_user_attr(key="best_Y_test", value=best_Y_test)
    trial.set_user_attr(key="best_Y_test_male", value=best_Y_test_male)
    trial.set_user_attr(key="best_Y_test_female", value=best_Y_test_female)
    return best_Y_test_male[1] - best_Y_test_female[1], best_Y_test[1]
# Optimize the hyperparameters using Optuna
def callback(study, trial):
    # if study.best_trials.number == trial.number:
    #     # Update the study's user attributes with the best trial's additional info
    #     study.set_user_attr('best_Y_test_male', value = trial.user_attrs['best_Y_test_male'])
    #     study.set_user_attr('best_Y_test_female', value = trial.user_attrs['best_Y_test_female'])
    for best_trial in study.best_trials:
        if best_trial.number == trial.number:
            # Update the study's user attributes with the best trial's additional info
            study.set_user_attr('best_Y_test', trial.user_attrs['best_Y_test'])
            study.set_user_attr('best_Y_test_male', trial.user_attrs['best_Y_test_male'])
            study.set_user_attr('best_Y_test_female', trial.user_attrs['best_Y_test_female'])
            break
folder = "Movie_lens_main"
folder_list = glob.glob(f"./fairness_dataset/{folder}/*")
folder_list = [x.split("/")[-1] for x in folder_list]
folder_list = ["drama_sci-fi"]
for folder in folder_list:
    study = optuna.create_study(directions=['minimize','maximize'])  # or 'minimize' based on your goal
    # study.optimize(objective, n_trials=50)
    study.optimize(lambda trial: objective(trial, folder), n_trials=50, callbacks=[callback])
    # Get the best parameters
    # best_params = study.best_params
    # best_value = study.best_value
    # best_Y_test_male = study.user_attrs.get('best_Y_test_male', None)
    # best_Y_test_female = study.user_attrs.get('best_Y_test_female', None)
    # print(f"{folder} - best_Y_test_male :", best_Y_test_male)
    # print(f"{folder} - best_Y_test_female :", best_Y_test_female)
    
    if study.best_trials:
        for best_trial in study.best_trials:
            print(f"{folder} - Trial#{best_trial.number} - best_params :", best_trial.params)
            print(f"{folder} - - Trial#{best_trial.number} - best_value :", best_trial.values)
            print(f"{folder} - Trial#{best_trial.number} - best_Y_test: {best_trial.user_attrs['best_Y_test']}")
            print(f"{folder} - Trial#{best_trial.number} - best_Y_test_male: {best_trial.user_attrs['best_Y_test_male']}")
            print(f"{folder} - Trial#{best_trial.number} - best_Y_test_female: {best_trial.user_attrs['best_Y_test_female']}")
        file_path = f'param_tune_result/{folder}.txt'
        with open(file_path, 'w') as file:
            for best_trial in study.best_trials:
                file.write(f"{folder} - Trial#{best_trial.number} - best_params :{best_trial.params}\n")
                file.write(f"{folder} - - Trial#{best_trial.number} - best_value :{best_trial.values}\n")
                file.write(f"{folder} - Trial#{best_trial.number} - best_Y_test: {best_trial.user_attrs['best_Y_test']}\n")
                file.write(f"{folder} - Trial#{best_trial.number} - best_Y_test_male: {best_trial.user_attrs['best_Y_test_male']}\n")
                file.write(f"{folder} - Trial#{best_trial.number} - best_Y_test_female: {best_trial.user_attrs['best_Y_test_female']}\n")              
                file.write("-"*50 + "\n")
    