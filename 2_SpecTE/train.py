
import optuna
from datetime import datetime

from pretrain import train as train_pretrain
from pretrain import get_args_parser as get_args_parser_pretrain
from pretrain import get_dataset_info as get_dataset_info_pretrain

# from fine_tuning import train as train_finetune
from fine_tuning import get_args_parser as get_args_parser_finetune
from fine_tuning import get_dataset_info as get_dataset_info_finetune
from fine_tuning import star_one_train as train_finetune
from fine_tuning import predict



from blending_train import get_args_parser as get_args_parser_blending
from blending_train import blending



import pickle
import os



def main(model_pwd_path,finetune_path):


# ********************预训练**************************   
    if finetune_path==None:  
     
        # 定义预训练args参数
        args_pretrain = get_args_parser_pretrain()
        args_pretrain = args_pretrain.parse_args()
        dataset_info_pretrain = get_dataset_info_pretrain(args_pretrain)
        
        # 定义工作路径    
        args_pretrain.path_log= os.path.join(model_pwd_path,"pretrain/")
        # args_pretrain.path_log= r"F:/optuna_log/pretrain/"

        best_loss, finetune_path=train_pretrain(args_pretrain, dataset_info=dataset_info_pretrain,model_number= "OP")

        del args_pretrain,dataset_info_pretrain

#***************************5_50 *************************************
    # *******************微调****************************
    # 定义微调args参数
    args_fine_tune = get_args_parser_finetune()
    args_fine_tune = args_fine_tune.parse_args()
    args_fine_tune.date_range ='5_50' 
    dataset_info_finetune = get_dataset_info_finetune(args_fine_tune)
    
    # 定义预训练迁移路径  
    args_fine_tune.finetune = finetune_path
    
    # 定义工作路径
    args_fine_tune.path_log= os.path.join(model_pwd_path,"fine_tuning/")
    # args_fine_tune.train=False
    

    # 微调参数
    model_path_list=[]
    # none
    
    args_fine_tune.parameter_group='none'
    best_loss,model_path = train_finetune(args_fine_tune,dataset_info_finetune)
    model_path_list.append(model_path)

    # two
    args_fine_tune.parameter_group='two'
    best_loss,model_path = train_finetune(args_fine_tune,dataset_info_finetune)
    model_path_list.append(model_path)

    # each
    args_fine_tune.parameter_group='each'
    best_loss,model_path = train_finetune(args_fine_tune,dataset_info_finetune)
    model_path_list.append(model_path)


    # *******************集成****************************
    blending_parser=get_args_parser_blending()
    blending_args = blending_parser.parse_args()
    blending_args.date_range = '5_50'
    blending_args.model_path=model_path_list
    # blending_args.model_path=[r"F:\My_trial\paper\T\all\3_T=[50]-size=[69]\fine_tune\each_MAE(Pa=[69]-Di=[160]-Ha=[16]-De=[4]-mlp=[4.0])_5_50_stdFlux",
    #                           r"F:\My_trial\paper\T\all\3_T=[50]-size=[69]\fine_tune\none_MAE(Pa=[69]-Di=[160]-Ha=[16]-De=[4]-mlp=[4.0])_5_50_stdFlux",
    #                           r"F:\My_trial\paper\T\all\3_T=[50]-size=[69]\fine_tune\two_MAE(Pa=[69]-Di=[160]-Ha=[16]-De=[4]-mlp=[4.0])_5_50_stdFlux",]
    blending_args.path_save = os.path.join(model_pwd_path,"blending/5_50/")
    os.makedirs(blending_args.path_save, exist_ok=True)
    best_loss=blending(blending_args)
    print(best_loss)


#***************************50_999 *************************************
    # *******************微调****************************
    # 定义微调args参数
    args_fine_tune = get_args_parser_finetune()
    args_fine_tune = args_fine_tune.parse_args()

    args_fine_tune.date_range = '50_999'
    dataset_info_finetune = get_dataset_info_finetune(args_fine_tune)
    
    # 定义预训练迁移路径  
    args_fine_tune.finetune = finetune_path
    
    # 定义工作路径
    args_fine_tune.path_log= os.path.join(model_pwd_path,"fine_tuning/")
    args_fine_tune.train=True

    # 微调参数
    model_path_list=[]
    # none
    args_fine_tune.parameter_group='none'
    best_loss,model_path = train_finetune(args_fine_tune,dataset_info_finetune)
    model_path_list.append(model_path)

    # two
    args_fine_tune.parameter_group='two'
    best_loss,model_path = train_finetune(args_fine_tune,dataset_info_finetune)
    model_path_list.append(model_path)

    # each
    args_fine_tune.parameter_group='each'
    best_loss,model_path = train_finetune(args_fine_tune,dataset_info_finetune)
    model_path_list.append(model_path)


    # *******************集成****************************
    blending_parser=get_args_parser_blending()
    blending_args = blending_parser.parse_args()
    blending_args.date_range = '50_999'
    blending_args.model_path=model_path_list
    blending_args.path_save = os.path.join(model_pwd_path,"blending/50_999/")
    os.makedirs(blending_args.path_save, exist_ok=True)
    best_loss=blending(blending_args)
    print(best_loss)

    return best_loss
    




if __name__ == "__main__":

    model_path=r'./2_SpecTE/model_log/'
    finetune =None
    # finetune =r"./2_SpecTE/model_log/pretrain/SpecTE(Pa=[115]-Di=[160]-Ha=[16]-De=[8])/OP/weight_best.pkl"
    main(model_path,finetune)






