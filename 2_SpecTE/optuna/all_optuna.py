
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
from blending_train import get_dataset_info as get_dataset_info_blending
from blending_train import blending

import pickle
import os

# 定义工作路径
# save_path = r'F:/My_trial/all/drop_pretrain/'
save_path =  r'./model_test/all/1_finetune_lr2/'
# save_path =  r'./model_test/all/test/'

# 如果需要在已有的study上运行，则加载
study_path =True    # True：默认加载study_temp，False：不加载  或者直接填路径

# 定义搜索的次数
n_trials=6

#定义一个全局变量用于计数
No_i = 1

def objective(trial,dataset_info_pretrain,dataset_info_finetune):
    global No_i, save_path
    
    try:
        # 参数列表
        
        # ******模型参数******
        #编码器
        patch_size = 230
        dim = 160
        depth = 8
        heads = 16
        mlp_ratio=4.0
        #解码器
        de_dim = 80
        de_depth = 1
        de_heads = 16
        

        # *****训练参数******

        # 预训练
        weight_decay_pretrain = 0.4
        drop_rate_pretrain = 0.
        
        # 微调
        lr=0.0005
        batch = 128
        weight_decay=0.1
        
        
        drop_rate=0.0
        attn_drop_rate=0.
        patch_drop_rate= 0.        #0.3比0.好一点点
        drop_path_rate=0.1
        pos_drop_rate = 0.3
        proj_drop_rate = 0.1


        # 优化列表
        # 已调参


        if No_i < 1:
            lr = 0.001
        elif  No_i<2:
            lr = 0.0005
        elif  No_i<3:
            lr = 0.0005  
        elif  No_i<4:
            lr = 0.0004
        elif  No_i<5:
            lr = 0.0003
        elif  No_i<6:
            lr = 0.0001  
        elif  No_i<7:
            lr = 0.005  


        elif No_i < 8:
            lr = 0.001
            drop_rate = 0.05
        elif  No_i<9:
            lr = 0.0006
            drop_rate = 0.05
        elif  No_i<10:
            lr = 0.0005  
            drop_rate = 0.05
        elif  No_i<11:
            lr = 0.0004
            drop_rate = 0.05
        elif  No_i<12:
            lr = 0.0003
            drop_rate = 0.05
        elif  No_i<13:
            lr = 0.0001  
            drop_rate = 0.05
        else:
            lr = 0.005   
            drop_rate = 0.05      


        trial.set_user_attr('lr', lr)
        trial.set_user_attr('drop_rate', drop_rate)

        # 定义当前工作路径
        model_working_path = os.path.join(save_path, "{}_lr=[{}]-drop=[{}]/".format(No_i,lr,drop_rate))
        No_i=No_i+1
        os.makedirs(model_working_path, exist_ok=True)




        
#********************预训练**************************        
        # # 定义预训练args参数
        # args_pretrain = get_args_parser_pretrain()
        # args_pretrain = args_pretrain.parse_args()
        # # dataset_info_pretrain = get_dataset_info_pretrain(args_pretrain)


        # args_pretrain.Hyperparameters_SpecTE={'patch_size':patch_size, # 将输入图像分割成补丁的大小。
        #                          'embed_dim':dim, # 嵌入维度
        #                          'depth':depth, #Encoder的层数
        #                          'num_heads':heads, # 编码器注意力头的数量
        #                          'decoder_embed_dim':de_dim, # 解码器的嵌入维度
        #                          'decoder_depth':de_depth, # 解码器的层数
        #                          'decoder_num_heads':de_heads, # 解码器的注意力头数量
        #                          'mlp_ratio':mlp_ratio, # MLP中隐层与嵌入维度的比例
        #                          'drop_rate':drop_rate_pretrain,
        #                          }
       
        # # 定义工作路径    
        # args_pretrain.path_log= os.path.join(model_working_path,"pretrain/")
        # # args_pretrain.path_log= r"F:/optuna_log/pretrain/"


        # args_pretrain.weight_decay = weight_decay_pretrain
        # # dataset_info = get_dataset_info(args)
        # best_loss, path=train_pretrain(args_pretrain, dataset_info=dataset_info_pretrain,model_number= "OP")
        
        # trial.set_user_attr('pretrain_loss', best_loss)

        # del args_pretrain,dataset_info_pretrain
        
#***************************微调*************************************
        # 定义微调args参数
        args_fine_tune = get_args_parser_finetune()
        args_fine_tune = args_fine_tune.parse_args()
        # dataset_info_finetune = get_dataset_info_finetune(args_fine_tune)

        # 定义预训练迁移路径  
        # args_fine_tune.finetune = path
        
        # 定义工作路径
        args_fine_tune.path_log= os.path.join(model_working_path,"fine_tune/")
        # args_fine_tune.path_log= r"F:/optuna_log/fine_tune/"

        # 微调参数
        args_fine_tune.batch_size = batch
        args_fine_tune.blr = lr
        args_fine_tune.weight_decay = weight_decay

        args_fine_tune.Hyperparameters_SpecTE={'patch_size':patch_size, # 将输入图像分割成补丁的大小。
                                 'embed_dim':dim, # 嵌入维度
                                 'depth':depth, #Encoder的层数
                                 'num_heads':heads, # 编码器注意力头的数量
                                 'mlp_ratio':mlp_ratio, # MLP中隐层与嵌入维度的比例
                                 'drop_rate':drop_rate,
                                 'attn_drop_rate':attn_drop_rate,
                                 'drop_path_rate':drop_path_rate,
                                 'pos_drop_rate':pos_drop_rate,
                                 'patch_drop_rate':patch_drop_rate,
                                 'proj_drop_rate':proj_drop_rate
                                 }

        model_path_list=[]
        # none
        args_fine_tune.parameter_group='none'
        best_loss,model_path = train_finetune(args_fine_tune,dataset_info_finetune)
        model_path_list.append(model_path)
        trial.set_user_attr('none_loss', best_loss.item())

        # two
        args_fine_tune.parameter_group='two'
        best_loss,model_path = train_finetune(args_fine_tune,dataset_info_finetune)
        model_path_list.append(model_path)
        trial.set_user_attr('two_loss', best_loss.item())

        # each
        args_fine_tune.parameter_group='each'
        best_loss,model_path = train_finetune(args_fine_tune,dataset_info_finetune)
        model_path_list.append(model_path)
        trial.set_user_attr('each_loss', best_loss.item())

        # para_dict = {'AL':['Teff[K]', 'Logg', 'FeH','CH', 'NH', 'OH', 'MgH', 'AlH', 'SiH', 'SH', 'KH', 'CaH', 'TiH', 'CrH', 'MnH', 'NiH'],}
        # No = 0
        # if args_fine_tune.train:
        #     for key, value in para_dict.items():
        #         model_number= key + str(No),
        #         best_loss=train_finetune(args_fine_tune, dataset_info=dataset_info_finetune,
        #             train_label=value,
        #             model_number= key + str(No), 
        #             cuda=True)
        #         print(f'model: {model_number}, label_list: {value}')
        # if args_fine_tune.predict:
        #     best_loss=predict(args_fine_tune, dataset_info_finetune, para_dict)



#************************************集成*****************************************
        blending_parser=get_args_parser_blending()
        blending_args = blending_parser.parse_args()
        blending_args.model_path=model_path_list
        # blending_args.model_path=[r"F:\My_trial\paper\T\all\3_T=[50]-size=[69]\fine_tune\each_MAE(Pa=[69]-Di=[160]-Ha=[16]-De=[4]-mlp=[4.0])_5_50_stdFlux",
        #                           r"F:\My_trial\paper\T\all\3_T=[50]-size=[69]\fine_tune\none_MAE(Pa=[69]-Di=[160]-Ha=[16]-De=[4]-mlp=[4.0])_5_50_stdFlux",
        #                           r"F:\My_trial\paper\T\all\3_T=[50]-size=[69]\fine_tune\two_MAE(Pa=[69]-Di=[160]-Ha=[16]-De=[4]-mlp=[4.0])_5_50_stdFlux",]
        blending_args.path_save = os.path.join(model_working_path,"blending/")
        os.makedirs(blending_args.path_save, exist_ok=True)
        best_loss=blending(blending_args)
        


        return best_loss
    

    
    except Exception as e:
        print(f"An exception occurred during optimization: {str(e)}")
        return 6


def save_study_at_every_step(study, trial):
    """每次试验完成后调用的回调函数，用于保存 study 对象。"""
    filename = os.path.join(save_path, f'study_temp.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(study, f)
        
    # 将 trials 数据保存为 CSV
    df = study.trials_dataframe()
    df.to_csv(os.path.join(save_path,'optuna_trials_temp.csv'), index=False)    



def main():

    global save_path,study_path,n_trials,No_i

    # 加载或创建study对象
    if study_path==False:
        # 创建（study）对象。
        study = optuna.create_study()
    else:
        # 加载已有（study）对象。
        if study_path==True:
            #如果study_path为True，则默认加载'study_temp.pkl'文件
            study_path = os.path.join(save_path, 'study_temp.pkl')   

        with open(study_path, 'rb') as f:
            study = pickle.load(f)
            # 查看study对象存在多少个 trial。并更改i值
            No_i = len(study.trials)


    # 使用try except 块来捕获可能的异常  
    try:

        # # 加载"预训练"数据
        # args = get_args_parser_pretrain()
        # args = args.parse_args()
        # dataset_info_pretrain = get_dataset_info_pretrain(args)

        # 加载"微调"数据
        args = get_args_parser_finetune()
        args = args.parse_args()
        dataset_info_finetune = get_dataset_info_finetune(args)    


        # 运行优化过程
        # study.optimize(objective, n_trials=10)
        study.optimize(lambda trial: objective(trial, None, dataset_info_finetune),n_trials=n_trials,callbacks=[save_study_at_every_step])
        
        # 打印最佳参数值
        print('Best Hyperparameters_ViT:', study.best_params)

        # 定义study保存文件夹
        study_save_path = os.path.join(save_path,'study_log/')
        os.makedirs(study_save_path, exist_ok=True)

        # 用时间命名study文件
        current_time = datetime.now().strftime('%m%d-%H%M')
        filename = os.path.join(study_save_path,f'study_{current_time}.pkl')

        # 保存 study 对象
        with open(filename, 'wb') as f:
            pickle.dump(study, f)

        # 将 trials 数据保存为 CSV
        df = study.trials_dataframe()
        df.to_csv(os.path.join(study_save_path,f'optuna_trials_{current_time}.csv'), index=False)

    except Exception as e:
        print(f"An exception occurred during optimization: {str(e)}")
        # 保存 study 对象
        with open(os.path.join(save_path,'study_err.pkl'), 'wb') as f:
            pickle.dump(study, f)

        # 将 trials 数据保存为 CSV
        df = study.trials_dataframe()
        df.to_csv(os.path.join(save_path,'optuna_trials_err.csv'), index=False) 


def save_py_to_work_dir():
    # 保存一份当前的代码到工作目录方便日后查看
    global save_path
    import shutil
    shutil.copy(os.path.abspath(__file__), save_path)


if __name__ == "__main__":
    main()
    save_py_to_work_dir()











# # 创建optunastudy对象
# study = optuna.create_study()


# try:
#     args = get_args_parser_pretrain()
#     args = args.parse_args()
#     # with open('F:/optuna_log/study.pkl', 'rb') as f:
#     #     study = pickle.load(f)
#     dataset_info_pretrain = get_dataset_info_pretrain(args)
#     args = get_args_parser_finetune()
#     args = args.parse_args()
#     dataset_info_finetune = get_dataset_info_finetune(args)


#     # 运行优化过程
#     # study.optimize(objective, n_trials=10)
#     study.optimize(lambda trial: objective(trial, dataset_info_pretrain,dataset_info_finetune), n_trials=60)
#     # 打印最佳参数值
#     print('Best Hyperparameters_ViT:', study.best_params)

#     # 保存 study 对象
#     with open('F:/optuna_log/study.pkl', 'wb') as f:
#         pickle.dump(study, f)
    
#     # 将 trials 数据保存为 CSV
#     df = study.trials_dataframe()
#     df.to_csv('F:/optuna_log/optuna_trials.csv', index=False)

# except Exception as e:
#     print(f"An exception occurred during optimization: {str(e)}")
#     df = study.trials_dataframe()
#     df.to_csv('F:/optuna_log/optuna_trials.csv', index=False)
#     # 保存study对象
#     with open('F:/optuna_log/study.pkl', 'wb') as f:
#         pickle.dump(study, f)