from matplotlib.pyplot import axis
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import torch.utils.data as Data
import argparse
import torch.backends.cudnn as cudnn

from tqdm import tqdm
import pickle

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from timm.models.layers import trunc_normal_

import timm.optim.optim_factory as optim_factory
import util.lr_sched as lr_sched
from torch.utils.tensorboard import SummaryWriter

from util.util import *
from util.loss_curves import plot_and_save_loss_curves

from models.SpecTE_estimator import SpecTE_Estimator



def get_args_parser():
    # 定义参数
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=256,  #256
                        help ='The size of the batch.256')
    
    parser.add_argument('--n_epochs', type=int, default=30,#30  
                        help='Number of epochs to train.')

    parser.add_argument('--parameter_group',choices=['none', 'two', 'each'], default='none',
                        help='Parameter combination training rules.')

    #===========================状态参数=============================
    parser.add_argument('--train', type=bool, default=True,
                        help='Whether to enable the training function for the program.')
    
    parser.add_argument('--predict', type=bool, default=True,
                        help='Whether to enable the prediction function for the program.')
    
    parser.add_argument('--finetune', type=str, default=r"./2_SpecTE/model_log/pretrain/SpecTE(Pa=[230]-Di=[160]-Ha=[16]-De=[8])\OP\weight_best.pkl",
                         help='The path of the model to be fine tuned, if filled with None, will be trained from scratch.')

    # "E:\my_star\model_log\pretrain\目前最好一次：MAE(Pa=[115]-Di=[160]-Ha=[32]-De=[4])_all_add-noise0.05_w-d=0.4000\OP\weight_best.pkl"
    
    parser.add_argument('--cuda', default=True,
                        help='device to use for training / testing')
    
    parser.add_argument('--seed', default=0, type=int)


    #==========================模型参数===== =========================
    parser.add_argument('--net', choices=['SpecTE'], default='SpecTE',
                        help='The models you need to use.',)

    # Hyperparameters_Vit
    # Hyperparameters_SpecTE
    parser.add_argument('--Hyperparameters_SpecTE', 
                        default={'patch_size':115, # 将输入图像分割成补丁的大小。   # 230
                                 'embed_dim':160, # 嵌入维度
                                 'depth':8, #Encoder的层数         # 8
                                 'num_heads':16, # 编码器注意力头的数量
                                 'mlp_ratio':4., # MLP中隐层与嵌入维度的比例
                                 'drop_rate':0.0,     # 0.05
                                 'attn_drop_rate':0.0, #
                                 'drop_path_rate':0.1, #0.13
                                 'pos_drop_rate':0.3, # 0.13
                                 'patch_drop_rate':0.00, #
                                 'proj_drop_rate':0.1 #0.05
                                 },
                        help='''SpecTE的参数''')
    
    parser.add_argument('--global_pool',  type=bool, default=False,
                        help='The models you need to use.',)


    #=============================Optimizer parameters优化器参数==================================
    parser.add_argument('--weight_decay', type=float, default=0.1,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=0.001, metavar='LR',    #0.0003      0.0005
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0.000005, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=3, metavar='N',
                        help='epochs to warmup LR')

    #======================= =Dataset parameters数据选择和数据处理==================================
    
    parser.add_argument('--date_range', choices=['5_50', '50_999', 'all'], default='5_50',
                        help='选择数据集信噪比范围.',)
    
    parser.add_argument('--flux_size', default=3450, type=int,
                        help='images input size')
    
    parser.add_argument('--noise_model', type=list, default=[False,0.1],    # add_training_noise
                        help='加噪声 ')
    
    parser.add_argument('--Flux_std', type=bool, default=True,
                        help='Whether to 标准化流量数据.',)
    
    parser.add_argument('--label_list', type=list, 
                        default=['Teff[K]', 'Logg', 'RV', 'CH', 'NH', 'OH', 'NaH', 'MgH', 'AlH', 'SiH', 'SH', 
                                 'KH', 'CaH', 'TiH',  'VH', 'CrH','MnH', 'FeH', 'NiH', 'snrg',],  
                                 
                        help='The label data that needs to be learned.')
    # ['Teff[K]', 'Logg', 'RV', 'CH', 'NH', 'OH', 'NaH', 'MgH', 'AlH', 'SiH', 'SH', 
    #                              'KH', 'CaH', 'TiH',  'VH', 'CrH','MnH', 'FeH', 'NiH', 'snrg', 'obsid', 'uid',]
    # ['Teff[K]', 'Logg', 'CH', 'NH', 'OH', 'MgH', 'AlH', 'SiH', 'SH','KH', 'CaH', 'TiH', 'CrH','MnH', 'FeH', 'NiH', 'snrg'],  

     #===============================定义路径=======================================
    # parser.add_argument('--path_data_set', type=str, default = r'E:/my_star/预处理/data_after_processing/result',
    #                     help='数据所在位置',) r'./1_Data_download_and_preprocessing/Denoising_reference_set/data_after_processing/result'

    parser.add_argument('--path_data_set', type=str, default= r'./1_Data_download_and_preprocessing/Parameter_estimation_reference_set/data_after_processing/result',   # ./optuna_log/
                        help='The path of the data after preprocessed.')

    parser.add_argument('--path_log', type=str, default= './2_SpecTE/model_log/fine_tuning/',   # ./optuna_log/
                        help='The path to save the model and training logs after training.')
    
    parser.add_argument('--path_data_processed', type=str, default='./2_SpecTE/data_processed/fine_tuning_dataset/', 
                        help='The path to save training data after processing.')

    parser.add_argument('--loss_type', choices=['PDPL', 'MSE','MAE','SoothL1Loss'], default='PDPL',
                        help='PDPL:Probability Density Parameter Loss.概率密度分布',)
    
    # parser.add_argument('--weight_mode', choices=['loss','mae'], default='loss',
    #                     help='PDPL:Probability Density Parameter Loss.概率密度分布',)

    #===============================测试状态=======================================
    parser.add_argument('--DeepEnsemble', type=bool, default=True,
                        help='Whether to use a fine-tuning model')

    return parser




def get_dataset_info(args):
    
    setup_seed(args.seed)
    target_list=['label_config.pkl', 'test_flux.pkl', 'test_label.csv', 'train_flux.pkl', 'train_label.csv', 'valid_flux.pkl', 'valid_label.csv']
    data_set_temp_path = args.path_data_processed + "/{}{}".format(args.date_range,'_stdFlux' if args.Flux_std else '')
    if not os.path.exists(data_set_temp_path):
        os.makedirs(data_set_temp_path)
    dir_list = os.listdir(data_set_temp_path)
    print(dir_list)

    # 先判断有没有处理好的文件 有就不用再处理了
    if dir_list == target_list:
        label_config = pickle.load(open(data_set_temp_path + "/label_config.pkl", 'rb'))
        mean_label = label_config["label_mean"]
        std_label = label_config["label_std"]

        x_train = pickle.load(open(data_set_temp_path + "/train_flux.pkl", 'rb'))
        x_valid = pickle.load(open(data_set_temp_path + "/valid_flux.pkl", 'rb'))
        x_test = pickle.load(open(data_set_temp_path + "/test_flux.pkl", 'rb'))
        
        # y_train = pickle.load(open(data_set_temp_path + "/train_label.pkl", 'rb'))
        # y_valid = pickle.load(open(data_set_temp_path + "/valid_label.pkl", 'rb'))
        # y_test = pickle.load(open(data_set_temp_path + "/test_label.pkl", 'rb'))

        y_train = pd.read_csv(os.path.join(data_set_temp_path, 'train_label.csv'))
        y_valid = pd.read_csv(os.path.join(data_set_temp_path, 'valid_label.csv'))
        y_test = pd.read_csv(os.path.join(data_set_temp_path, 'test_label.csv'))
        del label_config 

    # 不需要集合时直接处理数据集
    elif args.date_range != 'all':

        # 计算数据集的均值和方差
                    
        label_path_all = args.path_data_set + "/labels_all.csv"
        label_all = pd.read_csv(label_path_all)
        label_all = label_all[args.label_list]
        # label_all = np.load(label_path_all,allow_pickle=True)
        # label_all = pd.DataFrame(label_all, columns=args.label_list)  

        label_all = label_all.applymap(lambda x: float(x))  
        std_label = np.sqrt(label_all.iloc[:, :19].var())
        mean_label = label_all.iloc[:, :19].mean()
        del label_all,label_path_all

        if args.Flux_std:
            flux_path_all = args.path_data_set + "/flux_all.npy"
            flux_all = np.load(flux_path_all)
            rows_with_nan = np.isnan(flux_all).any(axis=1)
            indices_of_rows_with_nan = np.where(rows_with_nan)[0]
            flux_all = np.delete(flux_all, indices_of_rows_with_nan, axis=0)
            std_flux = np.std(flux_all, axis=0)
            mean_flux = np.mean(flux_all, axis=0)
            del flux_all,flux_path_all

        path_flux = args.path_data_set + "/flux_{}.npy".format(args.date_range)
        label_flux = args.path_data_set + "/labels_{}.csv".format(args.date_range)
        # 加载数据
        flux = np.load(path_flux)
        label = pd.read_csv(label_flux)
        label = label[args.label_list].to_numpy()

        
        # 去除存在无穷值的数据
        rows_with_nan = np.isnan(flux).any(axis=1)
        indices_of_rows_with_nan = np.where(rows_with_nan)[0]
        flux = np.delete(flux, indices_of_rows_with_nan, axis=0)
        label = np.delete(label, indices_of_rows_with_nan, axis=0)
        label = pd.DataFrame(label, columns=[args.label_list])  
        label = label.applymap(lambda x: float(x))  
        
        
        # 标签标准化
        # std_label = np.sqrt(label.iloc[:, :17].var())
        # mean_label = label.iloc[:, :17].mean()
        # 标签标准化
        # 标准化标签（假设label_std是 DataFrame）

        label_std = (label.iloc[:, :19].values - mean_label.values) / std_label.values
        # 如果需要，可以将label_std转换回DataFrame：
        label_std = pd.DataFrame(label_std, columns=args.label_list[:19])


        # label_std = (label.iloc[:, :19].values - mean_label.values[:, np.newaxis]) / std_label.values[:, np.newaxis]

        # label_std = (label.iloc[:, :19] - mean_label) / std_label
        
        label_std['snrg'] = label['snrg']
        # 流量标准化
        if args.Flux_std:
            # std_flux = np.std(flux, axis=0)
            # mean_flux = np.mean(flux, axis=0)
            flux = (flux-mean_flux)/std_flux
        # else:
        #     Flux_3sigma_sc=StandardScaler()
        #     std_flux_T = Flux_3sigma_sc.fit_transform(flux.T)   # 对每条光谱数据进行标准化    
        #     flux = std_flux_T.T

        # 填充空值
        if pd.isnull(flux).any() or any(pd.isnull(label_std).any()):
            pd.DataFrame(flux).fillna(1, inplace=True)
            pd.DataFrame(label_std).fillna(1, inplace=True)
        
        # 划分数据集
        x_train, x_valid_test, y_train, y_valid_test = train_test_split(flux, label_std, test_size=0.3, random_state=123)
        x_test, x_valid, y_test, y_valid = train_test_split(x_valid_test, y_valid_test, test_size=0.67, random_state=123)

        print('size:', x_train.shape, y_train.shape,)
        print('Training set:', x_train.shape[0], 'samples', 'flux type:', type(x_train), 'Training labels:', type(y_train))
        print('Validation set:', x_valid.shape[0], 'samples')
        print('Testing set:', x_test.shape[0], 'samples')
        

        # X_train_torch = (pickle.load(open(args.path_reference_set + "train_flux.pkl", 'rb')) - flux_mean) / flux_std
        # X_valid_torch = (pickle.load(open(args.path_reference_set + "valid_flux.pkl", 'rb')) - flux_mean) / flux_std

        # 保存
        label_config = {
            "label_list": args.label_list,
            "label_mean": mean_label,
            "label_std": std_label,
            "flux_mean": mean_flux,
            "flux_std": std_flux,
        }

        with open(os.path.join(data_set_temp_path, 'label_config.pkl'), 'wb') as f:
            pickle.dump(label_config, f)
        with open(os.path.join(data_set_temp_path, 'train_flux.pkl'), 'wb') as f:
            pickle.dump(x_train, f)
        with open(os.path.join(data_set_temp_path, 'valid_flux.pkl'), 'wb') as f:
            pickle.dump(x_valid, f)
        with open(os.path.join(data_set_temp_path, 'test_flux.pkl'), 'wb') as f:
            pickle.dump(x_test, f)
        y_train.to_csv(os.path.join(data_set_temp_path, 'train_label.csv'), index=False)
        y_valid.to_csv(os.path.join(data_set_temp_path, 'valid_label.csv'), index=False)
        y_test.to_csv(os.path.join(data_set_temp_path, 'test_label.csv'), index=False)
        # with open(os.path.join(data_set_temp_path, 'train_label.pkl'), 'wb') as f:
        #     pickle.dump(y_train, f)
        # with open(os.path.join(data_set_temp_path, 'valid_label.pkl'), 'wb') as f:
        #     pickle.dump(y_valid, f)
        # with open(os.path.join(data_set_temp_path, 'test_label.pkl'), 'wb') as f:
        #     pickle.dump(y_test, f)


    # 用高低信噪比的数据合并成all数据集，保证训练集验证集测试集的集合一致
    elif args.date_range == 'all' and dir_list!=['label_config.pkl', 'test_flux.pkl', 'test_label.csv', 'train_flux.pkl', 'train_label.csv', 'valid_flux.pkl', 'valid_label.csv']:
        data_set_5_50_path = args.path_data_processed + "/dataset/5_50{}".format('_stdFlux' if args.Flux_std else '')
        data_set_50_999_path = args.path_data_processed + "/dataset/50_999{}".format('_stdFlux' if args.Flux_std else '')
        if os.path.exists(data_set_5_50_path) and os.path.exists(data_set_50_999_path):
            dir_list1 = os.listdir(data_set_5_50_path)
            dir_list2 = os.listdir(data_set_50_999_path)
            if dir_list1 == dir_list2 == ['label_config.pkl', 'test_flux.pkl', 'test_label.csv', 'train_flux.pkl', 'train_label.csv', 'valid_flux.pkl', 'valid_label.csv']:

                label_config_5_50 = pickle.load(open(data_set_5_50_path + "/label_config.pkl", 'rb'))
                mean_label_5_50 = label_config_5_50["label_mean"]
                std_label_5_50 = label_config_5_50["label_std"]

                label_config_50_999 = pickle.load(open(data_set_50_999_path + "/label_config.pkl", 'rb'))
                mean_label_50_999 = label_config_50_999["label_mean"]
                std_label_50_999 = label_config_50_999["label_std"]

                if (mean_label_5_50.equals(mean_label_50_999)) and (std_label_5_50.equals(std_label_50_999)):
                    x_train_5_50 = pickle.load(open(data_set_5_50_path + "/train_flux.pkl", 'rb'))
                    x_valid_5_50 = pickle.load(open(data_set_5_50_path + "/valid_flux.pkl", 'rb'))
                    x_test_5_50 = pickle.load(open(data_set_5_50_path + "/test_flux.pkl", 'rb'))

                    y_train_5_50 = pd.read_csv(os.path.join(data_set_5_50_path, 'train_label.csv'))
                    y_valid_5_50 = pd.read_csv(os.path.join(data_set_5_50_path, 'valid_label.csv'))
                    y_test_5_50 = pd.read_csv(os.path.join(data_set_5_50_path, 'test_label.csv'))

                    x_train_50_999 = pickle.load(open(data_set_50_999_path + "/train_flux.pkl", 'rb'))
                    x_valid_50_999 = pickle.load(open(data_set_50_999_path + "/valid_flux.pkl", 'rb'))
                    x_test_50_999 = pickle.load(open(data_set_50_999_path + "/test_flux.pkl", 'rb'))

                    y_train_50_999 = pd.read_csv(os.path.join(data_set_50_999_path, 'train_label.csv'))
                    y_valid_50_999 = pd.read_csv(os.path.join(data_set_50_999_path, 'valid_label.csv'))
                    y_test_50_999 = pd.read_csv(os.path.join(data_set_50_999_path, 'test_label.csv'))

                    x_train = np.concatenate([x_train_5_50, x_train_50_999], axis=0)
                    x_valid = np.concatenate([x_valid_5_50, x_valid_50_999], axis=0)
                    x_test = np.concatenate([x_test_5_50, x_test_50_999], axis=0)

                    y_train = pd.concat([y_train_5_50, y_train_50_999], axis=0)
                    y_valid = pd.concat([y_valid_5_50, y_valid_50_999], axis=0)
                    y_test = pd.concat([y_test_5_50, y_test_50_999], axis=0)

                    # 保存
                    label_config = {
                        "label_list": ['Teff[K]', 'Logg', 'CH', 'NH', 'OH', 'MgH', 'AlH', 'SiH', 'SH',
                                                        'KH', 'CaH', 'TiH', 'CrH','MnH', 'FeH', 'NiH', 'snrg'],
                        "label_mean": mean_label_5_50,
                        "label_std": std_label_5_50,
                    }

                    with open(os.path.join(data_set_temp_path, 'label_config.pkl'), 'wb') as f:
                        pickle.dump(label_config, f)
                    with open(os.path.join(data_set_temp_path, 'train_flux.pkl'), 'wb') as f:
                        pickle.dump(x_train, f)
                    with open(os.path.join(data_set_temp_path, 'valid_flux.pkl'), 'wb') as f:
                        pickle.dump(x_valid, f)
                    with open(os.path.join(data_set_temp_path, 'test_flux.pkl'), 'wb') as f:
                        pickle.dump(x_test, f)
                    y_train.to_csv(os.path.join(data_set_temp_path, 'train_label.csv'), index=False)
                    y_valid.to_csv(os.path.join(data_set_temp_path, 'valid_label.csv'), index=False)
                    y_test.to_csv(os.path.join(data_set_temp_path, 'test_label.csv'), index=False)

                    dir_list = os.listdir(data_set_temp_path)
                    print("数据集5_50和50_999验证无误,生成的训练集验证集同步")
                    del x_train_5_50, x_valid_5_50, x_test_5_50, y_train_5_50, y_valid_5_50, y_test_5_50 
                    del x_train_50_999,x_valid_50_999,x_test_50_999,y_train_50_999,y_valid_50_999,y_test_50_999
                    del x_train, x_valid, x_test, y_train, y_valid, y_test

                else:print("注意: all 与5_50和50_999不分割不同步")    
            else:print("注意: all 与5_50和50_999不分割不同步")
        else:print("注意: all 与5_50和50_999不分割不同步")

         

    else :
        raise Exception("检查{}下的内容。".format(data_set_temp_path))
    

    # 光谱像素点数:3456->3450
    x_train = x_train[:, 3:-3]
    x_valid = x_valid[:, 3:-3]
    x_test = x_test[:, 3:-3]

    X_train_torch = torch.tensor(x_train, dtype=torch.float32)
    X_valid_torch = torch.tensor(x_valid, dtype=torch.float32)
    X_test_torch = torch.tensor(x_test, dtype=torch.float32)

    y_train_torch = y_train[args.label_list].values
    y_valid_torch = y_valid[args.label_list].values
    y_test_torch = y_test[args.label_list].values

    y_train_torch = torch.tensor(y_train_torch, dtype=torch.float32)
    y_valid_torch = torch.tensor(y_valid_torch, dtype=torch.float32)
    y_test_torch = torch.tensor(y_test_torch, dtype=torch.float32)

    print("x_train_torch.shape:", X_train_torch.shape)
    # print("x_train_torch.dtype:", X_train_torch.dtype)
    print("y_train_torch.shape:", y_train_torch.shape)
    # print("y_train_torch.dtype:", y_train_torch.dtype)

    print("x_valid_torch:\n", X_valid_torch.shape)
    print("y_valid_torch:\n", y_valid_torch.shape)
    print("x_test_torch:\n", X_test_torch.shape)
    print("y_test_torch:\n", y_test_torch.shape)

    # print("x_train_torch:\n", X_train_torch)
    # print("y_train_torch:\n", y_train_torch)

    train_dataset = Data.TensorDataset(X_train_torch, y_train_torch)
    valid_dataset = Data.TensorDataset(X_valid_torch, y_valid_torch)
    test_dataset = Data.TensorDataset(X_test_torch, y_test_torch)

    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    valid_loader = Data.DataLoader(
        dataset=valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # True
        num_workers=0,
    )
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    mean_label = mean_label.values
    std_label = std_label.values
    print("label,std\n", std_label, "\nlabel,mean\n", mean_label)
    dataset_info = {
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "valid_labels": y_valid,
        "label_mean": mean_label,
        "label_std": std_label,
        "test_loader": test_loader,
        "test_labels": y_test
    }

    return dataset_info


def train(args, dataset_info, train_label=['Teff[K]', 'Logg', 'FeH'], model_number="SP0", cuda=True):

    setup_seed(args.seed)
    if torch.cuda.is_available()and args.cuda:
        cuda = True
    else:
        cuda = False

    cudnn.benchmark = True

    # 定义模型和对应的模型名
    if args.net == "SpecTE":
        Hyperparameters = args.Hyperparameters_SpecTE
        patch_size=Hyperparameters['patch_size']
        embed_dim=Hyperparameters['embed_dim']
        depth=Hyperparameters['depth']
        num_heads=Hyperparameters['num_heads']      
        mlp_ratio=Hyperparameters['mlp_ratio']
        drop_rate=Hyperparameters['drop_rate']
        pos_drop_rate=Hyperparameters['pos_drop_rate']
        patch_drop_rate=Hyperparameters['patch_drop_rate']
        # proj_drop_rate=Hyperparameters['proj_drop_rate']
        attn_drop_rate=Hyperparameters['attn_drop_rate']
        drop_path_rate=Hyperparameters['drop_path_rate']

        

        model_name = "SpecTE(Pa=[{}]-Di=[{}]-Ha=[{}]-De=[{}]-mlp=[{}])".format(patch_size, embed_dim, num_heads,depth,mlp_ratio)

        # print("model name : ", model_name)
        net = SpecTE_Estimator(inpute_size=args.flux_size, 
                                num_lable = len(train_label),
                                global_pool=False,
                                patch_size=patch_size,
                                embed_dim=embed_dim, 
                                depth=depth, 
                                num_heads=num_heads,
                                mlp_ratio=mlp_ratio,
                                drop_rate=drop_rate,
                                pos_drop_rate = pos_drop_rate,
                                patch_drop_rate= patch_drop_rate,
                                attn_drop_rate=attn_drop_rate,
                                drop_path_rate=drop_path_rate
                                )
        if cuda:
            net = net.to("cuda")
    else:
        raise Exception("模型名字错误，程序已终止。")
    
    model_name += "_{}{}".format(args.date_range,'_stdFlux' if args.Flux_std else '')
    if args.noise_model[0]:
        model_name += "_add-noise{}".format(args.noise_model[1])    
    # model_name += "_{}".format(args.loss_type)
    model_name = "{}_{}".format(args.parameter_group,model_name)

    # 定义优化器 和学习器
    # optimizer = torch.optim.Adam(net.parameters(), lr=0.001)  # 优化器选择Adam
    # optimizer = torch.optim.Adam(net.parameters(), lr=0.001)  # 优化器选择Adam
    # # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.2)  # 用学习率调整器动态调整优化器的学习率
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.2)
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * args.batch_size / 256 
    param_groups = optim_factory.param_groups_weight_decay(net, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr , betas=(0.9, 0.95))

    # 定义损失函数
    criterion = PDPL_loss  
    # 定义Tensorboard路径
    log_dir = args.path_log + model_name + '/' + model_number  # 日志
    writer = SummaryWriter(log_dir=log_dir)
    
        
        # 进行加载
    # 如果在命令行参数中指定了finetune（微调）则进行微调
    if args.finetune :
        # 加载预训练模型的权重，map_location='cpu'指定将所有张量加载到CPU上
        checkpoint_model = torch.load(args.finetune, map_location='cpu')

        # 打印加载预训练模型的信息
        print("Load pre-trained checkpoint from: %s" % args.finetune)
        
        # 从checkpoint中获取模型的状态字典
        # checkpoint_model = checkpoint['model']
        
        # 获取当前模型的状态字典
        state_dict = net.state_dict()
        
        # 检查预训练模型的分类头（'head.weight'和'head.bias'）是否与当前模型匹配
        # 如果它们存在但形状不匹配，则从预训练模型中删除这些键
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # 如果需要，对位置嵌入进行插值，以适应当前模型的输入尺寸
        # 这是必要的，因为不同的模型可能需要不同长度的位置嵌入
        # interpolate_pos_embed(model, checkpoint_model)

        # 加载预训练模型，strict=False允许模型和预训练权重在一定程度上不匹配
        msg = net.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        # 如果使用全局池化，则确保缺少的键是分类头和全连接层的权重和偏置
        # 这意味着这些层需要根据新任务重新初始化
        if args.global_pool:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        else:
            # 如果不使用全局池化，只检查分类头的权重和偏置
            assert set(msg.missing_keys) == {'head.weight', 'head.bias','mu.weight','mu.bias','sigma.0.weight','sigma.0.bias'}

        # 对分类头进行手动初始化，使用截断的正态分布
        # 这是因为我们可能已从预训练模型中删除了分类头的权重，或者需要重新调整它们
        trunc_normal_(net.mu.weight, std=2e-5)



    label_index = [args.label_list.index(i) for i in train_label]   # 把标签转成数值

    best_loss = np.inf
    best_mae = np.inf
    # Iterative optimization
    valid_out_mu = []
    valid_out_sigma = []
    best_std_mean = None

    valid_out_mu_mae = []
    valid_out_sigma_mae = []
    train_loss_list = []
    valid_loss_list = []
    lr_list=[]



    # 开始训练
    for epoch in tqdm(range(1, args.n_epochs + 1)):
        
        net.train() # 训练模式
        torch.cuda.empty_cache()    # 清空缓存

        train_mae = np.zeros(len(train_label)) # 记录训练过程中的平均绝对误差
        train_loss = 0.0    # 记录模型训练过程中的累计损失（所有批次数据的损失之和）

        # Train
        for step, (batch_x, batch_y) in enumerate(dataset_info["train_loader"]):

            if args.noise_model[0] and epoch > 5:
                batch_x = add_noise(batch_x,args.noise_model[1])

            batch_y = batch_y[:, label_index]
            if cuda:
                batch_x = batch_x.to("cuda")
                batch_y = batch_y.to("cuda")

            mu, sigma = net(batch_x)
            # 不同损失函数    
            loss = PDPL_loss(batch_y, mu, sigma)        



            train_loss += loss.to("cpu").data.numpy()


            optimizer.zero_grad()       # 清空梯度
            loss.backward()             # 反向传播，计算梯度
            optimizer.step()            # 更新梯度

            n_iter = (epoch - 1) * len(dataset_info["train_loader"]) + step + 1  # 局迭代步数n_iter，用于记录 Tensorboard 日志及其他训练过程中的信息。

            mu = mu.to("cpu").data.numpy() * dataset_info["label_std"][label_index] + dataset_info["label_mean"][
                label_index]                              # 反归一化处理 得到真实的预测值
            batch_y = batch_y.to("cpu").data.numpy() * dataset_info["label_std"][label_index] + \
                      dataset_info["label_mean"][label_index]   # 反归一化处理 得到真实的标签

            mae = mean_absolute_error(batch_y, mu, multioutput='raw_values')   # 计算平均绝对误差（MAE）
            train_mae += mae

            lr_sched.adjust_learning_rate(optimizer, step / len(dataset_info["train_loader"]) + epoch, args)
            writer.add_scalar('Train/loss', loss.to("cpu").data.numpy(), n_iter)
            # for i in range(len(train_label)):
            #     writer.add_scalar('Train/%s_MAE' % train_label[i], mae[i], n_iter)

        # scheduler.step()    # 更新优化器的学习率
        lr = optimizer.state_dict()['param_groups'][0]['lr']   # 返回优化器的状态字典
        train_loss /= (step + 1)   # train_loss/N 平均训练损失值
        train_mae /= (step + 1)   # 平均 MAE
        torch.cuda.empty_cache()   # 清空缓存
        net.eval()         # 将网络设置为评估模式


        # 定义一些验证集要记录和保存的参数
        valid_mae = np.zeros(len(label_index))  # 记录验证集的平均 MAE
        vlaid_diff_std = np.zeros(len(label_index))   # 记录验证集的平均差值
        valid_loss = 0.0#保存验证集的结果

        # 保存验证集估计的数据
        output_label = np.zeros(shape=(len(dataset_info["valid_loader"].dataset), len(train_label)))    # 原始数
        output_label_mu = np.zeros_like(output_label)       # 预测值mu
        output_label_sigma = np.zeros_like(output_label)    # 预测值sigma
        


        # Valid
        for step, (batch_x, batch_y) in enumerate(dataset_info["valid_loader"]):

            with torch.no_grad():  # 上下文管理器:代码块中所有的计算都不会计算梯度
                batch_y = batch_y[:, label_index]
                if cuda:
                    batch_x = batch_x.to("cuda")
                    batch_y = batch_y.to("cuda")


                mu, sigma = net(batch_x)
                if args.loss_type == "PDPL":
                    loss = PDPL_loss(batch_y, mu, sigma)
                
                else:
                    loss = criterion(mu, batch_y)

                
                valid_loss += loss.to("cpu").data.numpy()

                n_iter = (epoch - 1) * len(dataset_info["valid_loader"]) + step + 1

                sigma = np.sqrt(sigma.to("cpu").data.numpy()) * dataset_info["label_std"][label_index]
                mu = mu.to("cpu").data.numpy() * dataset_info["label_std"][label_index] + dataset_info["label_mean"][
                    label_index]
                batch_y = batch_y.to("cpu").data.numpy() * dataset_info["label_std"][label_index] + \
                          dataset_info["label_mean"][label_index]

                # 保存验证集估计结果 (反归一化后的数据)
                output_label[step * dataset_info["valid_loader"].batch_size:step * dataset_info["valid_loader"].batch_size + mu.shape[0]] = batch_y
                output_label_mu[step * dataset_info["valid_loader"].batch_size:step * dataset_info["valid_loader"].batch_size + mu.shape[0]] = mu
                output_label_sigma[step * dataset_info["valid_loader"].batch_size:step * dataset_info["valid_loader"].batch_size + mu.shape[0]] = sigma

                diff_std = (mu - batch_y).std(axis=0)  # 计算每列数据的标准差
                # sigma_mean = sigma.mean(axis=0)        # 计算每列数据的平均值

                vlaid_diff_std += diff_std

                mae = mean_absolute_error(batch_y, mu, multioutput='raw_values')    # 用sklearn计算mae

                valid_mae += mae

            writer.add_scalar('Valid/loss', loss.to("cpu").data.numpy(), n_iter)
            # print(step?)


        valid_loss /= (step)
        valid_mae /= (step)
        valid_mae_temp=valid_mae
        valid_mae_temp[0] = valid_mae_temp[0]/100
        valid_mae_mean=valid_mae_temp.mean()

        torch.save(net.state_dict(), log_dir + '/weight_temp.pkl')

        if valid_loss < best_loss:
            best_loss = valid_loss
            # 保存模型参数
            torch.save(net.state_dict(), log_dir + '/weight_best.pkl')
            # 保存验证集估计的数据
            valid_raw = np.array(output_label)
            valid_out_mu = np.array(output_label_mu)
            if args.loss_type == "PDPL": valid_out_sigma = np.array(output_label_sigma)

            
        if valid_mae_mean < best_mae:
            best_mae = valid_mae_mean
            torch.save(net.state_dict(), log_dir + '/weight_best_mae.pkl')
            # valid_out_mu_mae = np.array(output_label_err)
            # if args.loss_type == "PDPL":valid_out_sigma_mae = np.array(output_label_err)

        print("EPOCH %d | lr %f | train_loss %.4f | valid_loss %.4f" % (epoch, lr, train_loss, valid_loss),
              "| valid_mae", valid_mae,
              "| valid_diff_std", vlaid_diff_std)
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        lr_list.append(lr)

    #画loss曲线图 保存loss表
    plot_and_save_loss_curves(train_loss_list, valid_loss_list, args.path_log + model_name + '/' + model_number,model_name,lr_list=lr_list)



    # 保存验证集输出的结果
    file_path = os.path.join(args.path_log + model_name, "predict_valid.csv")

    if os.path.exists(file_path):
        # 如果文件存在，则读取
        df_predict_valid = pd.read_csv(file_path)
    else:
        # 如果文件不存在，创建一个空的 DataFrame
        df_predict_valid = pd.DataFrame()
        df_predict_valid = pd.DataFrame(dataset_info["valid_labels"],columns=['snrg'])


    
    # 添加或更新数据
    for i in range(len(train_label)):
        df_predict_valid[train_label[i]]= valid_raw[:, i]
        df_predict_valid[f"{train_label[i]}_mu"] = valid_out_mu[:, i]
        if args.loss_type == "PDPL": 
            df_predict_valid[f"{train_label[i]}_sigma"] = valid_out_sigma[:, i]

            
    # 保存 
    df_predict_valid.to_csv(file_path, index=False)



    return best_loss

    # 保存验证集输出的结果（带原始数据）
    # data_set_temp_path = args.path_data_processed + "/dataset/{}{}".format(args.date_range,'_stdFlux' if args.Flux_std else '')
    # df = pd.read_csv(data_set_temp_path + "/valid_label.csv")
    # for i in range(len(train_label)):
    #     df["%s_%s" % (model_name, train_label[i])] = valid_out_mu[:, i]
    #     df["%s_%s_err" % (model_name, train_label[i])] = valid_out_sigma[:, i]
    # df.to_csv(log_dir + "/valid_label_out.csv", index=False)


def predict(args, dataset_info,para_dict):
    setup_seed(args.seed)
    # 定义预测函数
    def one_predict(args,para_dict, test_loader, model_path):
        """
        预测函数，对于给定单个模型进行预测

        参数:
            args: 命令行参数对象，包含了模型相关的超参数
            para_dict: 字典类型，包含了模型的输出参数列表
            test_loader: 数据加载器，用于加载测试数据
            model_path: 字符串类型，模型文件的路径

        返回值:
            预测结果的标签和方差
        """
        print(model_path)

        # 本模型输出的label_list
        train_label = para_dict[model_path.split("/")[-1][:2]]

        if args.net == "SpecTE":
            Hyperparameters = args.Hyperparameters_SpecTE
            patch_size=Hyperparameters['patch_size']
            embed_dim=Hyperparameters['embed_dim']
            depth=Hyperparameters['depth']
            num_heads=Hyperparameters['num_heads']      
            mlp_ratio=Hyperparameters['mlp_ratio']
            net = SpecTE_Estimator(inpute_size=args.flux_size, 
                                num_lable = len(train_label),
                                global_pool=False,
                                patch_size=patch_size,
                                embed_dim=embed_dim, 
                                depth=depth, 
                                num_heads=num_heads,
                                mlp_ratio=mlp_ratio,
                                drop_rate=0.1,
                                pos_drop_rate = 0.05,
                                patch_drop_rate= 0.,
                                proj_drop_rate = 0.,
                                attn_drop_rate=0.,
                                drop_path_rate=0.05
                                ).to("cuda") 
        
        net.eval()
        net.load_state_dict(torch.load(model_path + "/weight_best.pkl"))
        
        output_label = np.zeros(shape=(len(test_loader.dataset), len(train_label)))
        output_label_err = np.zeros_like(output_label)
        for step, batch in tqdm(enumerate(test_loader)):
            with torch.no_grad():
                batch_x = batch[0]             
                mu, sigma = net(batch_x.to("cuda"))
                mu = mu.to("cpu").data.numpy()
                sigma = np.sqrt(sigma.to("cpu").data.numpy())


            output_label[step * test_loader.batch_size:step * test_loader.batch_size + mu.shape[0]] = mu
            output_label_err[step * test_loader.batch_size:step * test_loader.batch_size + mu.shape[0]] = sigma
        
        return [output_label, output_label_err]

    
    # 读取数据
    label_mean = dataset_info["label_mean"]
    label_std = dataset_info["label_std"]
    test_loader = dataset_info["test_loader"]

    # 定义模型名   
    if args.net == "SpecTE":
        Hyperparameters = args.Hyperparameters_SpecTE
        patch_size=Hyperparameters['patch_size']
        embed_dim=Hyperparameters['embed_dim']
        depth=Hyperparameters['depth']
        num_heads=Hyperparameters['num_heads']      
        mlp_ratio=Hyperparameters['mlp_ratio']
        

        model_name = "SpecTE(Pa=[{}]-Di=[{}]-Ha=[{}]-De=[{}]-mlp=[{}])".format(patch_size, embed_dim, num_heads,depth,mlp_ratio)

    model_name += "_{}{}".format(args.date_range,'_stdFlux' if args.Flux_std else '')
    if args.noise_model[0]:
        model_name += "_add-noise{}".format(args.noise_model[1])
    model_name = "{}_{}".format(args.parameter_group,model_name)

    # 用已有模型进行预测，存放到output_list_dir
    model_path = args.path_log + model_name
    model_list = [entry.name for entry in os.scandir(model_path) if entry.is_dir()]  # 所有模型
    output_list_dir = {key: [] for key in para_dict.keys()}  # 存放每组参数的预测结果
    if not args.DeepEnsemble:
        for key, label_list in para_dict.items():
            if key+'0' in model_list:
                output_list_dir[key].append(one_predict(args,para_dict,test_loader, model_path=model_path + "/{}0".format(key)))
    else:
        for model in model_list:

            out = one_predict(args, para_dict, test_loader, model_path=model_path + "/" + model)
            output_list_dir[model[:2]].append(out)
            
    
    # 从output_list_dir提取出mu_list[]和sigma_list[]
    mu_list = []     # 经过模型输出的mu
    sigma_list = []   # 经过模型输出的sigma
    # output_list_dir[一个参数集].shape = [模型号,mu/sigma, 样本数,参数集的参数数]
    for i in range(min(len(lst) for lst in output_list_dir.values())):
        # i=一个参数集训练的第i个模型
        mu = None
        sigma = None
        for j, key in enumerate(output_list_dir.keys()):   # 第j个参数集
            if j==0:
                mu = output_list_dir[key][i][0]   
                sigma = output_list_dir[key][i][1]
            else:
                mu = np.hstack((mu,output_list_dir[key][i][0]))  # np.hstack: 将两个数组沿着水平方向进行拼接
                sigma = np.hstack((sigma,output_list_dir[key][i][1]))
        mu_list.append(mu)  
        sigma_list.append(sigma)
    del output_list_dir
    mu_list = np.array(mu_list)
    if args.loss_type == "PDPL": sigma_list = np.array(sigma_list)

    # 计算多个模型求得的均值结果
    out_mu = mu_list.mean(0)
    if args.loss_type == "PDPL": 
        out_sigma = ((mu_list ** 2 + sigma_list ** 2)).mean(0) - out_mu ** 2   # 计算一组正态分布的平均方差
        out_sigma = np.sqrt(out_sigma)               # 平均标准差

    true_label_list = [label for sublist in para_dict.values() for label in sublist]
    train_label = ['Teff[K]', 'Logg', 'RV', 'CH', 'NH', 'OH', 'NaH', 'MgH', 'AlH', 'SiH', 'SH', 
                                 'KH', 'CaH', 'TiH',  'VH', 'CrH','MnH', 'FeH', 'NiH',]
    # train_label = ['Teff[K]', 'Logg', 'CH', 'NH', 'OH', 'MgH', 'AlH', 'SiH', 'SH', 'KH', 'CaH', 'TiH', 'CrH','MnH', 'FeH', 'NiH']
    train_label_index = [train_label.index(label) for label in true_label_list if label in train_label]
    train_label = true_label_list
    # train_label_index = [args.label_list.index(i) for i in train_label]



    # 打印结果 生成结果表并保存
    if args.path_log is not None:
        # print(dataset_info["test_labels"])
        
        #读取原始数据
        true_mu = pd.DataFrame(dataset_info["test_labels"], columns=train_label)
        finally_loss = PDPL_loss(torch.tensor(true_mu.values), torch.tensor(mu),  torch.tensor(sigma))

# 反归一化
        out_mu = out_mu * label_std[train_label_index] + label_mean[train_label_index]
        if args.loss_type == "PDPL":
            out_sigma *= label_std[train_label_index]
        del mu_list,sigma_list

        true_mu = true_mu* label_std[train_label_index] + label_mean[train_label_index]
        diff_std = (true_mu - out_mu).std(axis=0)
        diff_mean = (true_mu - out_mu).mean(axis=0)
        mae = mean_absolute_error(true_mu, out_mu, multioutput='raw_values')
        
        df_evaluation = pd.DataFrame({
            "μ": diff_mean.values,
            "σ": diff_std.values,
            "MAE": mae
        }, index=diff_mean.index)

        # 按照下面的顺序输出
        label_order = ['Teff[K]', 'Logg', 'RV', 'FeH', 'MgH', 'SiH', 'SH', 'KH', 'CaH', 'TiH', 'CrH', 
                                 'NiH', 'CH', 'NH', 'OH',  'AlH', 'MnH','NaH', 'VH',]


        # 重新排序
        df_evaluation = df_evaluation.loc[label_order]

        print(df_evaluation)
        df_evaluation.to_csv(model_path+ "/evaluation_result{}.csv".format(model_name), index=True)   
        
        # 创建预测的平均值和标准差DataFrame
 
        df_mu = pd.DataFrame(out_mu, columns=train_label)
        df_sigma = pd.DataFrame(out_sigma, columns=train_label)

        # 合并原始数据、预测平均值和标准差
        snrg= pd.DataFrame(dataset_info["test_labels"],columns=['snrg'])
                     
        predict_result = pd.concat([snrg,true_mu, df_mu.add_suffix('_mu'), df_sigma.add_suffix('_sigma')], axis=1)
        predict_result.to_csv(model_path + "/predict_result.csv", index=False)
        
        return finally_loss, model_path 

        # # 保存带原始数据的结果
        # data_set_temp_path = args.path_data_processed + "/{}{}".format(args.date_range,'_stdFlux' if args.Flux_std else '')
        # df = pd.read_csv(data_set_temp_path+'/test_label.csv')
        # cols = [col for col in df.columns if col in train_label]
        # df[cols] = df[cols] * label_std[train_label_index] + label_mean[train_label_index]
        # for i in range(len(train_label)):
        #     df["%s_%s" % (model_name, train_label[i])] = out_mu[:, i]
        #     if args.loss_type == "PDPL": df["%s_%s_err" % (model_name, train_label[i])] = out_sigma[:, i]
        # # df.to_csv(args.path_data_processed+'/result/test_label.csv'[:-4] + "_%s_out.csv" % model_name, index=False)
        # df.to_csv(model_path+'/test_label.csv'[:-4] + "_%s_out.csv" % model_name, index=False)
        # # df.to_csv(model_path + "/valid_label_{}_out.csv".format(model_name), index=False)
            

        # # 保存测试集输出的结果
        # df_predict_valid = pd.DataFrame()
        # # 添加或更新数据
        # for i in range(len(train_label)):
        #     df_predict_valid[train_label[i]] = out_mu[:, i]
        #     if args.loss_type == "PDPL": df_predict_valid[f"{train_label[i]}_err"] = out_sigma[:, i]
        # # 保存
        # file_path = os.path.join(model_path, "predict_test.csv") 
        # df_predict_valid.to_csv(file_path, index=False)

def get_para_dict(args):
    parameter_group=args.parameter_group
    if parameter_group=='none':
        para_dict = {'AL':['Teff[K]', 'Logg', 'RV', 'CH', 'NH', 'OH', 'NaH', 'MgH', 'AlH', 'SiH', 'SH', 
                                 'KH', 'CaH', 'TiH',  'VH', 'CrH','MnH', 'FeH', 'NiH',],}
    elif parameter_group=='two':
        # V1.0
        # para_dict = {
        #     'SP':['Teff[K]', 'Logg', 'FeH'],
        #     'CA':['RV', 'CH', 'NH', 'OH', 'NaH', 'MgH', 'AlH', 'SiH', 'SH', 'KH', 'CaH', 'TiH',  'VH', 'CrH','MnH', 'NiH',]}
                    # 'CH', 'NH', 'OH', 'MgH', 'AlH', 'SiH', 'SH', 'KH', 'CaH', 'TiH', 'CrH', 'MnH', 'NiH'
        # V 1.1
        para_dict = {
            'SP':['Teff[K]', 'Logg', 'FeH', 'RV'],
            'CA':['CH', 'NH', 'OH', 'NaH', 'MgH', 'AlH', 'SiH', 'SH', 'KH', 'CaH', 'TiH',  'VH', 'CrH','MnH', 'NiH',]}

    elif parameter_group=='each':
        # v1.0
        # para_dict = {
        #     'te':['Teff[K]'],
        #     'Lo':['Logg'],
        #     'RV':['RV'],
        #     'Fe':['FeH'],
        #     'Mg':['MgH'],
        #     'Si':['SiH'],
        #     'SH':['SH'],
        #     'KH':['KH'],
        #     'Ca':['CaH'],
        #     'Ti':['TiH'],
        #     'Cr':['CrH'],
        #     'Ni':['NiH'],
        #     'CH':['CH'],
        #     'NH':['NH'],
        #     'OH':['OH'],
        #     'Al':['AlH'],
        #     'Mn':['MnH'],
        #     'Na':['NaH'],
        #     'VH':['VH'],
        #     }
        # v2.0
        para_dict = {
            'te':['Teff[K]'],
            'Fe':['FeH'],
            'NH':['NH'],
            'RV':['RV'],
            'ot':['Logg', 'CH', 'OH', 'MgH', 'AlH', 'SiH', 'SH', 'KH', 'CaH', 'TiH', 'CrH', 'MnH', 'NiH', 'NaH', 'VH']
            }     
        # v3.0  对比不如 v2.0
        # para_dict = {
        #     'te':['Teff[K]'],
        #     'Fe':['FeH'],
        #     'RV':['RV'],
        #     'ot':['Logg', 'CH', 'NH', 'OH', 'MgH', 'AlH', 'SiH', 'SH', 'KH', 'CaH', 'TiH', 'CrH', 'MnH', 'NiH', 'NaH', 'VH']
        #     }  
        # v3.1   对比不如 v2.0
        # para_dict = {
        #     'te':['Teff[K]'],
        #     'Al':['AlH'],
        #     'RV':['RV'],
        #     'CH':['CH'],
        #     'ot':['Logg', 'CH', 'NH', 'OH', 'MgH' , 'SiH', 'SH', 'KH', 'CaH', 'TiH', 'CrH', 'MnH', 'NiH', 'NaH', 'VH','FeH']
        #     }  
        
    return para_dict


def star_one_train(args,dataset_info):

    para_dict = get_para_dict(args)

    No = 0
    if args.train:
        for key, value in para_dict.items():
            model_number= key + str(No),
            train(args, dataset_info=dataset_info,
                  train_label=value,
                  model_number= key + str(No), 
                  cuda=True)
            print(f'model: {model_number}, label_list: {value}')

    if args.predict:
        return predict(args, dataset_info, para_dict)        

    return 7

if __name__ == "__main__":

    args = get_args_parser()
    args = args.parse_args()
    setup_seed(args.seed)

    dataset_info = get_dataset_info(args)
    para_dict = get_para_dict(args)
    
    No = 1
    if args.train:
        for key, value in para_dict.items():
            model_number= key + str(No),
            train(args, dataset_info=dataset_info,
                  train_label=value,
                  model_number= key + str(No), 
                  cuda=True)
            print(f'model: {model_number}, label_list: {value}')


    if args.predict:
        best_loss=predict(args, dataset_info, para_dict)
        print(best_loss)


    

    
