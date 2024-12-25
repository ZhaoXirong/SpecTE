from matplotlib.pyplot import axis
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import torch.utils.data as Data
import argparse
import torch.backends.cudnn as cudnn
import joblib

from tqdm import tqdm
import pickle

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from timm.models.layers import trunc_normal_
from sklearn.linear_model import LinearRegression
import timm.optim.optim_factory as optim_factory
import util.lr_sched as lr_sched
from torch.utils.tensorboard import SummaryWriter

from util.util import *
from util.loss_curves import plot_and_save_loss_curves

from models.SpecTE_estimator import SpecTE_Estimator
from collections import defaultdict




def get_args_parser():



    # 定义参数
    parser = argparse.ArgumentParser()

    #===========================核心参数=============================
    parser.add_argument('--date_range', choices=['5_50', '50_999'], default='5_50',
                        help='Select the range of S/N for the dataset.',)
        
    parser.add_argument('--path_data_set', type=str, default= r'4_SpecTE-LAMOST_catalog\1_download_and_preprocessing_LAMOSTDR11\Fits_preprocessed\5-6',   
                        help='The path of preprocessed data.')
    
    parser.add_argument('--path_save', type=str, default= r'4_SpecTE-LAMOST_catalog\2_catalog_SpecTE-LAMOST\5-6', 
                        help='Storage path of catalog.')

    
    parser.add_argument('--path_log', type=str, default= './2_SpecTE/model_log/', 
                        help='The path of trained model.')

    parser.add_argument('--batch_size', type=int, default=256,  #128
                        help ='The size of the batch.')
    

    #===========================其他参数=============================

    
    parser.add_argument('--seed', default=0, type=int)



    parser.add_argument('--net', choices=['SpecTE'], default='SpecTE',
                        help='The models you need to use.',)
    

    # Hyperparameters_SpecTE
    parser.add_argument('--Hyperparameters_SpecTE', 
                        default={'patch_size':115, # 将输入图像分割成补丁的大小。   # 230
                                 'embed_dim':160, # 嵌入维度
                                 'depth':8, #Encoder的层数       
                                 'num_heads':16, # 编码器注意力头的数量
                                 'mlp_ratio':4., # MLP中隐层与嵌入维度的比例
                                 'drop_rate':0.0,     # 0.05
                                 'attn_drop_rate':0.0, #
                                 'drop_path_rate':0.1, #0.13
                                 'pos_drop_rate':0.3, # 0.13
                                 'patch_drop_rate':0.00, #
                                 'proj_drop_rate':0.1 #0.05
                                 },
                        help='''Model parameters of SpecTE''')

    
    parser.add_argument('--flux_size', default=3450, type=int,
                        help='images input size')
    
    parser.add_argument('--noise_model', type=list, default=[False,0.1],    # add_training_noise
                        help='Add noise ')
    
    parser.add_argument('--Flux_std', type=bool, default=True,
                        help='Whether to standardized flux data.',)
    
    parser.add_argument('--label_list', type=list, 
                        default=['Teff[K]', 'Logg', 'RV', 'CH', 'NH', 'OH', 'NaH', 'MgH', 'AlH', 'SiH', 'SH', 
                                 'KH', 'CaH', 'TiH',  'VH', 'CrH','MnH', 'FeH', 'NiH', ],      #'snrg',
                        help='The label data that needs to be learned.')

    parser.add_argument('--DeepEnsemble', type=bool, default=True,
                        help='Whether to use a fine-tuning model')    


    return parser



def get_dataset_info(args, no = 0):
    
    setup_seed(args.seed)
    
    # 遍历文件获得文件名
    data_set_path = args.path_data_set + "/flux/"
    data_list = os.listdir(data_set_path)
    print(data_list)

    # 遍历文件获得文件名
    info_path = args.path_data_set + "/info/"
    info_list = os.listdir(info_path)
    print(info_list)


    # 加载数据
    flux = np.load(data_set_path+data_list[no])
    info = pd.read_csv(info_path + info_list[no])

    

    # 流量标准化
    label_config = pickle.load(open(args.path_log + "label_config.pkl", 'rb'))
    std_flux=label_config["flux_std"]
    mean_flux=label_config["flux_mean"]
    if args.Flux_std:
            
            flux = (flux-mean_flux)/std_flux

   
    # 填充空值
    if pd.isnull(flux).any():
        pd.DataFrame(flux).fillna(1, inplace=True)
    

    print('flux_shape:', flux.shape)
    print('info_shape:', info.shape)
    # flux = flux[:, 3:-3]

    # 将数据转化为tensor
    flux_torch = torch.tensor(flux, dtype=torch.float32)
    flux_dataset = Data.TensorDataset(flux_torch)

    loader = Data.DataLoader(
        dataset=flux_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    
    mean_label = label_config["label_mean"]
    std_label = label_config["label_std"]

    mean_label = mean_label[args.label_list].values
    std_label = std_label[args.label_list].values
    # print("label,std\n", std_label, "\nlabel,std\n", mean_label)

    dataset_info = {
        "label_mean": mean_label,
        "label_std": std_label,
        "loader": loader,
        "info": info
    }

    return dataset_info



def primary_predict(args, dataset_info,para_dict):
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
    test_loader = dataset_info["loader"]

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
    model_path = args.path_log +'fine_tuning/' + model_name
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
    sigma_list = np.array(sigma_list)

    # 计算多个模型求得的均值结果
    out_mu = mu_list.mean(0)
    out_sigma = ((mu_list ** 2 + sigma_list ** 2)).mean(0) - out_mu ** 2   # 计算一组正态分布的平均方差
    out_sigma = np.sqrt(out_sigma)               # 平均标准差

    true_label_list = [label for sublist in para_dict.values() for label in sublist]
    # train_label = ['Teff[K]', 'Logg', 'CH', 'NH', 'OH', 'MgH', 'AlH', 'SiH', 'SH', 'KH', 'CaH', 'TiH', 'CrH','MnH', 'FeH', 'NiH']
    train_label = args.label_list
    # ['Teff[K]', 'Logg', 'RV', 'FeH', 'MgH', 'SiH', 'SH', 'KH', 'CaH', 'TiH', 'CrH', 
    #                              'NiH', 'CH', 'NH', 'OH',  'AlH', 'MnH','NaH', 'VH',]
    train_label_index = [train_label.index(label) for label in true_label_list if label in train_label]
    train_label = true_label_list
    # train_label_index = [args.label_list.index(i) for i in train_label]

    


    # 打印结果 生成结果表并保存
    if args.path_log is not None:
        # print(dataset_info["test_labels"])
        
        #读取原始数据
        info = pd.DataFrame(dataset_info["info"])

        # 反归一化
        out_mu = out_mu * label_std[train_label_index] + label_mean[train_label_index]
        out_sigma *= label_std[train_label_index]
        del mu_list,sigma_list


        # 创建预测的平均值和标准差DataFrame
 
        df_mu = pd.DataFrame(out_mu, columns=train_label)
        df_sigma = pd.DataFrame(out_sigma, columns=train_label)

        # 读取原始数据
        info= pd.DataFrame(dataset_info["info"])
                     
        predict_result = pd.concat([info, df_mu.add_suffix('_mu'), df_sigma.add_suffix('_sigma')], axis=1)
        # predict_result.to_csv("./predict_result.csv", index=False)
        
        return predict_result

    

def star_one_predict(args,dataset_info):

    parameter_group=args.parameter_group
    
    if parameter_group=='none':
        para_dict = {'AL':['Teff[K]', 'Logg', 'RV', 'CH', 'NH', 'OH', 'NaH', 'MgH', 'AlH', 'SiH', 'SH', 
                                 'KH', 'CaH', 'TiH',  'VH', 'CrH','MnH', 'FeH', 'NiH',],}
    elif parameter_group=='two':
        para_dict = {
            'SP':['Teff[K]', 'Logg', 'FeH', 'RV'],
            'CA':['CH', 'NH', 'OH', 'NaH', 'MgH', 'AlH', 'SiH', 'SH', 'KH', 'CaH', 'TiH',  'VH', 'CrH','MnH', 'NiH',]}

    elif parameter_group=='each':
        
        para_dict = {
            'te':['Teff[K]'],
            'Lo':['Logg'],
            'RV':['RV'],
            'Fe':['FeH'],
            'Mg':['MgH'],
            'Si':['SiH'],
            'SH':['SH'],
            'KH':['KH'],
            'Ca':['CaH'],
            'Ti':['TiH'],
            'Cr':['CrH'],
            'Ni':['NiH'],
            'CH':['CH'],
            'NH':['NH'],
            'OH':['OH'],
            'Al':['AlH'],
            'Mn':['MnH'],
            'Na':['NaH'],
            'VH':['VH'],
            }

    return primary_predict(args, dataset_info, para_dict)        


def blending_train(args, dataset, train_label):


    param_name = train_label    

    # 存放读到的初级预测结果
    x_predict_list =[]
    x_predict_err_list = []

    for j ,data in enumerate(dataset):
        
        #  读取一条概率密度函数
        x_predict = data[[param_name+'_mu',param_name+'_sigma']]
        x_predict_err = data[[param_name+'_sigma']]

        x_predict = x_predict.to_numpy()
        x_predict_err = x_predict_err.to_numpy()


        x_predict_list.append(x_predict)
        x_predict_err_list.append(x_predict_err)

    # 将一个列表拼到一个numpy
    x_predict_Param = np.hstack(x_predict_list)
        

    # 读取模型
    blending_model_path = os.path.join(args.path_log ,f"blending/{args.date_range}/model/{param_name}_model_{args.date_range}.joblib")
    model = joblib.load(blending_model_path)


    # 预测
    y_pred = model.predict(x_predict_Param)
    y_err = np.mean(x_predict_err_list, axis=0)
 
    # 预测结果/预测不确定性
    return y_pred,y_err


def star_one_blending(args,primary_result_list):

    
    para_list = ['Teff[K]', 'Logg', 'RV', 'FeH', 'MgH', 'SiH', 'SH', 'KH', 'CaH', 'TiH', 'CrH', 'NiH', 'CH', 'NH', 'OH',  'AlH', 'MnH','NaH', 'VH',]

    # 按照输出结果一致的格式初始化
    catalog_result = primary_result_list[0][['obsid','ra','dec','snrg']].copy()

    # 逐个参数训练
    for para in para_list:
        pridect_result,err = blending_train(args, dataset=primary_result_list, train_label=para)
        catalog_result.loc[:, para] = pridect_result
        catalog_result.loc[:,para+'_uncertainty'] = err

    # catalog_result.to_csv("./predict_result.csv", index=False)
    return catalog_result

def main(args):
    # args.data_range = '50_999'

    # 遍历文件获得文件名
    data_set_path = args.path_data_set + "/flux/"
    data_list = os.listdir(data_set_path)

    catalog_list = []
    for i in range(len(data_list)):     #len(data_list)
        print("-----------------------",i,"-----------------------------")
        dataset_info = get_dataset_info(args,i)

        primary_result_list=[]

        args.parameter_group = 'none'
        none_df=star_one_predict(args,dataset_info)
        # primary_result_list.append(none_df.add_suffix('_' + str(1)))
        primary_result_list.append(none_df)
        
        args.parameter_group = 'two'
        two_df=star_one_predict(args,dataset_info)
        # primary_result_list.append(none_df.add_suffix('_' + str(1)))
        primary_result_list.append(two_df)

        args.parameter_group = 'each'
        each_df=star_one_predict(args,dataset_info)
        # primary_result_list.append(none_df.add_suffix('_' + str(1)))
        primary_result_list.append(each_df)
        
        catalog_result=star_one_blending(args,primary_result_list)

        catalog_list.append(catalog_result)

    catalog = pd.concat(catalog_list, ignore_index=True)

    if args.path_save.endswith('.csv'):
        # 直接保存到指定路径
        catalog.to_csv(args.path_save, index=False)
    else:
        # 保存到指定路径
        os.makedirs(args.path_save, exist_ok=True)
        catalog.to_csv(os.path.join(args.path_save,"catalog.csv"), index=False)
    


if __name__ == "__main__":

    args = get_args_parser()
    args = args.parse_args()
    setup_seed(args.seed)
    
    args.date_range='5_50'
        
    args.path_data_set = r'4_SpecTE-LAMOST_catalog\1_download_and_preprocessing_LAMOSTDR11\Fits_preprocessed\5-6'
    args.path_save =  r'4_SpecTE-LAMOST_catalog\2_catalog_SpecTE-LAMOST\5-6'
    main(args)

    args.path_data_set = r'4_SpecTE-LAMOST_catalog\1_download_and_preprocessing_LAMOSTDR11\Fits_preprocessed\6-8'
    args.path_save =  r'4_SpecTE-LAMOST_catalog\2_catalog_SpecTE-LAMOST\6-8'
    main(args)

    args.path_data_set = r'4_SpecTE-LAMOST_catalog\1_download_and_preprocessing_LAMOSTDR11\Fits_preprocessed\8-10'
    args.path_save =  r'4_SpecTE-LAMOST_catalog\2_catalog_SpecTE-LAMOST\8-10'
    main(args)

    args.path_data_set = r'4_SpecTE-LAMOST_catalog\1_download_and_preprocessing_LAMOSTDR11\Fits_preprocessed\10-13'
    args.path_save =  r'4_SpecTE-LAMOST_catalog\2_catalog_SpecTE-LAMOST\10-13'
    main(args)

    args.path_data_set = r'4_SpecTE-LAMOST_catalog\1_download_and_preprocessing_LAMOSTDR11\Fits_preprocessed\13-16'
    args.path_save =  r'4_SpecTE-LAMOST_catalog\2_catalog_SpecTE-LAMOST\13-16'
    main(args)

    args.path_data_set = r'4_SpecTE-LAMOST_catalog\1_download_and_preprocessing_LAMOSTDR11\Fits_preprocessed\16-20'
    args.path_save =  r'4_SpecTE-LAMOST_catalog\2_catalog_SpecTE-LAMOST\16-20'
    main(args)

    args.path_data_set = r'4_SpecTE-LAMOST_catalog\1_download_and_preprocessing_LAMOSTDR11\Fits_preprocessed\20-25'
    args.path_save =  r'4_SpecTE-LAMOST_catalog\2_catalog_SpecTE-LAMOST\20-25'
    main(args)

    args.path_data_set = r'4_SpecTE-LAMOST_catalog\1_download_and_preprocessing_LAMOSTDR11\Fits_preprocessed\25-30'
    args.path_save =  r'4_SpecTE-LAMOST_catalog\2_catalog_SpecTE-LAMOST\25-30'
    main(args)

    args.path_data_set = r'4_SpecTE-LAMOST_catalog\1_download_and_preprocessing_LAMOSTDR11\Fits_preprocessed\30-35'
    args.path_save =  r'4_SpecTE-LAMOST_catalog\2_catalog_SpecTE-LAMOST\30-35'
    main(args)

    args.path_data_set = r'4_SpecTE-LAMOST_catalog\1_download_and_preprocessing_LAMOSTDR11\Fits_preprocessed\35-40'
    args.path_save =  r'4_SpecTE-LAMOST_catalog\2_catalog_SpecTE-LAMOST\35-40'
    main(args)

    args.path_data_set = r'4_SpecTE-LAMOST_catalog\1_download_and_preprocessing_LAMOSTDR11\Fits_preprocessed\40-50'
    args.path_save =  r'4_SpecTE-LAMOST_catalog\2_catalog_SpecTE-LAMOST\40-50'
    main(args)


    args.date_range='50_999'
    
    
    args.path_data_set = r'4_SpecTE-LAMOST_catalog\1_download_and_preprocessing_LAMOSTDR11\Fits_preprocessed\50-60'
    args.path_save =  r'4_SpecTE-LAMOST_catalog\2_catalog_SpecTE-LAMOST\50-60'
    main(args)

    args.path_data_set = r'4_SpecTE-LAMOST_catalog\1_download_and_preprocessing_LAMOSTDR11\Fits_preprocessed\60-80'
    args.path_save =  r'4_SpecTE-LAMOST_catalog\2_catalog_SpecTE-LAMOST\60-80'
    main(args)

    args.path_data_set = r'4_SpecTE-LAMOST_catalog\1_download_and_preprocessing_LAMOSTDR11\Fits_preprocessed\80-100'
    args.path_save =  r'4_SpecTE-LAMOST_catalog\2_catalog_SpecTE-LAMOST\80-100'
    main(args)

    args.path_data_set = r'4_SpecTE-LAMOST_catalog\1_download_and_preprocessing_LAMOSTDR11\Fits_preprocessed\100-150'
    args.path_save =  r'4_SpecTE-LAMOST_catalog\2_catalog_SpecTE-LAMOST\100-150'
    main(args)

    args.path_data_set = r'4_SpecTE-LAMOST_catalog\1_download_and_preprocessing_LAMOSTDR11\Fits_preprocessed\150-999'
    args.path_save =  r'4_SpecTE-LAMOST_catalog\2_catalog_SpecTE-LAMOST\150-999'
    main(args)


    

