import os
import numpy as np  
import pandas as pd
import torch    
import torch.nn as nn
import pickle
import torch.utils.data as Data
import argparse
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
from collections import defaultdict
from tqdm import tqdm
import pickle


parser = argparse.ArgumentParser()

parser.add_argument(
    '--net_list', default=["none_MAE(Pa=[230]-Di=[160]-Ha=[16]-De=[8]-mlp=[4.0])_50_999",
                           "two_MAE(Pa=[230]-Di=[160]-Ha=[16]-De=[8]-mlp=[4.0])_50_999",
                           "each_MAE(Pa=[230]-Di=[160]-Ha=[16]-De=[8]-mlp=[4.0])_50_999"
                           ],
    help='''
        model name list(需保证在模型log文件夹内)
        '''
)

parser.add_argument(
    '--model_path', default=None,
    help='''
        可以直接放模型的路径列表（优先级高）
        '''
)


parser.add_argument(
    '--path_label_config', type=str, default='./2_SpecTE/data_processed/fine_tuning_dataset/5_50_stdFlux/label_config.pkl',
    help='存放标签的一些信息，包括数据和标签的均值方差等，用于还原',
)

# parser.add_argument(
#     '--path_labels', type=str, default='./data/LABELS',
#     help='The path of the labels.',
# )

# 存放模型数据的路径
parser.add_argument(
    '--path_log', type=str, default= './2_SpecTE/model_log/fine_tune/',
    help='The path to save the model data after training.'
)

# # 存放原始数据 （暂时没用到）
# parser.add_argument(
#     '--path_preprocessed', type=str, default='./data/data_Preprocessed',   # D:\文\jupyter\My\data\data_Preprocessed  .data/data_Preprocessed
#     help='The path to save the model data after training.'
# )

# 保存位置
parser.add_argument(
    '--path_save', type=str, default='./2_SpecTE/model_log/blending/',   
    help='The path to save the model data after training.'
)


parser.add_argument(
    '--label_list', type=list, default=['Teff[K]', 'Logg', 'RV', 'CH', 'NH', 'OH', 'NaH', 'MgH', 'AlH', 'SiH', 'SH', 
                                 'KH', 'CaH', 'TiH',  'VH', 'CrH','MnH', 'FeH', 'NiH', 'snrg',],  
    
    # ['Teff[K]', 'Logg', 'CH', 'NH', 'OH', 'MgH', 'AlH', 'SiH', 'SH',
    #                                       'KH', 'CaH', 'TiH', 'CrH','MnH', 'FeH', 'NiH', 'snrg'],   
    help='The label data that needs to be learned.'
)
parser.add_argument(
    '--date_range', choices=['5_50', '50_999', 'all'], default='50_999',
    help='选择数据集信噪比范围.',
)


parser.add_argument(
    '--mode_train',  choices=['linear', 'FC'], default='linear',
    help ='The size of the batch.'
)




# parser.add_argument(
#     '--std', type=bool, default=False,
#     help='''
#         loss: 可选：'PDPL', 'MSE','MAE','SoothL1Loss'
#         '''
# )





def get_dataset_info(args):


    # 读预测结果
    valid = None
    test = None
    # 读取多种网络的结果
    for i in range(len(args.net_list)):
        model_path = args.path_log + args.net_list[i]
        if args.model_path:
            model_path = args.model_path[i]
        if os.path.exists(model_path):
            dir_list = os.listdir(model_path)
            files_to_check = ['predict_result.csv', 'predict_valid.csv']
            # 寻找其中的文件
            if all(file in dir_list for file in files_to_check):
                predict_test = pd.read_csv(os.path.join(model_path, 'predict_result.csv'))
                predict_valid = pd.read_csv(os.path.join(model_path, 'predict_valid.csv'))
                predict_valid = predict_valid.add_suffix('_' + str(i))
                predict_test = predict_test.add_suffix('_' + str(i))
            else:
                raise Exception("没有找到预测结果文件。")    
        else:
            raise Exception("没有找到模型文件。")
        
        if valid is None:
            # x_valid = predict_valid.to_xarray()
            # x_test = predict_test.to_xarray()
            valid = predict_valid
            test = predict_test
        else:
            # x_valid = xr.merge([x_valid, predict_valid.to_xarray()])
            # x_test = xr.merge([x_test, predict_test.to_xarray()])
            valid = pd.concat([valid, predict_valid], axis=1)
            test = pd.concat([test, predict_test], axis=1)
    

    label_config = pickle.load(open(args.path_label_config, 'rb'))
    mean_label = label_config["label_mean"]
    std_label = label_config["label_std"]
    
    # x_valid_torch = torch.tensor(valid.values, dtype=torch.float32)
    # X_test_torch = torch.tensor(valid.values, dtype=torch.float32)


    # # # train_dataset = Data.TensorDataset(X_train_torch, y_train_torch)
    # valid_dataset = Data.TensorDataset(x_valid_torch, y_valid_torch)
    # test_dataset = Data.TensorDataset(X_test_torch, y_test_torch)

    # valid_loader = Data.DataLoader(
    #     dataset=valid_dataset,
    #     batch_size=args.train_parameters['batch'],
    #     shuffle=True,  
    #     num_workers=0,
    # )
    # test_loader = Data.DataLoader(
    #     dataset=test_dataset,
    #     batch_size=args.train_parameters['batch'],
    #     shuffle=False,
    #     num_workers=0,
    # )
    # mean_label = mean_label[args.label_list].values
    # std_label = std_label[args.label_list].values

    # dataset_info = {
    #     "y_valid": y_valid,
    #     "y_test": y_test,
    #     "x_valid": x_valid,
    #     "x_test": x_test,
    #     "valid_loader": valid_loader,
    #     "test_loader": test_loader
    # }

    dataset_info = {
        "valid": valid,
        "test": test,
        "mean_label":mean_label,
        "std_label":std_label
    }

    return dataset_info


def train(args, dataset_info, train_label):


    # 加载数据集

    valid = dataset_info["valid"]
    test = dataset_info["test"]

    # 训练多元线性回归模型
    for i, param_name in enumerate(train_label):

        # 读取当前参数param的真实值   
        y_valid_Param = valid[param_name+'_'+str(1)]
        y_test_Param = test[param_name+'_'+str(1)]

        y_valid_Param=y_valid_Param.to_numpy()
        y_test_Param=y_test_Param.to_numpy()


        x_valid_list =[]
        x_test_list = []
        text_err_list = []

        for j in range(len(args.net_list)):
            
            #  读取一条概率密度函数
                
            x_valid_P = valid[[param_name+'_mu_'+str(j),param_name+'_sigma_'+str(j)]]
            x_test_P = test[[param_name+'_mu_'+str(j),param_name+'_sigma_'+str(j)]]
            x_test_err = test[[param_name+'_sigma_'+str(j)]]
            
            # a="不要sigma"
            # if a=="不要sigma":
            #     x_valid_P = valid[[param_name+'_mu_'+str(j),]]
            #     x_test_P = test[[param_name+'_mu_'+str(j),]]
            # 只读取估计结果
            # x_valid_P = valid[param_name+'_mu_'+str(j)]
            # x_test_P = test[param_name+'_mu_'+str(j)]

            x_valid_P = x_valid_P.to_numpy()
            x_test_P = x_test_P.to_numpy()
            x_test_err = x_test_err.to_numpy()


            x_valid_list.append(x_valid_P)
            x_test_list.append(x_test_P)
            text_err_list.append(x_test_err)
            # if x_valid_Param is None:
            #     x_valid_Param = x_valid_P
            #     x_test_Param = x_test_P
            # else:
            #     x_valid_Param = np.hstack((x_valid_Param, x_valid_P))
            #     x_test_Param = np.hstack((x_test_Param, x_valid_P))
        x_valid_Param = np.hstack(x_valid_list)
        x_test_Param = np.hstack(x_test_list)
        # x_valid_Param = np.stack(x_valid_list, axis=1)
        # x_test_Param = np.stack(x_test_list, axis=1)

            # 用多元线性回归估计
        if args.mode_train == "linear":
        
            # model = RandomForestRegressor(n_estimators=100, random_state=42)
            model = LinearRegression()

            # model.fit(x_test_Param, y_test_Param)
            model.fit(x_valid_Param, y_valid_Param)


            # 预测
            y_pred = model.predict(x_test_Param)
            y_err = np.mean(text_err_list, axis=0)

        elif args.mode_train == "directMethod":
            # 从每个数据集中提取列并转换为 NumPy 数组
            data_sigma = np.array([test[param_name +'_sigma_'+ str(index)] for index in range(len(args.net_list))])
            
            # 计算每个索引处的最小值所在的数据集索引
            min_indices = np.argmin(data_sigma, axis=0)
            
            y_pred = np.array([test[param_name +'_mu_'+ str(index)][i] for i, index in enumerate(min_indices)])



        diff_std = (y_test_Param - y_pred).std(axis=0)
        diff_mean = (y_test_Param - y_pred).mean(axis=0)

        mae = mean_absolute_error(y_test_Param, y_pred, multioutput='raw_values')


        print(param_name," MAE:",mae)
        print(param_name," σ:",diff_std)
        print(param_name," μ:",diff_mean)


        # 保存模型
        model_save_path=os.path.join(args.path_save,"model")
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        blending_model_path = os.path.join(model_save_path,f"{param_name}_model_{args.date_range}.joblib") 

        
        joblib.dump(model, blending_model_path)

        # # 保存测试集预测结果
        # blending_result_path = f"../data/Bi-GRU-Attention_predictions/between_5_50/Blending/{param_name}_prediction.npy"
        # np.save(blending_result_path, y_pred_blending_Param)
 
    # 真实参数/预测结果
    return y_test_Param, y_pred,y_err


def blending(args):
    
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

    all_result = defaultdict(lambda: [[], []])
    # all_result = defaultdict(lambda: [[], [], []])
    a_list = []
    b_list = []
    #读数据
    dataset_info = get_dataset_info(args)
    

    # 按照输出结果一致的格式初始化
    catalog_result = dataset_info['test'][['snrg_0']].copy().rename(columns={'snrg_0': 'snrg'}) 
  

    # 逐个参数训练
    for key, value in para_dict.items():
        a,b,err = train(args, dataset_info=dataset_info, train_label=value)
        all_result[key][0] = a
        all_result[key][1] = b
        # all_result[key][2] = err

        # 保存为星表的格式
        catalog_result.loc[:, value[0]] = a
        catalog_result.loc[:,value[0]+'_predict'] = b
        catalog_result.loc[:,value[0]+'_err'] = err

        std=dataset_info["std_label"][value].values
        mean=dataset_info["mean_label"][value].values
        a = (a-mean)/ std
        b = (b-mean)/ std
        a_list.append(a)
        b_list.append(b)
        
    
    y_test_snr = dataset_info["test"]['snrg_1']
    all_result['snrg'][0] = np.concatenate((all_result['snrg'][0], y_test_snr), axis=0)


    
    print("result")
    
    # # 5-50
    # print("5-50")
    # args.net_list=["BGANet_(GRU=[64-32]-Ns=5-Hd=1)_dr=0.2_sfdr=0.25_5_50_PDPL——分AB训练",
    #                "BGANet_(GRU=[128-64-32]-Ns=10-Hd=1)_dr=0.2_sfdr=0.25_5_50_PDPL——分AB训练",
    #                "BGANet_(GRU=[128-64-32]-Ns=15-Hd=1)_dr=0.2_sfdr=0.25_5_50_PDPL——分AB训练"]
    # args.date_range = '5_50'
    # dataset_info = get_dataset_info(args)
    # for key, value in para_dict.items():
    #     a,b = train(args, dataset_info=dataset_info, train_label=value)
    #     all_result[key][0] = a
    #     all_result[key][1] = b 

    # y_test_snr = dataset_info["y_valid"]['snrg']
    # all_result['snrg'][0] = y_test_snr


    # # 50-999
    # print("50-999")
    # args.net_list=["BGANet_(GRU=[64-32]-Ns=5-Hd=1)_dr=0.2_sfdr=0.25_50_999_PDPL——分AB训练",
    #                "BGANet_(GRU=[128-64-32]-Ns=10-Hd=1)_dr=0.2_sfdr=0.25_50_999_PDPL——分AB训练",
    #                "BGANet_(GRU=[128-64-32]-Ns=15-Hd=1)_dr=0.2_sfdr=0.25_50_999_PDPL——分AB训练"]
    # args.date_range = '50_999'
    # dataset_info = get_dataset_info(args)
    # for key, value in para_dict.items():
    #     a,b = train(args, dataset_info=dataset_info, train_label=value)
        
    #     all_result[key][0]=np.concatenate((all_result[key][0], a), axis=0)
    #     all_result[key][1]=np.concatenate((all_result[key][1], b), axis=0)
    # y_test_snr = dataset_info["y_valid"]['snrg']
    # all_result['snrg'][0] = np.concatenate((all_result['snrg'][0], y_test_snr), axis=0)



    # 统计   
    df = pd.DataFrame()
    
    for key, value in para_dict.items():  

        diff_std = (all_result[key][0] - all_result[key][1]).std(axis=0)
        diff_mean = (all_result[key][0] - all_result[key][1]).mean(axis=0)

        mae = mean_absolute_error(all_result[key][0], all_result[key][1], multioutput='raw_values')
        mae = mae[0] if len(mae) == 1 else np.mean(mae)

        # print(key," MAE:",mae)
        # print(key," σ:",diff_std)
        # print(key," μ:",diff_mean)
        df[str(value)] = pd.Series({'μ': diff_mean, 'σ': diff_std, 'MAE': mae})    

    df = df.T
    print(df)


    # 直接保存统计结果
    df.to_csv(args.path_save+'统计结果_{}.csv'.format(args.date_range), index=True)    
    catalog_result.to_csv(os.path.join(args.path_save,"catalog{}.csv".format(args.date_range)), index=False)

    # 保存原始结果
    # 转换为普通字典
    simple_dict = {k: v for k, v in all_result.items()}
    with open(args.path_save+'原始结果_{}.pkl'.format(args.date_range), 'wb') as file:
        pickle.dump(simple_dict, file)


    # 计算归一化后的mae，做为消融实验的评价指标
    a_array = np.array(a_list)
    b_array = np.array(b_list)
    all_a = a_array.T
    all_b = b_array.T
    mae = mean_absolute_error(all_a, all_b, multioutput='raw_values')
    mae=mae.mean()
    return mae


if __name__ == "__main__":
    args = parser.parse_args()
    args.model_path=[r"G:\Star\model_test\all\1_finetune_lr\0_lr=[0.001]-drop=[0.0]\fine_tune\each_SpecTE(Pa=[230]-Di=[160]-Ha=[16]-De=[8]-mlp=[4.0])_5_50_stdFlux",
                     r"G:\Star\model_test\all\1_finetune_lr\0_lr=[0.001]-drop=[0.0]\fine_tune\none_SpecTE(Pa=[230]-Di=[160]-Ha=[16]-De=[8]-mlp=[4.0])_5_50_stdFlux",
                     r"G:\Star\2_SpecTE\model_log\fine_tuning\two_SpecTE(Pa=[230]-Di=[160]-Ha=[16]-De=[8]-mlp=[4.0])_5_50_stdFlux",]

    print (blending(args))
    

    
    