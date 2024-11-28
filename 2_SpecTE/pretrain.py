from matplotlib.pyplot import axis
import torch
import pandas as pd
import numpy as np
import os
import torch.utils.data as Data
import argparse

from tqdm import tqdm
import pickle


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.backends.cudnn as cudnn

from torch.utils.tensorboard import SummaryWriter

import timm.optim.optim_factory as optim_factory

from util.util import *
import util.lr_sched as lr_sched
from util.loss_curves import plot_and_save_loss_curves


from models.SpecTE_pretrain import SpecTE
# from models.vit1D import VisionTransformer


def get_args_parser():
    # 定义参数
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=256,
                        help ='The size of the batch.256')
    
    parser.add_argument('--n_epochs', type=int, default=30,#30  
                        help='Number of epochs to train.')

    parser.add_argument('--save_predict', type=bool, default=False,
                        help='Whether to save the prediction results.')

    #==========================Model parameters模型参数==============================
    parser.add_argument('--net', choices=['SpecTE'], default='SpecTE',
                        help='The models you need to use.',)
    
    parser.add_argument('--flux_size', default=3450, type=int,
                        help='images input size')

    # Hyperparameters_MAE
    # Hyperparameters_SpecTE
    parser.add_argument('--Hyperparameters_SpecTE', 
                        default={'patch_size':230, # 将输入图像分割成补丁的大小。  230
                                 'embed_dim':160, # 嵌入维度
                                 'depth':8, #Encoder的层数        8
                                 'num_heads':16, # 编码器注意力头的数量
                                 'decoder_embed_dim':80, # 解码器的嵌入维度
                                 'decoder_depth':1, # 解码器的层数
                                 'decoder_num_heads':16, # 解码器的注意力头数量
                                 'mlp_ratio':4., # MLP中隐层与嵌入维度的比例
                                 'drop_rate':0.0, # 随机丢弃的概率
                                 },
                        help='''SpecTE的参数''')

    #=============================Optimizer parameters优化器参数==================================
    parser.add_argument('--weight_decay', type=float, default=0.40,     #  0.4
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=0.005, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0.000005, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=4, metavar='N',    #2
                        help='epochs to warmup LR')

    #======================= =Dataset parameters数据选择和数据处理==================================
    parser.add_argument('--date_range', choices=['0_10', '10_20', '20_30','30_40','40_50','all'], default='all',
                        help='选择数据集信噪比范围.',)
    
    parser.add_argument('--noise_model', type=list, default=[True,0.05],    # add_training_noise
                        help='Train: Whether to add Gaussian noise with a mean of 1 and a variance of 1 during training    Predict: Whether to use a model trained with noise ')
    
    parser.add_argument('--Flux_std', type=bool, default=False,
                        help='Whether to 标准化流量数据.  （测试没用）',)
    
    parser.add_argument('--cuda', default=True,
                        help='device to use for training / testing')
    
    parser.add_argument('--seed', default=1, type=int)

     #===============================定义路径=======================================
    # parser.add_argument('--path_data_set', type=str, default = r'E:/my_star/预处理/data_after_processing/result',
    #                     help='数据所在位置',)
    parser.add_argument('--path_data_set', type=str, default= r'./1_Data_download_and_preprocessing/Denoising_reference_set/data_after_processing/result',   # ./optuna_log/
                        help='The path of the data after preprocessed.')
    
    parser.add_argument('--path_log', type=str, default= './2_SpecTE/model_log/pretrain/',   # ./optuna_log/
                        help='The path to save the model and training logs after training.')
    
    parser.add_argument('--path_data_processed', type=str, default='./2_SpecTE/data_processed/pretrain_dataset', 
                        help='The path to save training data after processing.')

    return parser


def get_dataset_info(args):
    
    setup_seed(args.seed)
    # 定义中间数据保存路径
    data_set_temp_path = args.path_data_processed + "/{}{}".format(args.date_range,'_stdFlux' if args.Flux_std else '')
    if not os.path.exists(data_set_temp_path):
        os.makedirs(data_set_temp_path)
    dir_list = os.listdir(data_set_temp_path)
    print(dir_list)
    target_list=['label_config.pkl', 'train_flux.npy', 'train_label.npy', 'valid_flux.npy', 'valid_label.npy']

    #普通处理
    if dir_list != target_list and args.date_range != 'all':


        path_flux_low = os.path.join(args.path_data_set, "{}/flux_low_reordered.npy".format(args.date_range))
        path_flux_high = os.path.join(args.path_data_set, "{}/flux_high_reordered.npy".format(args.date_range))
        path_match_snrg = os.path.join(args.path_data_set, "{}/match_updated.csv".format(args.date_range))
        print(args.path_data_set)

        # 加载数据
        flux_low = np.load(path_flux_low)
        flux_high = np.load(path_flux_high)
        match_snrg=pd.read_csv(path_match_snrg)

        columns = ['obsid_1', 'snrg_1', 'obsid_2', 'snrg_2']
        match_snrg = match_snrg[columns]

        # 去除存在无穷值的数据
        rows_with_nan_or_inf_low = np.isnan(flux_low).any(axis=1) | np.isinf(flux_low).any(axis=1)
        rows_with_nan_or_inf_high = np.isnan(flux_high).any(axis=1) | np.isinf(flux_high).any(axis=1)
        rows_to_delete = rows_with_nan_or_inf_low | rows_with_nan_or_inf_high

        indices_of_rows_with_nan = np.where(rows_to_delete)[0]

        flux_low = np.delete(flux_low, indices_of_rows_with_nan, axis=0)
        flux_high = np.delete(flux_high, indices_of_rows_with_nan, axis=0)
        match_snrg = match_snrg.drop(indices_of_rows_with_nan).reset_index(drop=True)

        # 流量标准化
        if args.Flux_std:
            # std_flux = np.std(flux, axis=0)
            # mean_flux = np.mean(flux, axis=0)
            # flux = (flux-mean_flux)/std_flux

            # 低信噪比
            Flux_3sigma_sc_low=StandardScaler()
            std_flux_T = Flux_3sigma_sc_low.fit_transform(flux_low.T)   # 对每条光谱数据进行标准化    
            flux_low = std_flux_T.T
            # 高信噪比
            Flux_3sigma_sc_high=StandardScaler()
            std_flux_T = Flux_3sigma_sc_high.fit_transform(flux_high.T)
            flux_high = std_flux_T.T

        
        # 划分数据集
        x_train, x_valid, y_train, y_valid,match_train, match_valid= train_test_split(flux_low, flux_high, match_snrg,test_size=0.2, random_state=args.seed)
        # x_valid, x_test, y_valid, y_test = train_test_split(x_valid_test, y_valid_test, test_size=0.67, random_state=args.seed)

        print('size:', x_train.shape, y_train.shape,)
        print('Training set:', x_train.shape[0], 'samples', 'flux type:', type(x_train), 'Training labels:', type(y_train))
        print('Validation set:', x_valid.shape[0], 'samples')



        # 保存
        label_config = {
            "match_train": match_train,
            "match_valid": match_valid,
        }
        
        with open(os.path.join(data_set_temp_path, 'label_config.pkl'), 'wb') as f:
            pickle.dump(label_config, f)
        np.save(os.path.join(data_set_temp_path, 'train_flux.npy'), x_train)
        np.save(os.path.join(data_set_temp_path, 'valid_flux.npy'), x_valid)
        np.save(os.path.join(data_set_temp_path, 'train_label.npy'), y_train)
        np.save(os.path.join(data_set_temp_path, 'valid_label.npy'), y_valid)

    # 处理all时从已经处理好的数据中读取可以保证测试集验证集一致   
    elif args.date_range == 'all' and dir_list != target_list :

        match_train_list=[]
        match_valid_list=[]
        x_train_list=[]
        x_valid_list=[]
        y_train_list=[]
        y_valid_list=[]

        for i in range(5):
            sub_data_set_temp_path = args.path_data_processed + "/{}{}".format("{}_{}".format(i*10,i*10+10),'_stdFlux' if args.Flux_std else '')
            if not os.path.exists(sub_data_set_temp_path):
                os.makedirs(sub_data_set_temp_path)
            sub_dir_list = os.listdir(sub_data_set_temp_path)

            if sub_dir_list == target_list:
                sub_label_config = pickle.load(open(sub_data_set_temp_path + "/label_config.pkl", 'rb'))
                # sub_match_snrg = sub_label_config["match_snrg"]
                match_train_list.append(sub_label_config["match_train"])
                match_valid_list.append(sub_label_config["match_valid"])    

                sub_x_train = np.load(sub_data_set_temp_path + "/train_flux.npy")
                sub_x_valid = np.load(sub_data_set_temp_path + "/valid_flux.npy")
                
                sub_y_train = np.load(sub_data_set_temp_path + "/train_label.npy")
                sub_y_valid = np.load(sub_data_set_temp_path + "/valid_label.npy")

                # match_snrg_list.append(sub_match_snrg)
                x_train_list.append(sub_x_train)  
                x_valid_list.append(sub_x_valid)
                y_train_list.append(sub_y_train)
                y_valid_list.append(sub_y_valid)

            else:
                raise Exception("There was an error loading data. Please check the files in the {} path. Due to the default consistency between the training set validation set of all and all other subsets in the program, only subsets are allowed to be read when processing all data. Please ensure that each subset has been processed properly.".format(sub_data_set_temp_path))
            
        match_train = pd.concat(match_train_list, axis=0, ignore_index=True)
        match_valid = pd.concat(match_valid_list, axis=0, ignore_index=True)
            
        del match_train_list,match_valid_list

        x_valid = np.concatenate(x_valid_list, axis=0)
        y_valid = np.concatenate(y_valid_list, axis=0)
        
        del x_valid_list,y_valid_list

        x_train = np.concatenate(x_train_list, axis=0)
        
        del x_train_list
        y_train = np.concatenate(y_train_list, axis=0)

        del y_train_list
                
        label_config = {
            "match_train": match_train,
            "match_valid": match_valid,
        }

        # with open(os.path.join(data_set_temp_path, 'label_config.pkl'), 'wb') as f:
        #     pickle.dump(label_config, f)

        # np.save(os.path.join(data_set_temp_path, 'train_flux.npy'), x_train)
        # np.save(os.path.join(data_set_temp_path, 'valid_flux.npy'), x_valid)
        # np.save(os.path.join(data_set_temp_path, 'train_label.npy'), y_train)
        # np.save(os.path.join(data_set_temp_path, 'valid_label.npy'), y_valid)


    # 读取已经处理的数据
    elif dir_list == ['label_config.pkl', 'train_flux.npy', 'train_label.npy', 'valid_flux.npy', 'valid_label.npy']:
        label_config = pickle.load(open(data_set_temp_path + "/label_config.pkl", 'rb'))
        match_train = label_config["match_train"]
        match_valid = label_config["match_valid"]


        # x_train = pickle.load(open(data_set_temp_path + "/train_flux.pkl", 'rb'))
        # x_valid = pickle.load(open(data_set_temp_path + "/valid_flux.pkl", 'rb'))
        x_train = np.load(data_set_temp_path + "/train_flux.npy")
        x_valid = np.load(data_set_temp_path + "/valid_flux.npy")
        y_train = np.load(data_set_temp_path + "/train_label.npy")
        y_valid = np.load(data_set_temp_path + "/valid_label.npy")

        # y_train = pickle.load(open(data_set_temp_path + "/train_label.pkl", 'rb'))
        # y_valid = pickle.load(open(data_set_temp_path + "/valid_label.pkl", 'rb'))
 

        # y_train = pd.read_csv(os.path.join(data_set_temp_path, 'train_label.csv'))
        # y_valid = pd.read_csv(os.path.join(data_set_temp_path, 'valid_label.csv'))
        # y_test = pd.read_csv(os.path.join(data_set_temp_path, 'test_label.csv'))
        del label_config           
    else :
        raise Exception("检查{}下的内容。".format(data_set_temp_path))
    
    X_train_torch = torch.tensor(x_train, dtype=torch.float32)
    X_valid_torch = torch.tensor(x_valid, dtype=torch.float32)


    y_train_torch =  torch.tensor(y_train, dtype=torch.float32)
    y_valid_torch = torch.tensor(y_valid, dtype=torch.float32)



    print("x_train_torch.shape:", X_train_torch.shape)
    print("y_train_torch.shape:", y_train_torch.shape)
    print("x_valid_torch:\n", X_valid_torch.shape)
    print("y_valid_torch:\n", y_valid_torch.shape)


    # print("x_train_torch:\n", X_train_torch)
    # print("y_train_torch:\n", y_train_torch)

    train_dataset = Data.TensorDataset(X_train_torch, y_train_torch)
    valid_dataset = Data.TensorDataset(X_valid_torch, y_valid_torch)


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


    dataset_info = {
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        # "match_snrg": match_snrg
    }

    return dataset_info


def train(args, dataset_info, model_number=None):

    setup_seed(args.seed)
    if torch.cuda.is_available()and args.cuda:
        cuda = True
    else:
        cuda = False
    cudnn.benchmark = True # 设置cuDNN在程序开始时自动寻找最适合当前配置的高效算法，以加速固定大小的输入数据的计算。

    # 定义模型和对应的模型名
    if args.net == "SpecTE":
        Hyperparameters = args.Hyperparameters_SpecTE
        patch_size=Hyperparameters['patch_size']
        embed_dim=Hyperparameters['embed_dim']
        depth=Hyperparameters['depth']
        num_heads=Hyperparameters['num_heads']      
        decoder_embed_dim=Hyperparameters['decoder_embed_dim']
        decoder_depth=Hyperparameters['decoder_depth']
        decoder_num_heads=Hyperparameters['decoder_num_heads']
        mlp_ratio=Hyperparameters['mlp_ratio']
        # norm_layer=Hyperparameters['norm_layer']
        # norm_pix_loss=Hyperparameters['norm_pix_loss']
        drop_rate=Hyperparameters['drop_rate']



        # eg: SpecTE(Nr=[16-32-64]-Ns=3)_post-RNN
        model_name = "SpecTE(Pa=[{}]-Di=[{}]-Ha=[{}]-De=[{}])".format(patch_size, embed_dim, num_heads,depth)


        # print("model name : ", model_name)
        net = SpecTE(inpute_size=args.flux_size, 
                             patch_size=patch_size,
                             embed_dim=embed_dim, 
                             depth=depth, 
                             num_heads=num_heads,
                             decoder_embed_dim=decoder_embed_dim,
                             decoder_depth=decoder_depth,
                             decoder_num_heads=decoder_num_heads,
                             mlp_ratio=mlp_ratio,
                            #  norm_layer=norm_layer,
                            #  norm_pix_loss=norm_pix_loss,
                             drop_rate=drop_rate)   
        if cuda:
            net = net.to("cuda")
    else:
        raise Exception("模型名字错误，程序已终止。")
    model_name += "{}{}".format('_'+args.date_range if args.date_range != 'all' else '' ,'_stdFlux' if args.Flux_std else '')
    # if args.noise_model[0]:
    #     model_name += "_add-noise{:.2f}".format(args.noise_model[1])    
    # model_name += "_w-d={:.4f}".format(args.weight_decay)
        
    print("Model = %s" % str(net))
    print("Modelname:", model_name)
    

    # 定义优化器 和学习器

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * args.batch_size / 256 
           #目的是在不同大小的批处理时保持学习率的相对稳定，以获得更好的训练效果

    # param_groups = optim_factory.param_groups_weight_decay(net, args.weight_decay)
    param_groups = optim_factory.param_groups_weight_decay(net, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    # optimizer = optim_factory.create_optimizer_v2(model.parameters(), **optimizer_cfg)
    print(optimizer)
#     optimizer = torch.optim.Adam(net.parameters(), lr=0.001)  # 优化器选择Adam
#  # 用学习率调整器动态调整优化器的学习率
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
    
  
    # 定义Tensorboard路径
    if model_number:
        log_dir = args.path_log + model_name + '/' + model_number  # 日志
    else:
        log_dir = args.path_log + model_name
    writer = SummaryWriter(log_dir=log_dir)
        
    # 定义训练中的参数    
    best_loss = np.inf


    train_loss_list=[]
    valid_loss_list=[]
    lr_list=[]

    # 开始训练
    for epoch in tqdm(range(1, args.n_epochs + 1)):
        
        net.train() # 训练模式
        torch.cuda.empty_cache()    # 清空缓存

        train_loss = 0.0    # 记录模型训练过程中的累计损失（所有批次数据的损失之和）
        
        # Train
        for step, (batch_x, batch_y) in enumerate(dataset_info["train_loader"]):

            if args.noise_model[0] and epoch > 5:
                batch_x = add_noise(batch_x,args.noise_model[1])

            if cuda:
                batch_x = batch_x.to("cuda")
                batch_y = batch_y.to("cuda")

            loss, _ = net(batch_x,batch_y)


            train_loss += loss.to("cpu").data.numpy()

            optimizer.zero_grad()       # 清空梯度
            loss.backward()             # 反向传播，计算梯度
            optimizer.step()            # 更新梯度

            n_iter = (epoch - 1) * len(dataset_info["train_loader"]) + step + 1  # 局迭代步数n_iter，用于记录 Tensorboard 日志及其他训练过程中的信息。

            # pred_flux = pred.to("cpu").data.numpy()                              # 反归一化处理 得到真实的预测值

            lr_sched.adjust_learning_rate(optimizer, step / len(dataset_info["train_loader"]) + epoch, args)
            writer.add_scalar('Train/loss', loss.to("cpu").data.numpy(), n_iter)
            # for i in range(len(train_label)):
            #     writer.add_scalar('Train/%s_MAE' % train_label[i], mae[i], n_iter)

        # scheduler.step()    # 更新优化器的学习率
        lr = optimizer.state_dict()['param_groups'][0]['lr']   # 返回优化器的状态字典
        train_loss /= (step + 1)   # train_loss/N 平均训练损失值
        train_loss_list.append(train_loss)

        torch.cuda.empty_cache()   # 清空缓存
        net.eval()         # 将网络设置为评估模式


        # 定义一些验证集要记录和保存的参数
        valid_loss = 0.0#保存验证集的结果
        # 定义一个和loader中x一样大小的np
        pred_flux_list = [] 


        # Valid
        for step, (batch_x, batch_y) in enumerate(dataset_info["valid_loader"]):

            with torch.no_grad():  # 上下文管理器:代码块中所有的计算都不会计算梯度
                if cuda:
                    batch_x = batch_x.to("cuda")
                    batch_y = batch_y.to("cuda")

                loss, pred = net(batch_x,batch_y)
                
                valid_loss += loss.to("cpu").data.numpy()

                n_iter = (epoch - 1) * len(dataset_info["valid_loader"]) + step + 1


                pred = pred.to("cpu").data.numpy()

                pred_flux_list.append(pred)  # 将预测结果添加到列表中

                

            writer.add_scalar('Valid/loss', loss.to("cpu").data.numpy(), n_iter)
            # print(step?)

        pred_flux = np.concatenate(pred_flux_list, axis=0)
        valid_loss /= (step+1)
        valid_loss_list.append(valid_loss)
        lr_list.append(lr)

        torch.save(net.state_dict(), log_dir + '/weight_temp.pkl')

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(net.state_dict(), log_dir + '/weight_best.pkl')
            pred_flux_best = pred_flux
            

        print("EPOCH %d | lr %f | train_loss %.4f | valid_loss %.4f" % (epoch, lr, train_loss, valid_loss))


    # 保存模型 
    file_path = os.path.join(log_dir, "predict_valid.npy")
    # 保存预测结果
    if args.save_predict:
        np.save(file_path, pred_flux_best)
    # 画训练过程图
    plot_and_save_loss_curves(train_loss_list, valid_loss_list, log_dir, model_name, lr_list=lr_list)

    return best_loss, log_dir + '/weight_best.pkl'



if __name__ == "__main__":

    args = get_args_parser()
    args = args.parse_args()

    setup_seed(args.seed)
    dataset_info = get_dataset_info(args)

    train(args, dataset_info=dataset_info,model_number= "SP1")
