import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_and_save_loss_curves(train_loss_list, valid_loss_list, file_path, model_name, lr_list=None):
    """
    绘制训练和验证损失曲线，并在图表上标注验证损失的最小值。
    如果提供学习率列表，则在同一图表的右侧Y轴上绘制学习率曲线。
    将图表保存为PNG文件，并将损失数据保存为CSV文件。
    
    参数:
    - train_loss_list: 训练损失的列表。
    - valid_loss_list: 验证损失的列表。
    - lr_list: 学习率的列表。
    - file_path: 图表和CSV文件保存的路径。
    - model_name: 作为图表的副标题的模型名称。
    """
    png_file_path = os.path.join(file_path, 'training_validation_loss_curve.png')
    csv_file_path = os.path.join(file_path, 'loss_data.csv')
    
    min_valid_loss = min(valid_loss_list)
    min_loss_index = valid_loss_list.index(min_valid_loss)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(train_loss_list, label='Training Loss', color='blue')
    ax1.plot(valid_loss_list, label='Validation Loss', color='red')
    
    ax1.scatter(min_loss_index, min_valid_loss, color='green', label='Minimum Validation Loss')
    ax1.text(min_loss_index, min_valid_loss, f'{min_valid_loss:.5f}', color='green', verticalalignment='bottom')
    
    ax1.set_title('Training and Validation Loss\n' + model_name)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    if lr_list is not None:
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.plot(lr_list, label='Learning Rate', color='orange', linestyle='--')
        ax2.set_ylabel('Learning Rate', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        ax2.legend(loc='upper right')

    ax1.legend(loc='upper left')

    plt.savefig(png_file_path)
    
    loss_data = {
        'Training Loss': train_loss_list,
        'Validation Loss': valid_loss_list
    }
    
    if lr_list is not None:
        loss_data['Learning Rate'] = lr_list
    
    df_loss_data = pd.DataFrame(loss_data)
    df_loss_data.to_csv(csv_file_path, index_label='Epoch')




# def plot_and_save_loss_curves(train_loss_list, valid_loss_list, file_path, model_name):
#     """
#     绘制训练和验证损失曲线，并在图表上标注验证损失的最小值。
#     将图表保存为PNG文件，并将损失数据保存为CSV文件。
    
#     参数:
#     - train_loss_list: 训练损失的列表。
#     - valid_loss_list: 验证损失的列表。
#     - file_path: 图表和CSV文件保存的路径。
#     - model_name: 作为图表的副标题的模型名称。
#     """

#     png_file_path = os.path.join(file_path, 'training_validation_loss_curve.png')
#     csv_file_path = os.path.join(file_path, 'loss_data.csv')
    
#     # 找出验证损失的最小值及其索引
#     min_valid_loss = min(valid_loss_list)
#     min_loss_index = valid_loss_list.index(min_valid_loss)

#     # 创建一个绘图
#     plt.figure(figsize=(10, 6))

#     # 绘制训练损失和验证损失曲线
#     plt.plot(train_loss_list, label='Training Loss', color='blue')
#     plt.plot(valid_loss_list, label='Validation Loss', color='red')
    
#     # 在最小验证损失点标注
#     plt.scatter(min_loss_index, min_valid_loss, color='green', label='Minimum Validation Loss')
#     plt.text(min_loss_index, min_valid_loss, f'{min_valid_loss:.5f}', color='green', verticalalignment='bottom')

#     # 添加图例
#     plt.legend()

#     # 添加标题、副标题和轴标签
#     plt.title('Training and Validation Loss\n' + model_name)
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')

#     # 添加网格
#     plt.grid(True)

#     # 保存图表
#     plt.savefig(png_file_path)

#     # 可选：显示图表
#     # plt.show()

#     # 创建DataFrame并保存为CSV文件
#     loss_data = pd.DataFrame({
#         'Training Loss': train_loss_list,
#         'Validation Loss': valid_loss_list
#     })
#     loss_data.to_csv(csv_file_path, index_label='Epoch')
