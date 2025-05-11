#!/usr/bin/env python
# coding: utf-8

# In[165]:
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import time
import os
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# matplotlib.use('Agg')
matplotlib.use('TkAgg')

# ## General setting

# In[166]:
# For reproducibility
np.random.seed(0)
# For reproducibility
torch.manual_seed(0)
torch.cuda.empty_cache()
torch.set_default_dtype(torch.float32)
torch.backends.cudnn.deterministic = True


# # ## Implementation of early stopping (saving only) in PyTorch
# # In[167]:
# class MyEarlyStopping:
#     def __init__(
#         self,
#         _delta=1e-7,
#         _patience=10,
#         _verbose=False,
#         _model_path="temp.pth",
#     ):
#         """Early stopping for PyTorch.
#         _delta:         Threshold to qualify an improvement.
#         _patience:      # to wait after last loss improved.
#         _verbose:       Prints a message for each improvement or not.
#         _model_path:    Path for the checkpoint to be saved to.
#         """
#         self.loss_min = np.Inf
#         self.patience = _patience
#         self.early_stop = False
#         self.verbose = _verbose
#         self.best_score = None
#         self.delta = _delta
#         self.path = _model_path
#         self.counter = 0
#
#     def __call__(self, loss_in, model):
#         """Check if the loss has improved, and if so, save the model."""
#         score = None
#         if self.best_score is None:
#             self.best_score = score
#             self.save_checkpoint(loss_in, model)
#         elif score < self.best_score + self.delta:
#             self.counter += 1
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_score = score
#             self.save_checkpoint(loss_in, model)
#             self.counter = 0
#
#     def save_checkpoint(self, val_loss, model):
#         """Saves model when loss decrease."""
#         if self.verbose:
#             print("ES: {self.val_loss_min:.4f} --> {val_loss:.4f}.")
#         torch.save(model, self.path)
#         self.loss_min = val_loss


# ## Dataloader generation
# In[168]:
class MyDataset(Dataset):
    """Custom dataset
    x:  torch.tensor, the input data
    y:  torch.tensor, the output data
    """
    def __init__(self, x, y):
        self.x, self.y = x, y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# ## Data preprocessing
# In[169]:
def process_data(path, x_offset=720, y_offset=-3840):
    """Preprocess the data
        path: str, the path to the data
        x_offset: int, the x offset
        y_offset: int, the y offset
    Returns:
        x: torch.tensor, the input data
        y: torch.tensor, the output data
    """
    # For safety, drop the NaN values
    data = pd.read_csv(path, header=0).dropna()
    data = data.replace(100, -105)

    # 提取时间列，假设时间列名称为 'time'
    time_col = data['time'] if 'time' in data.columns else None

    scaler_x = StandardScaler()
    x = torch.tensor(scaler_x.fit_transform(data.iloc[:, :-3]), dtype=torch.float32)

    scaler_y = StandardScaler()
    y = torch.tensor(scaler_y.fit_transform(data.iloc[:, -2:]), dtype=torch.float32)

    return x.cuda(), y.cuda(), time_col, scaler_x, scaler_y


# ## Unit test for single floor indoor localization (regression)
# In[170]:
def unit_test(
    # Unit test
    # _epoch: int, the number of epochs
    # _batch_size: int, the batch size
    # _lr: float, the learning rate
    # _weight_decay: float, the weight decay
    # _factor: float, the factor
    # _patience: int, the patience
    # _verbose: bool, the verbose
    # _x_offset: int, the x offset
    # _y_offset: int, the y offset
    # _path_tra: str, the path to the training data
    # _path_tes: str, the path to the test data
    # _fig_title: str, the title of the figure
    # _res_title: str, the title of the result
    _epoch=25,            # 25
    _batch_size=32,       # 16
    _lr=1e-3,             # 1e-3
    _weight_decay=1e-5,   # 1e-5
    _factor=0.5,          # 0.5
    _patience=5,          # 5
    _verbose=True,
    _x_offset=17325.23,
    _y_offset=2970.54,
    # SC4  17325.23   2970.54
    _path_tra="./train_DNN.csv",
    _path_tes="./test_DNN.csv",
    _fig_title="DNNloss.pdf",
    _res_title="DNNpre_res.csv",
):

    # 指定保存结果的文件夹位置
    base_path = "./DNN-R/"
    # 创建完整的文件夹路径
    res_path = os.path.join(base_path)
    # 创建文件夹（如果不存在的话）
    os.makedirs(res_path, exist_ok=True)
    print(f"Results will be saved in: {res_path}")

    # res_path = "./refe/"
    # os.makedirs(res_path, exist_ok=True)

    tra_x, tra_y, tra_time, scaler_x, scaler_y = process_data(_path_tra, _x_offset, _y_offset)
    tes_x, tes_y, tes_time, _, _ = process_data(_path_tes, _x_offset, _y_offset)

    # 划分训练集和验证集（验证集占 20%）
    tra_x, val_x, tra_y, val_y = train_test_split(tra_x.cpu(), tra_y.cpu(), test_size=0.2, random_state=0)
    # 重新转换为 GPU 张量
    tra_x, val_x = tra_x.cuda(), val_x.cuda()
    tra_y, val_y = tra_y.cuda(), val_y.cuda()

    feature_dim = tra_x.shape[1]
    print("模型的特征值维度:", feature_dim)

    tra_loader = DataLoader(
        MyDataset(tra_x, tra_y),
        batch_size=_batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        MyDataset(val_x, val_y),
        batch_size=_batch_size,
        shuffle=False,
    )
    tes_loader = DataLoader(
        MyDataset(tes_x, tes_y),
        batch_size=len(tes_x),
        shuffle=False,
    )

    epoch = _epoch
    encoder = torch.nn.Sequential(
        torch.nn.Linear(tra_x.shape[1], int(tra_x.shape[1] / 2)),
        torch.nn.ReLU(),

        torch.nn.Linear(int(tra_x.shape[1] / 2), int(tra_x.shape[1] / 3)),
        torch.nn.ReLU(),

        torch.nn.Linear(int(tra_x.shape[1] / 3), int(tra_x.shape[1] / 4)),
        torch.nn.ReLU(),

        # torch.nn.Linear(int(tra_x.shape[1] / 4), int(tra_x.shape[1] / 5)),
        # torch.nn.ELU(),
    )
    decoder = torch.nn.Sequential(
        # torch.nn.Linear(int(tra_x.shape[1] / 5), int(tra_x.shape[1] / 4)),
        # torch.nn.ELU(),
        torch.nn.Linear(int(tra_x.shape[1] / 4), int(tra_x.shape[1] / 3)),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),

        torch.nn.Linear(int(tra_x.shape[1] / 3), int(tra_x.shape[1] / 2)),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),

        torch.nn.Linear(int(tra_x.shape[1] / 2), tra_x.shape[1]),
    )
    estimator = torch.nn.Sequential(
        # torch.nn.Linear(int(tra_x.shape[1] / 5), int(tra_x.shape[1] / 4)),
        # torch.nn.ELU(),
        torch.nn.Linear(int(tra_x.shape[1] / 4), int(tra_x.shape[1] / 2)),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),

        torch.nn.Linear(int(tra_x.shape[1] / 2), int(tra_x.shape[1] / 2)),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),

        torch.nn.Linear(int(tra_x.shape[1] / 2), int(tra_x.shape[1] / 2)),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),

        torch.nn.Linear(int(tra_x.shape[1] / 2), int(tra_x.shape[1] / 2)),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),

        torch.nn.Linear(int(tra_x.shape[1] / 2), 2),
    )

    # Loss function
    loss_fn = torch.nn.MSELoss().cuda()
    # Train the sae
    sae = torch.nn.Sequential(
        encoder,
        decoder,
    ).cuda()
    sae_optimizer = torch.optim.Adam(
        sae.parameters(),
    )
    for i in range(epoch):
        sae.train()
        for x, _ in tra_loader:
            # Forward pass
            prd_y = sae(x.cuda())
            # Compute the loss
            loss = loss_fn(prd_y.cuda(), x.cuda())
            # Zero the gradients
            sae_optimizer.zero_grad()
            # Backward pass
            loss.backward()
            # Update the weights
            sae_optimizer.step()
    # Train the model and plot the loss
    loss_fn = torch.nn.MSELoss().cuda()
    model = torch.nn.Sequential(
        encoder,
        estimator,
    ).cuda()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=_lr,
        weight_decay=_weight_decay,
    )

    total_loss = []
    total_val_loss = []
    # early_stopping = MyEarlyStopping()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=_factor,
        # verbose=_verbose,
        patience=_patience,
    )

    # 清空文件内容（如果文件存在）
    with open(os.path.join(res_path, "MSE.log"), "w", encoding="utf-8") as f:
        pass  # 用 "w" 模式打开文件，写入空内容清空文件

    for i in range(int(epoch * 2)):
        model.train()
        for x, y in tra_loader:
            # Compute the loss
            loss = loss_fn(model(x.cuda()), y.cuda())
            # Zero the gradients
            optimizer.zero_grad()
            # Backward pass
            loss.backward()
            # Update the weights
            optimizer.step()
        scheduler.step(loss)
        total_loss.append(loss.item())

        # 计算验证集损失
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                prd_y = model(x.cuda())
                loss = loss_fn(prd_y.cuda(), y.cuda())
                val_loss += loss.item()
        val_loss /= len(val_loader)
        total_val_loss.append(val_loss)
        print(f"Validation Loss: {val_loss:.3f}")

        # 计算当前 epoch 的 MAE 和 R²
        with torch.no_grad():
            all_preds = []
            all_labels = []
            for x, y in tes_loader:
                prd_y = model(x.cuda())
                all_preds.append(prd_y.cpu().numpy())
                all_labels.append(y.cpu().numpy())

            # 将所有的预测值和真实值拼接起来
            all_preds = np.concatenate(all_preds, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)

            # 计算 MAE 和 R²
            mae = mean_absolute_error(all_labels, all_preds)
            r2 = r2_score(all_labels, all_preds)

        # 打印当前学习率
        current_lr = scheduler.optimizer.param_groups[0]['lr']

        # 保存每个 epoch 的结果
        with open(os.path.join(res_path, "MSE.log"), "a", encoding="utf-8") as f:
            print(f"Epoch: {i + 1} - Loss: {loss.item():.3f}, MAE: {mae:.3f}, R²: {r2:.3f}, LR: {current_lr}")
            f.write(f"Epoch {i + 1} - Loss: {loss.item():.3f}, MAE: {mae:.3f}, R²: {r2:.3f}, LR: {current_lr}\n")

        total_loss.append(loss.item())
        # early_stopping(loss, model)

        # 输出模型结构到文本文件
        with open(os.path.join(res_path, "SAE.log"), "w") as f:
            f.write(f"SAE Structure: {sae}\n")

        with open(os.path.join(res_path, "model.log"), "w") as f:
            f.write(f"Model Structure: {model}\n")

    # 保存训练好的模型
    torch.save(model.state_dict(), os.path.join(res_path, "trained_model.pth"))

    # Plot the loss
    fig = plt.figure(figsize=(15, 6), dpi=256)
    # plt.plot(
    #     range(1, len(total_loss) + 1),
    #     total_loss,
    #     marker="o",
    #     ls="--",
    #     lw=".75",
    # )
    plt.plot(range(1, len(total_val_loss) + 1), total_val_loss, label="Validation Loss", color='blue', linestyle='dashed', marker='o')
    plt.plot(range(1, len(total_loss) + 1), total_loss, label="Training Loss", color='red', linestyle='solid', marker='s')

    plt.xticks(np.arange(0, _epoch * 2 + 1, step=5))
    plt.ylim(0, 1)  # 根据需要调整 y 轴的范围
    plt.yticks(np.arange(0, 1.05, 0.05))  # 设置 y 轴刻度的间隔

    plt.xlabel("Epoch")
    plt.ylabel("Training loss (MSE)")
    plt.grid(
        which="both",
        linestyle="--",
        linewidth=0.25,
    )
    plt.legend()
    plt.savefig(
        os.path.join(res_path, _fig_title),
        bbox_inches="tight",
    )
    # Test the model
    model.eval()
    for tes_x, tes_y in tes_loader:
        prd_y = model(tes_x.cuda())
        # Move to cpu
        prd_y = prd_y.cpu().detach().numpy()
        tes_y = tes_y.cpu().detach().numpy()

        # 逆变换
        prd_y = scaler_y.inverse_transform(prd_y)
        tes_y = scaler_y.inverse_transform(tes_y)

        # Save the prediction results
        np.savetxt(
            os.path.join(res_path, "pre.csv"),
            prd_y,
            # using the delimiter
            delimiter=",",
        )
        df = pd.DataFrame(
            {
                "Timestamp": tes_time,  # 真实数据的日期
                "Prediction x": prd_y[:, 0],
                "Ground truth x": tes_y[:, 0],
                "Prediction y": prd_y[:, 1],
                "Ground truth y": tes_y[:, 1],
                "Absolute error x": abs(prd_y[:, 0] - tes_y[:, 0]),
                "Absolute error y": abs(prd_y[:, 1] - tes_y[:, 1]),
                "2D Manhattan error": abs(prd_y[:, 0] - tes_y[:, 0]) + abs(prd_y[:, 1] - tes_y[:, 1]),
                "2D Euclidean error": np.sqrt((prd_y[:, 0] - tes_y[:, 0]) ** 2 + (prd_y[:, 1] - tes_y[:, 1]) ** 2
                ),
            },
        )
        # 计算平均2D欧几里得误差并转换为米，保留四位小数
        mean_euclidean_error = np.mean(df['2D Euclidean error']) / 100

        # 获取测试集文件名并输出平均误差
        test_set_name = os.path.basename(_path_tra)
        average_error = mean_euclidean_error  # 假设这是你计算的平均误差

        # 构建保存路径
        error_file_path = os.path.join(res_path, "average_error.log")

        # 将平均误差输出到文件中
        with open(error_file_path, "w") as error_file:
            error_file.write(f"{test_set_name}: Test MSE: {average_error:.4f} m\n")

        # 输出到控制台确认
        print(f"{test_set_name}: Test MSE: {average_error:.4f} m")

        # 保存预测结果
        df.to_csv(os.path.join(res_path, _res_title), index=False)


start_time = time.time()
unit_test()
end_time = time.time()
period = end_time - start_time
print(f"DNN took {period:.2f} seconds to execute.")
