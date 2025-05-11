#!/usr/bin/env python
# coding: utf-8

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import time
import os
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_error, r2_score

matplotlib.use('TkAgg')

# -------------------------------
# 基本设置与可重复性
# -------------------------------
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.empty_cache()
torch.set_default_dtype(torch.float32)
torch.backends.cudnn.deterministic = True

# -------------------------------
# 自定义数据集
# -------------------------------
class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# -------------------------------
# 数据预处理函数
# -------------------------------
def process_data(path, x_offset=720, y_offset=-3840):
    """
    预处理数据：
      - 读取 CSV 并删除缺失值
      - 将 100 替换为 -105
      - 对输入特征和输出目标分别做标准化
    返回：
      x: tensor（已转到 GPU）
      y: tensor（已转到 GPU）
      time_col: 如果有时间列则返回
      scaler_x, scaler_y: 用于逆变换的标准化器
    """
    data = pd.read_csv(path, header=0).dropna()
    data = data.replace(100, -105)
    time_col = data['time'] if 'time' in data.columns else None

    scaler_x = StandardScaler()
    x = torch.tensor(scaler_x.fit_transform(data.iloc[:, :-3]), dtype=torch.float32)

    scaler_y = StandardScaler()
    y = torch.tensor(scaler_y.fit_transform(data.iloc[:, -2:]), dtype=torch.float32)

    return x.cuda(), y.cuda(), time_col, scaler_x, scaler_y

# -------------------------------
# 定义网络结构
# -------------------------------
import torch.nn as nn

# 构建编码器：提取低维表示
def build_encoder(input_dim):
    return nn.Sequential(
        nn.Linear(input_dim, int(input_dim/2)),
        nn.ReLU(),
        nn.Linear(int(input_dim/2), int(input_dim/3)),
        nn.ReLU(),
        nn.Linear(int(input_dim/3), int(input_dim/4)),
        nn.ReLU(),
    )

# 构建解码器：用于重构输入（SAE部分）
def build_decoder(input_dim):
    return nn.Sequential(
        nn.Linear(int(input_dim/4), int(input_dim/3)),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(int(input_dim/3), int(input_dim/2)),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(int(input_dim/2), input_dim),
    )

# 构建预测器：用于根据低维表示预测目标值（DNN部分）
def build_estimator(input_dim):
    return nn.Sequential(
        nn.Linear(int(input_dim/4), int(input_dim/2)),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(int(input_dim/2), int(input_dim/2)),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(int(input_dim/2), int(input_dim/2)),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(int(input_dim/2), int(input_dim/2)),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(int(input_dim/2), 2),
    )

# -------------------------------
# 主函数：阶段内先训练 SAE 再训练 DNN
# -------------------------------
def unit_test(
    sae_epoch=15,        # 每个阶段 SAE 训练的 epoch 数
    dnn_epoch=15,        # 每个阶段 DNN 训练的 epoch 数
    _batch_size=32,
    _lr=1e-3,
    _weight_decay=1e-5,
    _factor=0.2,
    _patience=10,
    _x_offset=-14654.83,
    _y_offset=6034.64,
    _path_tra1="./train1_simple_dnn.csv",
    _path_tra2="./train2_simple_dnn.csv",
    _path_tra3="./train3_simple_dnn.csv",
    _path_tes="./test_DNN.csv",
    _fig_title="DNNloss.pdf",
    _res_title="DNNpre_res.csv",
):
    # -------------------------------
    # 初始化路径与设备
    # -------------------------------
    base_path = "./Stage-wise/"
    res_path = os.path.join(base_path)
    os.makedirs(res_path, exist_ok=True)
    print(f"Results will be saved in: {res_path}")

    # -------------------------------
    # 加载数据
    # -------------------------------
    tra1_x, tra1_y, _, scaler_x, scaler_y = process_data(_path_tra1, _x_offset, _y_offset)
    tra2_x, tra2_y, _, _, _ = process_data(_path_tra2, _x_offset, _y_offset)
    tra3_x, tra3_y, _, _, _ = process_data(_path_tra3, _x_offset, _y_offset)
    tes_x, tes_y, tes_time, _, _ = process_data(_path_tes, _x_offset, _y_offset)

    # 构造 DataLoader
    tra_loader1 = DataLoader(MyDataset(tra1_x, tra1_y), batch_size=_batch_size, shuffle=True)
    tra_loader2 = DataLoader(MyDataset(tra2_x, tra2_y), batch_size=_batch_size, shuffle=True)
    tra_loader3 = DataLoader(MyDataset(tra3_x, tra3_y), batch_size=_batch_size, shuffle=True)
    tes_loader = DataLoader(MyDataset(tes_x, tes_y), batch_size=len(tes_x), shuffle=False)

    input_dim = tra1_x.shape[1]

    # -------------------------------
    # 全局初始化模型和优化器（关键修改点）
    # -------------------------------
    # SAE部分：编码器 + 解码器
    encoder = build_encoder(input_dim).cuda()
    decoder = build_decoder(input_dim).cuda()
    sae = nn.Sequential(encoder, decoder).cuda()
    sae_optimizer = torch.optim.Adam(sae.parameters(), lr=_lr, weight_decay=_weight_decay)
    sae_loss_fn = nn.MSELoss().cuda()

    # DNN部分：编码器 + 预测器
    estimator = build_estimator(input_dim).cuda()
    model = nn.Sequential(encoder, estimator).cuda()  # 共享 encoder
    dnn_optimizer = torch.optim.Adam(model.parameters(), lr=_lr, weight_decay=_weight_decay)
    dnn_loss_fn = nn.MSELoss().cuda()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(dnn_optimizer, factor=_factor, patience=_patience)

    # -------------------------------
    # 阶段配置：各阶段使用的数据集及对应 loss 权重
    # -------------------------------
    stage_configs = [
        [(tra_loader1, 1.0)],                                        # 阶段1：仅训练集1
        [(tra_loader1, 0.5), (tra_loader2, 1.0)],                     # 阶段2：训练集1（0.5）与训练集2（1）
        [(tra_loader1, 0.25), (tra_loader2, 0.5), (tra_loader3, 1.0)]   # 阶段3：训练集1（0.25）、训练集2（0.5）、训练集3（1）
    ]

    total_loss = []  # 用于记录所有阶段中 DNN 训练的 loss

    # -------------------------------
    # 增量学习循环
    # -------------------------------
    for stage_id, loaders_weights in enumerate(stage_configs, start=1):
        print(f"\n{'='*30} Starting Stage {stage_id} {'='*30}")

        # -------------------------------
        # SAE 训练部分（关键修改点：使用全局模型和优化器）
        # -------------------------------
        sae.train()
        with open(os.path.join(res_path, "SAE.log"), "a") as f:
            f.write(f"\nStage {stage_id} SAE Training\n")

        for ep in range(sae_epoch):
            epoch_loss = 0
            count = 0
            for loader, weight in loaders_weights:
                for x, _ in loader:
                    x = x.cuda()
                    loss = sae_loss_fn(sae(x), x) * weight
                    sae_optimizer.zero_grad()
                    loss.backward()
                    sae_optimizer.step()
                    epoch_loss += loss.item()
                    count += 1

            epoch_loss /= count
            print(f"SAE Stage {stage_id} Epoch {ep+1}/{sae_epoch}: Loss {epoch_loss:.3f}")
            with open(os.path.join(res_path, "SAE.log"), "a") as f:
                f.write(f"Epoch {ep+1}: Loss {epoch_loss:.3f}\n")

        # -------------------------------
        # DNN 训练部分（关键修改点：使用全局模型和优化器）
        # -------------------------------
        model.train()
        with open(os.path.join(res_path, "DNN.log"), "a") as f:
            f.write(f"\nStage {stage_id} DNN Training\n")

        stage_dnn_loss = []  # 只记录 DNN 的损失
        for ep in range(dnn_epoch):
            epoch_loss = 0
            count = 0
            for loader, weight in loaders_weights:
                for x, y in loader:
                    x, y = x.cuda(), y.cuda()
                    loss = dnn_loss_fn(model(x), y) * weight
                    dnn_optimizer.zero_grad()
                    loss.backward()
                    dnn_optimizer.step()
                    epoch_loss += loss.item()
                    count += 1

            epoch_loss /= count
            scheduler.step(epoch_loss)
            stage_dnn_loss.append(epoch_loss)
            total_loss.append(epoch_loss)

            # 计算当前 epoch 的 MAE 和 R²
            model.train()  # 保证模型处于 train 状态
            with torch.no_grad():
                all_preds = []
                all_labels = []
                for x, y in tes_loader:
                    prd_y = model(x.cuda())
                    all_preds.append(prd_y.cpu().numpy())
                    all_labels.append(y.cpu().numpy())

                all_preds = np.concatenate(all_preds, axis=0)
                all_labels = np.concatenate(all_labels, axis=0)
                mae = mean_absolute_error(all_labels, all_preds)
                r2 = r2_score(all_labels, all_preds)

            current_lr = scheduler.optimizer.param_groups[0]['lr']
            print(f"Epoch: {ep + 1}, - Loss: {loss.item():.3f}, MAE: {mae:.3f}, R²: {r2:.3f} LR: {current_lr}")
            with open(os.path.join(res_path, "MSE.log"), "a", encoding="utf-8") as f:
                f.write(f"Epoch {ep + 1} - Loss: {loss.item():.3f}, MAE: {mae:.3f}, R²: {r2:.3f}, LR: {current_lr}\n")

            with open(os.path.join(res_path, "SAE.log"), "w") as f:
                f.write(f"SAE Structure: {sae}\n")
            with open(os.path.join(res_path, "model.log"), "w") as f:
                f.write(f"Model Structure: {model}\n")
        print(f"Stage {stage_id} DNN training completed.")

    # 保存最终训练好的模型
    torch.save(model.state_dict(), os.path.join(res_path, "trained_model.pth"))

    # 绘制整体训练损失曲线（以 DNN阶段 loss 为例）
    fig = plt.figure(figsize=(15, 6), dpi=256)
    plt.plot(range(1, len(total_loss) + 1), total_loss, marker="o", ls="--", lw=".75")
    plt.xticks(np.arange(0, (len(total_loss))+1, step=5))
    plt.ylim(0, max(total_loss)*1.1)
    plt.xlabel("Epoch (across DNN training phases)")
    plt.ylabel("DNN Loss (MSE)")
    plt.grid(which="both", linestyle="--", linewidth=0.25)
    plt.savefig(os.path.join(res_path, _fig_title), bbox_inches="tight")

    # -------------------------------
    # Testing
    # -------------------------------
    model.eval()
    with torch.no_grad():
        for tes_x, tes_y in tes_loader:
            preds = model(tes_x.cuda())
            preds = preds.cpu().detach().numpy()
            tes_y = tes_y.cpu().detach().numpy()
            preds = scaler_y.inverse_transform(preds)
            tes_y = scaler_y.inverse_transform(tes_y)
            df = pd.DataFrame({
                "Timestamp": tes_time,
                "Prediction x": preds[:, 0],
                "Ground truth x": tes_y[:, 0],
                "Prediction y": preds[:, 1],
                "Ground truth y": tes_y[:, 1],
                "Absolute error x": abs(preds[:, 0] - tes_y[:, 0]),
                "Absolute error y": abs(preds[:, 1] - tes_y[:, 1]),
                "2D Manhattan error": abs(preds[:, 0] - tes_y[:, 0]) + abs(preds[:, 1] - tes_y[:, 1]),
                "2D Euclidean error": np.sqrt((preds[:, 0] - tes_y[:, 0])**2 + (preds[:, 1] - tes_y[:, 1])**2),
            })
            mean_euclidean_error = np.mean(df['2D Euclidean error']) / 100
            test_set_name = os.path.basename(_path_tra1)

            with open(os.path.join(res_path, "average_error.log"), "w") as error_file:
                error_file.write(f"{test_set_name}: Test MSE: {mean_euclidean_error:.4f} m\n")
            print(f"{test_set_name}: Test MSE: {mean_euclidean_error:.4f} m")
            df.to_csv(os.path.join(res_path, _res_title), index=False)


start_time = time.time()
unit_test()
end_time = time.time()
print(f"DNN took {(end_time - start_time):.2f} seconds to execute.")
