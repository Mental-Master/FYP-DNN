import os
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import scale
import warnings
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

# 设置 matplotlib 使用 TkAgg 后端
matplotlib.use('TkAgg')

# ------------------------------------------------------------------------
# >>> GPU 配置及随机数种子设置
# ------------------------------------------------------------------------
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(
        ",".join([str(i) for i in range(torch.cuda.device_count())])
    )
    print("Working on GPU: {}".format(torch.cuda.device_count()))

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
np.random.seed(12345)
torch.manual_seed(12345)
torch.cuda.manual_seed(12345)

torch.backends.cudnn.benchmark = True
torch.set_default_dtype(torch.float64)
torch.set_default_device("cuda")

torch.cuda.set_per_process_memory_fraction(0.5)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = "Serif"
plt.rcParams["figure.figsize"] = (10, 5)

# ------------------------------------------------------------------------
# >>> 路径及结果保存文件夹
# ------------------------------------------------------------------------
base_path = "./CL/"
res_path = os.path.join(base_path)
os.makedirs(res_path, exist_ok=True)
print(f"Results will be saved in: {res_path}")

# 删除已有的 .log 文件
for i in os.listdir(res_path):
    if i.endswith(".log"):
        file_path = os.path.join(res_path, i)
        if os.path.exists(file_path):
            os.remove(file_path)


# ------------------------------------------------------------------------
# >>> 数据加载函数及自定义 Dataset（附带样本权重）
# ------------------------------------------------------------------------
def load_data(csv_file):
    df = pd.read_csv(csv_file)
    # 假设：第一列为标签，后面列中第2列到倒数第5列为特征
    x = df.iloc[:, 1:-4].values
    x[x < -105] = -105
    x = scale(x, axis=1)
    y = df.iloc[:, 0].values
    # 转换为 Torch Tensor
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    return x, y


class WeightedDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, weight):
        self.x = x
        self.y = y
        # 对该数据集内每个样本赋予相同的权重（标量）
        self.weight = torch.tensor(weight, dtype=torch.float64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.weight


# 加载三个训练集
tra1_x, tra1_y = load_data("./train1_simple_dnn.csv")
tra2_x, tra2_y = load_data("./train2_simple_dnn.csv")
tra3_x, tra3_y = load_data("./train3_simple_dnn.csv")

# 加载验证集（格式与原代码一致）
val = pd.read_csv("./test_simple_dnn.csv", )
val_x = val.iloc[:, 1:-4].values
val_x[val_x < -105] = -105
print("Maximum value of validation data: {:.3f}".format(np.max(val_x)))
print("Minimum value of validation data: {:.3f}".format(np.min(val_x)))
val_x = scale(val_x, axis=1)
val_x = torch.from_numpy(val_x)
val_r = torch.from_numpy(val.iloc[:, 0].values)

# 移动验证数据到 GPU
val_x = val_x.cuda()
val_r = val_r.cuda()

# ------------------------------------------------------------------------
# >>> SAE 模型及训练（每轮先训练 SAE，再训练 CLS）
# ------------------------------------------------------------------------
FEA_DIM = tra1_x.shape[1]  # 特征维度
SAE_DIM = 116


class SAE(nn.Module):
    def __init__(self):
        super(SAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(FEA_DIM, int(FEA_DIM / 2)),
            nn.ELU(),
            nn.Linear(int(FEA_DIM / 2), int(FEA_DIM / 3)),
            nn.ELU(),
            nn.Linear(int(FEA_DIM / 3), SAE_DIM)
        )
        self.decoder = nn.Sequential(
            nn.Linear(SAE_DIM, int(FEA_DIM / 3)),
            nn.ELU(),
            nn.Linear(int(FEA_DIM / 3), int(FEA_DIM / 2)),
            nn.ELU(),
            nn.Linear(int(FEA_DIM / 2), FEA_DIM)
        )

    def forward(self, _x):
        return self.decoder(self.encoder(_x))


# 初始化 SAE 并定义优化器及损失函数
sae = SAE().cuda()
mse = nn.MSELoss().cuda()
sae_optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)

# ------------------------------------------------------------------------
# >>> CLS 模型及训练
# ------------------------------------------------------------------------
CLS_DIM = 400
HID_DIM = 512


class CLS(nn.Module):
    def __init__(self, _enc):
        super(CLS, self).__init__()
        self.encoder = _enc
        self.cls = nn.Sequential(
            self.encoder,
            nn.Linear(SAE_DIM, HID_DIM),
            nn.ELU(),
            nn.Linear(HID_DIM, HID_DIM),
            nn.ELU(),
            nn.Linear(HID_DIM, HID_DIM),
            nn.ELU(),
            nn.Linear(HID_DIM, HID_DIM),
            nn.ELU(),
            nn.BatchNorm1d(HID_DIM),
            nn.Linear(HID_DIM, CLS_DIM),
        )

    def forward(self, _x):
        return self.cls(_x)

    def save(self, _p):
        torch.save(self.cls, _p)

    def show(self):
        print(self.cls, file=open(os.path.join(res_path, "cls_model_shape.log"), "a+"))


# 使用 SAE 的 encoder 作为 CLS 的一部分（注意：这里传入的是 sae.encoder，后续 SAE 的更新会影响 CLS）
cls = CLS(sae.encoder.train()).cuda().train()

# 定义 CLS 的损失函数（交叉熵，需要对样本损失加权，所以 reduction='none'）
cre = nn.CrossEntropyLoss(reduction='none').cuda()
cls_optimizer = torch.optim.Adam(cls.parameters(), lr=5e-4)
# 使用简单的 StepLR 调度器（可根据需要调整）
cls_scheduler = torch.optim.lr_scheduler.StepLR(cls_optimizer, step_size=5, gamma=0.75)


# ------------------------------------------------------------------------
# >>> 定义训练函数（每个 epoch 先训练 SAE，再训练 CLS）
# ------------------------------------------------------------------------
def train_epoch(model_sae, model_cls, loader, optimizer_sae, optimizer_cls):
    model_sae.train()
    model_cls.train()
    total_sae_loss = 0.0
    total_cls_loss = 0.0

    # 遍历数据集（同一 loader 同时用于 SAE 和 CLS 训练）
    for x, y, sample_weight in loader:
        x = x.cuda()
        y = y.cuda()
        sample_weight = sample_weight.cuda()

        # === SAE 部分训练：重构损失 ===
        optimizer_sae.zero_grad()
        recon = model_sae(x)
        loss_sae = mse(recon, x)
        loss_sae.backward()
        optimizer_sae.step()
        total_sae_loss += loss_sae.item()

        # === CLS 部分训练：加权交叉熵 ===
        optimizer_cls.zero_grad()
        outputs = model_cls(x)
        loss_each = cre(outputs, y)  # 每个样本的 loss
        loss_cls = (loss_each * sample_weight).mean()
        loss_cls.backward()
        optimizer_cls.step()
        total_cls_loss += loss_cls.item()

    avg_sae_loss = total_sae_loss / len(loader)
    avg_cls_loss = total_cls_loss / len(loader)
    return avg_sae_loss, avg_cls_loss


# ------------------------------------------------------------------------
# >>> 训练阶段设置
# ------------------------------------------------------------------------
EPOCH_STAGE = 10  # 每个阶段训练的 epoch 数

# 阶段1：仅使用训练集1（权重均为 1.0）
dataset_stage1 = WeightedDataset(tra1_x, tra1_y, weight=1.0)

# 阶段2：训练集1（权重 0.5）与训练集2（权重 1.0）
from torch.utils.data import ConcatDataset

dataset_stage1_2 = WeightedDataset(tra1_x, tra1_y, weight=0.5)
dataset_stage2 = WeightedDataset(tra2_x, tra2_y, weight=1.0)
concat_dataset_stage2 = ConcatDataset([dataset_stage1_2, dataset_stage2])

# 阶段3：训练集1（权重 0.25）、训练集2（权重 0.5）和训练集3（权重 1.0）
dataset_stage1_3 = WeightedDataset(tra1_x, tra1_y, weight=0.25)
dataset_stage2_3 = WeightedDataset(tra2_x, tra2_y, weight=0.5)
dataset_stage3 = WeightedDataset(tra3_x, tra3_y, weight=1.0)
concat_dataset_stage3 = ConcatDataset([dataset_stage1_3, dataset_stage2_3, dataset_stage3])

# 定义各阶段 DataLoader
loader_stage1 = torch.utils.data.DataLoader(dataset_stage1, batch_size=16, shuffle=True,
                                            drop_last=True, generator=torch.Generator(device='cuda'))
loader_stage2 = torch.utils.data.DataLoader(concat_dataset_stage2, batch_size=16, shuffle=True,
                                            drop_last=True, generator=torch.Generator(device='cuda'))
loader_stage3 = torch.utils.data.DataLoader(concat_dataset_stage3, batch_size=16, shuffle=True,
                                            drop_last=True, generator=torch.Generator(device='cuda'))

# 用于记录训练过程中的损失
sae_loss_list = []
cls_loss_list = []

start_time = time.time()

# ------------------------------------------------------------------------
# >>> 阶段1训练（Train1 only）
# ------------------------------------------------------------------------
with open(os.path.join(res_path, "cls.log"), "a+") as f:
    f.write("=" * 25 + "\nTraining CLS Stage 1 (Train1 only)\n" + "=" * 25 + "\n")
for epoch in range(EPOCH_STAGE):
    avg_sae_loss, avg_cls_loss = train_epoch(sae, cls, loader_stage1, sae_optimizer, cls_optimizer)
    sae_loss_list.append(avg_sae_loss)
    cls_loss_list.append(avg_cls_loss)
    with open(os.path.join(res_path, "cls.log"), "a+") as f:
        f.write("Stage1 - Epoch: [{:3d}|{:3d}] | SAE Loss: {:.3f} | CLS Loss: {:.3f}\n".format(
            epoch + 1, EPOCH_STAGE, avg_sae_loss, avg_cls_loss))
    cls_scheduler.step()

# ------------------------------------------------------------------------
# >>> 阶段2训练（Train1 weight 0.5, Train2 weight 1.0）
# ------------------------------------------------------------------------
with open(os.path.join(res_path, "cls.log"), "a+") as f:
    f.write("=" * 25 + "\nTraining CLS Stage 2 (Train1:0.5, Train2:1.0)\n" + "=" * 25 + "\n")
for epoch in range(EPOCH_STAGE):
    avg_sae_loss, avg_cls_loss = train_epoch(sae, cls, loader_stage2, sae_optimizer, cls_optimizer)
    sae_loss_list.append(avg_sae_loss)
    cls_loss_list.append(avg_cls_loss)
    with open(os.path.join(res_path, "cls.log"), "a+") as f:
        f.write("Stage2 - Epoch: [{:3d}|{:3d}] | SAE Loss: {:.3f} | CLS Loss: {:.3f}\n".format(
            epoch + 1, EPOCH_STAGE, avg_sae_loss, avg_cls_loss))
    cls_scheduler.step()

# ------------------------------------------------------------------------
# >>> 阶段3训练（Train1:0.25, Train2:0.5, Train3:1.0）
# ------------------------------------------------------------------------
with open(os.path.join(res_path, "cls.log"), "a+") as f:
    f.write("=" * 25 + "\nTraining CLS Stage 3 (Train1:0.25, Train2:0.5, Train3:1.0)\n" + "=" * 25 + "\n")
for epoch in range(EPOCH_STAGE):
    avg_sae_loss, avg_cls_loss = train_epoch(sae, cls, loader_stage3, sae_optimizer, cls_optimizer)
    sae_loss_list.append(avg_sae_loss)
    cls_loss_list.append(avg_cls_loss)
    with open(os.path.join(res_path, "cls.log"), "a+") as f:
        f.write("Stage3 - Epoch: [{:3d}|{:3d}] | SAE Loss: {:.3f} | CLS Loss: {:.3f}\n".format(
            epoch + 1, EPOCH_STAGE, avg_sae_loss, avg_cls_loss))
    cls_scheduler.step()

cls.eval()
cls.show()


# ------------------------------------------------------------------------
# >>> 验证及保存预测结果
# ------------------------------------------------------------------------
def calculate_accuracy(model, data_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels, _ in data_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


# 计算验证集上准确率
def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        outputs = model(x)
        _, predicted = torch.max(outputs, 1)
    return predicted


pred = evaluate(cls, val_x, val_r)


# 保存预测结果与原始验证数据关联（与原代码类似）
def save_predictions_with_original_data(val_complete, pred, res_path):
    # 提取 loc, x, y 和时间戳列（假设第一列为 loc，最后两列为 x,y，且存在 time 列）
    df_loc = pd.DataFrame({
        "loc": val_complete.iloc[:, 0],
        "x": val_complete.iloc[:, -2],
        "y": val_complete.iloc[:, -1],
        "time": val_complete["time"]
    }).drop_duplicates()
    df_loc = df_loc.groupby('loc').first().reset_index()
    lookup_table = df_loc.set_index('loc')[['x', 'y']].to_dict('index')

    true = val_complete.iloc[:, 0].values
    timestamps = val_complete["time"].values

    df_complete = pd.DataFrame({
        "Time": timestamps,
        "Predicted": pred.cpu().numpy(),
        "Truth": true
    })

    non_wap_columns = [col for col in val_complete.columns if "WAP" not in col and col != "timestamp"]
    new_columns = val_complete[non_wap_columns].iloc[:, 1:]
    df_complete = pd.concat([df_complete, new_columns.reset_index(drop=True)], axis=1)

    def get_coordinates(loc_z):
        return lookup_table.get(loc_z)

    df_complete[['Predicted_x', 'Predicted_y']] = df_complete['Predicted'].apply(
        lambda loc: pd.Series(get_coordinates(loc)))
    df_complete[['Truth_x', 'Truth_y']] = df_complete['Truth'].apply(
        lambda loc: pd.Series(get_coordinates(loc)))

    def euclidean_distance(row):
        if pd.isnull(row['Predicted_x']) or pd.isnull(row['Truth_x']):
            return None
        return np.sqrt((row['Predicted_x'] - row['Truth_x']) ** 2 + (row['Predicted_y'] - row['Truth_y']) ** 2)

    df_complete['errors'] = df_complete.apply(euclidean_distance, axis=1)
    df_complete = df_complete.dropna(subset=['errors'])
    df_complete['errors_m'] = (df_complete['errors'] / 100).round(4)
    average_error = df_complete['errors_m'].mean()
    error_file_path = os.path.join(res_path, "average_error.log")
    with open(error_file_path, "a+") as error_file:
        error_file.write(f'Average Error: {average_error:.4f} m\n')
    print(f'Average Error: {average_error:.4f} m')
    df_complete.to_csv(os.path.join(res_path, "res_complete.csv"), index=False)


# 使用验证集进行预测并保存结果
pred = evaluate(cls, val_x, val_r)
save_predictions_with_original_data(val, pred, res_path)

# 计算分类指标
pred_np = pred.cpu().numpy()
true_np = val_r.cpu().numpy()

warnings.filterwarnings("ignore", message=".*ill-defined and being set to 0.0*")
precision = precision_score(true_np, pred_np, average=None, zero_division=1)
recall = recall_score(true_np, pred_np, average=None, zero_division=1)
f1 = f1_score(true_np, pred_np, average=None, zero_division=1)

unique_true_classes = np.unique(true_np)
unique_pred_classes = np.unique(pred_np)
actual_classes = np.intersect1d(unique_true_classes, unique_pred_classes)
target_names = [f'Class {i}' for i in actual_classes]
actual_classes = np.clip(actual_classes, 0, len(precision) - 1)
precision = precision[actual_classes]
recall = recall[actual_classes]
f1 = f1[actual_classes]

report = classification_report(true_np, pred_np, target_names=target_names, labels=actual_classes)
report_dict = classification_report(true_np, pred_np, target_names=target_names, labels=actual_classes,
                                    output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv(os.path.join(res_path, 'classification_report.csv'), index=True)

precision_df = pd.DataFrame(precision, columns=['Precision'], index=target_names)
precision_df.to_csv(os.path.join(res_path, 'precision.csv'), index=True)
recall_df = pd.DataFrame(recall, columns=['Recall'], index=target_names)
recall_df.to_csv(os.path.join(res_path, 'recall.csv'), index=True)
f1_score_df = pd.DataFrame(f1, columns=['F1-score'], index=target_names)
f1_score_df.to_csv(os.path.join(res_path, 'f1_score.csv'), index=True)

# 绘制 CLS 训练过程中的损失曲线
fig = plt.figure()
plt.plot(np.arange(1, len(cls_loss_list) + 1), cls_loss_list, "o-", alpha=0.75, color="blue", label="CLS Loss",
         linewidth=0.75)
plt.legend(loc="best", frameon=True, fontsize=8, framealpha=0.5, edgecolor="black", fancybox=False)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.savefig(os.path.join(res_path, "loss.png"), bbox_inches="tight", dpi=256)

end_time = time.time()
period = end_time - start_time
print(f"simple_dnn took {period:.2f} seconds to execute.")
