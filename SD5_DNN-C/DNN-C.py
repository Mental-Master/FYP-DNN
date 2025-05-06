# A simple DNN with 1 SAE and 2 hidden layers
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

# matplotlib.use('Agg')
matplotlib.use('TkAgg')

# ------------------------------------------------------------------------
# >>> General
# ------------------------------------------------------------------------
# GPU Configuration
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(
        ",".join([str(i) for i in range(torch.cuda.device_count())])
    )
    print("Working on GPU: {}".format(torch.cuda.device_count()))

# Set random seeds for reproducibility
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
np.random.seed(12345)
torch.manual_seed(12345)
torch.cuda.manual_seed(12345)

#  GPU optimization and memory management
torch.backends.cudnn.benchmark = True
torch.set_default_dtype(torch.float64)
torch.set_default_device("cuda")
torch.cuda.set_per_process_memory_fraction(0.5)

# Set the plot style
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = "Serif"
plt.rcParams["figure.figsize"] = (10, 5)

# ------------------------------------------------------------------------
# >>> Model Settings
# ------------------------------------------------------------------------
# File paths
train_file_path = "./train_simple_dnn.csv"
base_path = "./simple_dnn/"
res_path = os.path.join(base_path)
os.makedirs(res_path, exist_ok=True)
print(f"Results will be saved in: {res_path}")

# Read the training and validation data
tra = pd.read_csv("./train_simple_dnn.csv",)
val = pd.read_csv("./test_simple_dnn.csv",)

# Hyperparameters and dimensions
EPOCH_SAE = 20
EPOCH_MLC = 30
FEA_DIM = len([col for col in tra.columns if 'WAP' in col]) - 1
CLS_DIM = 150
HID_DIM = 512
SAE_DIM = 116
BATCH_SIZE = 16

# ------------------------------------------------------------------------
# >>> Data Loading
# ------------------------------------------------------------------------

# # Prepare input features and labels
tra_x = tra.iloc[:, 1:-4].values
val_x = val.iloc[:, 1:-4].values
tra_x[tra_x < -105] = -105
val_x[val_x < -105] = -105

# Get the maximum value
print("Maximum value of training and validation data: {:.3f}, {:.3f}".format(np.max(tra_x), np.max(val_x)))
# Get the minimum value
print("Minimum value of training and validation data: {:.3f}, {:.3f}".format(np.min(tra_x), np.min(val_x)))

# Scale features
tra_x = scale(tra_x, axis=1)
val_x = scale(val_x, axis=1)

# Get labels
tra_r = tra.iloc[:, 0].values
val_r = val.iloc[:, 0].values

# Convert to Torch Tensor
tra_x = torch.from_numpy(tra_x)
tra_r = torch.from_numpy(tra_r)
val_x = torch.from_numpy(val_x)
val_r = torch.from_numpy(val_r)

# Convert to dataloader
tra = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(tra_x, tra_r),
    batch_size=BATCH_SIZE, drop_last=True)

# Remove the log files
for i in os.listdir(res_path):
    if i.endswith(".log"):
        # Correctly join the directory and file name
        file_path = os.path.join(res_path, i)
        if os.path.exists(file_path):  # Check if the file exists
            os.remove(file_path)
        else:
            print(f"File not found: {file_path}")

# Loading Test Data and Labels
val_x = val_x.cuda()
val_r = val_r.cuda()

# Move the criterion to the GPU
cre = nn.CrossEntropyLoss().cuda()
mse = nn.MSELoss().cuda()

# Initialize the timer
start_time = time.time()

print("Label min:", val_r.min().item())
print("Label max:", val_r.max().item())
print("CLS_DIM:", CLS_DIM)


def calculate_accuracy(model, data_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy


def save_predictions_with_original_data(val_complete, pred, res_path):
    # 提取 loc, x, y 和 timestamp 列
    df_loc = pd.DataFrame({
        "loc": val_complete.iloc[:, 0],
        "x": val_complete.iloc[:, -2],
        "y": val_complete.iloc[:, -1],
        "time": val_complete["time"]  # 提取时间戳
    }).drop_duplicates()   # 去重

    # 去除 loc 列的重复值
    df_loc = df_loc.groupby('loc').first().reset_index()
    lookup_table = df_loc.set_index('loc').loc[:, ['x', 'y']].to_dict('index')

    # 真实标签 (loc) 和时间戳
    true = val_complete.iloc[:, 0].values
    timestamps = val_complete["time"].values  # 获取时间戳

    # 创建一个完整的 DataFrame 来保存结果
    df_complete = pd.DataFrame({
        "Time": timestamps,  # 真实数据的日期
        "Predicted": pred.cpu().numpy(),  # 预测的 loc
        "Truth": true  # 真实的 loc
    })

    # 一次性将 val_complete 中的其他列添加到结果 DataFrame 中
    # 这里只保留非 WAP 的列，假设 WAP 列的列名包含 "WAP"
    non_wap_columns = [col for col in val_complete.columns if "WAP" not in col and col != "timestamp"]
    new_columns = val_complete[non_wap_columns].iloc[:, 1:]  # 可能包含其他相关信息，去除第一列（loc列）
    df_complete = pd.concat([df_complete, new_columns.reset_index(drop=True)], axis=1)

    # 从 df_loc 查找 loc 对应的 x 和 y 坐标
    lookup_table = df_loc.set_index('loc').loc[:, ['x', 'y']].to_dict('index')

    def get_coordinates(loc_z):
        # 确保返回的永远是两个元素的元组，即使 loc_z 不存在
        return lookup_table.get(loc_z)

    # 获取预测和真实的 x 和 y 坐标
    df_complete[['Predicted_x', 'Predicted_y']] = df_complete['Predicted'].apply(
        lambda loc: pd.Series(get_coordinates(loc), dtype=float))
    df_complete[['Truth_x', 'Truth_y']] = df_complete['Truth'].apply(
        lambda loc: pd.Series(get_coordinates(loc), dtype=float))

    # 计算欧几里得距离误差
    def euclidean_distance(row):
        # 如果有 None 值，跳过计算
        if pd.isnull(row['Predicted_x']) or pd.isnull(row['Truth_x']):
            return None
        return np.sqrt((row['Predicted_x'] - row['Truth_x']) ** 2 + (row['Predicted_y'] - row['Truth_y']) ** 2)

    # 计算误差并存储
    df_complete['errors'] = df_complete.apply(euclidean_distance, axis=1)
    # 去掉包含 None 值的行
    df_complete = df_complete.dropna(subset=['errors'])

    # 将误差单位从厘米转换为米，保留4位小数
    df_complete['errors_m'] = (df_complete['errors'] / 100).round(4)
    # 计算平均误差 (单位: 米)
    average_error = df_complete['errors_m'].mean()

    # # 保存完整的预测结果到 CSV 文件，误差保留4位小数
    error_file_path = os.path.join(res_path, "average_error.log")
    with open(error_file_path, "a+") as error_file:
        error_file.write(f'Average Error: {average_error:.4f} m\n')
    print(f'Average Error: {average_error:.4f} m')

    df_complete.to_csv(os.path.join(res_path, "res_complete.csv"), index=False)


# ------------------------------------------------------------------------
# >>> Define SAE
# ------------------------------------------------------------------------
class SAE(nn.Module):
    """Stacked Autoencoder"""

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


# ------------------------------------------------------------------------
# >>> Define CLS
# ------------------------------------------------------------------------
class CLS(nn.Module):
    """Multi-label classification model for building and floor"""

    def __init__(self, _enc):
        # Initialize the model
        super(CLS, self).__init__()
        try:
            # Load the encoder
            self.encoder = _enc
        except FileNotFoundError:
            raise FileNotFoundError("Encoder not found.")
        self.cls = nn.Sequential(
            # Encoder
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
        """Forward propagation"""
        return self.cls(_x)

    def save(self, _p):
        """Save the model"""
        torch.save(self.cls, _p)

    def show(self):
        """Show the model"""
        print(self.cls, file=open(os.path.join(res_path, "cls_model_shape.log"), "a+"))


# ------------------------------------------------------------------------
# >>> Training SAE
# ------------------------------------------------------------------------
# Initialize the SAE
sae = SAE().cuda()
# Define the optimizer
opt_sae = torch.optim.Adam(sae.parameters(), lr=1e-3)
# Training ---------------------------------------------------------------
print("=" * 25 + "\n" + "Training SAE" + "\n" + "=" * 25, file=open(os.path.join(res_path, "sae.log"), "a+"))
for epoch in range(EPOCH_SAE):
    for _, (x, _) in enumerate(tra):
        x = x.cuda()
        los_sae = mse(sae(x), x)
        opt_sae.zero_grad()
        los_sae.backward()
        opt_sae.step()
    print(
        "Epoch: [{:3d}|{:3d}] \t | Training Loss {:.3f}".format(epoch + 1, EPOCH_SAE, los_sae.item()),
        file=open(os.path.join(res_path, "sae.log"), "a+")
    )

# ------------------------------------------------------------------------
# >>> CLS
# ------------------------------------------------------------------------
# Initialize the CLS
cls = CLS(sae.encoder.train()).cuda().train()
# Define the optimizer
opt_mlc = torch.optim.Adam(cls.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(opt_mlc, step_size=int(EPOCH_MLC / 2), gamma=0.75)

# Training ---------------------------------------------------------------
print("=" * 25 + "\n" + "Training CLS" + "\n" + "=" * 25, file=open(os.path.join(res_path, "cls.log"), "a+"))
train_accuracies = []  # 用于存储每个epoch的训练准确率
mlc_loss = []

for epoch in range(EPOCH_MLC):
    cls.train()
    for _, (x, y) in enumerate(tra):
        x = x.cuda()
        y = y.cuda()
        loss = cre(cls(x), y)
        opt_mlc.zero_grad()
        loss.backward()
        opt_mlc.step()

    # 每个epoch后计算并打印训练准确率
    train_accuracy = calculate_accuracy(cls, tra)
    train_accuracies.append(train_accuracy)  # 存储准确率
    print(
        "Epoch: [{:3d}|{:3d}] | Training Loss: {:.3f} | Training Accuracy: {:.2f}%"
        .format(epoch + 1, EPOCH_MLC, loss.item(), train_accuracy),
        file=open(os.path.join(res_path, "cls.log"), "a+"),
    )
    mlc_loss.append(loss.item())
    scheduler.step()

cls.eval()
cls.show()

# Plot the loss
fig = plt.figure()
plt.plot(
    np.arange(1, EPOCH_MLC + 1),
    mlc_loss,
    "o-",
    alpha=0.75,
    color="blue",
    label="Training Loss",
    linewidth=0.75,
)
plt.legend(
    loc="best",
    frameon=True,
    fontsize=8,
    framealpha=0.5,
    edgecolor="black",
    fancybox=False,
)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xticks(np.arange(0, EPOCH_MLC + 1, 2))
plt.ylim(-1, 8)
plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.5,
)
plt.savefig(
    os.path.join(res_path, "loss.png"),
    bbox_inches="tight",
    dpi=256,
)

# ------------------------------------------------------------------------
# >>> Testing
# ------------------------------------------------------------------------
# Loading the model
# Predicting
pred = cls(val_x)
# Get the predicted labels
pred = torch.argmax(pred, axis=1)
true = val_r

# Save predictions with original data
save_predictions_with_original_data(val, pred, res_path)

# Convert to numpy for sklearn metrics
pred = pred.cpu().numpy()
true = true.cpu().numpy()

# Disable warnings for UndefinedMetricWarning specifically
warnings.filterwarnings("ignore", message=".*ill-defined and being set to 0.0*")

# Calculate Precision, Recall, and F1-score
precision = precision_score(true, pred, average=None, zero_division=1)  # per-class precision
recall = recall_score(true, pred, average=None, zero_division=1)  # per-class recall
f1 = f1_score(true, pred, average=None, zero_division=1)  # per-class F1-score

# Calculate the actual unique classes present in true and pred
unique_true_classes = np.unique(true)
unique_pred_classes = np.unique(pred)

# The actual classes that appear in both true and pred
actual_classes = np.intersect1d(unique_true_classes, unique_pred_classes)

# Ensure target_names matches the actual classes
target_names = [f'Class {i}' for i in actual_classes]

# Ensure precision, recall, and f1 are based only on actual classes
# Clip actual_classes to match the index range of precision, recall, f1
actual_classes = np.clip(actual_classes, 0, len(precision) - 1)
precision = precision[actual_classes]  # Only use the actual classes
recall = recall[actual_classes]
f1 = f1[actual_classes]

# Print classification report with labels corresponding to actual classes
report = classification_report(true, pred, target_names=target_names, labels=actual_classes)

# Save classification report to CSV
report_dict = classification_report(true, pred, target_names=target_names, labels=actual_classes, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()

# Save the full classification report
report_df.to_csv(os.path.join(res_path, 'classification_report.csv'), index=True)

# 确保 precision, recall, f1 与 target_names 的长度一致
assert len(precision) == len(target_names), \
    f"Precision length {len(precision)} does not match target names length {len(target_names)}"
assert len(recall) == len(target_names), \
    f"Recall length {len(recall)} does not match target names length {len(target_names)}"
assert len(f1) == len(target_names), \
    f"F1-score length {len(f1)} does not match target names length {len(target_names)}"

# Extract Precision, Recall, and F1-score for each class and save them to separate CSV files
precision_df = pd.DataFrame(precision, columns=['Precision'], index=target_names)
precision_df.to_csv(os.path.join(res_path, 'precision.csv'), index=True)
recall_df = pd.DataFrame(recall, columns=['Recall'], index=target_names)
recall_df.to_csv(os.path.join(res_path, 'recall.csv'), index=True)
f1_score_df = pd.DataFrame(f1, columns=['F1-score'], index=target_names)
f1_score_df.to_csv(os.path.join(res_path, 'f1_score.csv'), index=True)

# Calculate accuracy per group
pred = torch.split(torch.from_numpy(pred), CLS_DIM)
true = torch.split(torch.from_numpy(true), CLS_DIM)

# Plot the accuracy for each group
fig = plt.figure()
plt.plot(
    np.arange(1, EPOCH_MLC + 1),  # x轴为epoch数
    train_accuracies,  # y轴为准确率
    "o-",  # 曲线风格
    alpha=0.75,
    color="blue",
    label="Training Accuracy",
    linewidth=0.75,
)
plt.legend(
    loc="best",
    frameon=True,
    fontsize=8,
    framealpha=0.5,
    edgecolor="black",
    fancybox=False,
)
plt.xlabel("Group")
plt.ylabel("Accuracy")
plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.5,
)
plt.xticks(np.arange(0, EPOCH_MLC + 1, 2))
plt.xlim(0, EPOCH_MLC + 1)
plt.yticks(np.arange(0, 1.1, 0.1))
plt.ylim(0, 1.1)
plt.savefig(
    os.path.join(res_path, "acc.png"),
    bbox_inches="tight",
    dpi=256,
)
plt.close(fig)

end_time = time.time()
period = end_time - start_time
print(f"simple_dnn took {period:.2f} seconds to execute.")
