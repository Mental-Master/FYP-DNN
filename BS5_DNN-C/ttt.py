import pandas as pd

# 指定读取的CSV文件路径
input_path = './train_simple_dnn.csv'

# 读取CSV文件
data = pd.read_csv(input_path)

# 确保 'time' 列为日期时间格式
data['time'] = pd.to_datetime(data['time'], format='%Y/%m/%d %H:%M:%S')

# 按时间排序数据
data = data.sort_values('time')

# 获取唯一的日期
unique_dates = data['time'].dt.date.unique()

# 计算分割点（均分为三份）
num_dates = len(unique_dates)
split_1 = num_dates // 3
split_2 = 2 * (num_dates // 3)

# 获取训练集1、训练集2和训练集3的日期
train1_dates = unique_dates[:split_1]
train2_dates = unique_dates[split_1:split_2]
train3_dates = unique_dates[split_2:]

# 分别提取训练集1、训练集2和训练集3
train1_data = data[data['time'].dt.date.isin(train1_dates)].copy()
train2_data = data[data['time'].dt.date.isin(train2_dates)].copy()
train3_data = data[data['time'].dt.date.isin(train3_dates)].copy()

# 指定保存路径
output_path = './'

# 将 'loc' 列移动到第一列
def reorder_columns(df):
    cols = df.columns.tolist()
    cols.insert(0, cols.pop(cols.index('loc')))
    return df[cols]

train1_data = reorder_columns(train1_data)
train2_data = reorder_columns(train2_data)
train3_data = reorder_columns(train3_data)

# 保存数据集
train1_data.to_csv(f'{output_path}train1_simple_dnn.csv', index=False)
train2_data.to_csv(f'{output_path}train2_simple_dnn.csv', index=False)
train3_data.to_csv(f'{output_path}train3_simple_dnn.csv', index=False)

print(f"数据集已均分为三份，并保存到 {output_path} 路径下.")
