import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

'''
    satisfaction_level：员工对公司满意度
    last_evaluation：上一次公司对员工的评价
    number_project：该员工同时负责多少项目
    average_montly_hours：每个月工作的时长
    time_spend_company：在公司工作多久
    Work_accident：是否有个工作事故

    标签：left：是否离开公司（1表示离职）

    promotion_last_5years：过去5年是否又被升职
    sales：部门
    salary：工资水平
'''

df = pd.read_csv('HR.csv', index_col=None)
# 检测是否有缺失数据
print(df.isnull().any())

print(df.head())
print(df.shape)

# 查看离职率
turnover_rate = df.left.value_counts() / len(df)
print(turnover_rate)

# 分组统计
turnover_Summary = df.groupby('left')
print(turnover_Summary.mean())

# 相关性分析
corr = df.corr()
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

# 将数据集中的字符串数据转换成数字类型数据
print(df.dtypes)
print(df["sales"].unique())
print(df["salary"].unique())

# pd.Series._accessors 该方法可以返回属性，让我们知道某个数据有哪些属性可以使用。具体做法先转换为category，然后再使用 cat.codes 来实现对整数的映射
df["sales"] = df["sales"].astype('category')
df["salary"] = df["salary"].astype('category')
df["sales"] = df["sales"].cat.codes
df["salary"] = df["salary"].cat.codes

target_name = 'left'
X = df.drop('left', axis=1)
y = df[target_name]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=123, stratify=y)
