import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

# 定义文件夹路径
train_input_directory = './transcriptions_out_train'  # 替换为你的训练文件夹路径
test_input_directory = './transcriptions_out_8_28'  # 替换为你的测试文件夹路径
output_csv_path = 'LR1.csv'  # 替换为你想要的输出文件路径
output_txt_path = 'LR_results_10000.txt'  # 替换为你想要的输出文本文件路径

# 获取训练文件夹中的所有文件
train_file_names = [f for f in os.listdir(train_input_directory) if f.endswith('.txt')]

# 提取训练文件名和topic标签
train_data = []
for file_name in tqdm(train_file_names, desc="Reading train files"):
    parts = file_name.split('-')
    if len(parts) >= 3:
        topic = parts[2]
        with open(os.path.join(train_input_directory, file_name), 'r', encoding='utf-8') as file:
            text = file.read()
            train_data.append([text, topic])

# 创建训练DataFrame
train_df = pd.DataFrame(train_data, columns=['Text', 'Topic'])

# 文本向量化
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(tqdm(train_df['Text'], desc="Vectorizing train data"))

# 训练模型
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, train_df['Topic'])

# 获取测试文件夹中的所有文件
test_file_names = [f for f in os.listdir(test_input_directory) if f.endswith('.txt')]

# 提取测试文件名和topic标签
test_data = []
for file_name in tqdm(test_file_names, desc="Reading test files"):
    parts = file_name.split('-')
    if len(parts) >= 3:
        topic = parts[2]
        with open(os.path.join(test_input_directory, file_name), 'r', encoding='utf-8') as file:
            text = file.read()
            test_data.append([text, topic])

# 创建测试DataFrame
test_df = pd.DataFrame(test_data, columns=['Text', 'Topic'])

# 向量化测试集
X_test_tfidf = vectorizer.transform(tqdm(test_df['Text'], desc="Vectorizing test data"))

# 预测测试集
y_test_pred = model.predict(X_test_tfidf)

# 评估测试集
test_accuracy = accuracy_score(test_df['Topic'], y_test_pred)
test_error_rate = 1 - test_accuracy
test_classification_rep = classification_report(test_df['Topic'], y_test_pred, digits=4)

# 保存结果到文本文件
with open(output_txt_path, 'w', encoding='utf-8') as output_file:
    output_file.write("Classification Report:\n")
    output_file.write(test_classification_rep)
    output_file.write("\n")
    output_file.write(f"Accuracy Score: {test_accuracy:.4f}\n")
    output_file.write(f"Error Rate: {test_error_rate:.4f}\n\n")
    output_file.write("Predictions:\n")
    for text, true_label, pred_label in zip(test_df['Text'], test_df['Topic'], y_test_pred):
        output_file.write(f"Text: {text}\nTrue Label: {true_label}\nPredicted Label: {pred_label}\n\n")

# 保存训练DataFrame为CSV文件
train_df.to_csv(output_csv_path, index=False)

print(f"CSV file saved to {output_csv_path}")
print(f"Classification results saved to {output_txt_path}")

print(f"Classification results on test set saved to 'classification_results_10000.txt'")
