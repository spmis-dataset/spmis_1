import os
import pandas as pd
import torch
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from tqdm import tqdm
from collections import defaultdict

# 定义文件夹路径
train_input_directory = './transcriptions_out_train'  # 替换为你的训练文件夹路径
test_input_directory = './transcriptions_out_8_28'  # 替换为你的测试文件夹路径
output_csv_path = 'bert_classify.csv'  # 替换为你想要的输出文件路径
output_txt_path = 'bert_5000_new.txt'  # 替换为你想要的输出文本文件路径

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

# 随机抽取5000条数据
train_df_sampled = train_df.sample(n=5000, random_state=42)

# 标签编码
label_encoder = LabelEncoder()
train_df_sampled['Topic'] = label_encoder.fit_transform(train_df_sampled['Topic'])

# 使用BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 编码训练数据
train_encodings = tokenizer(list(train_df_sampled['Text']), padding=True, truncation=True, max_length=512, return_tensors='pt')
train_labels = torch.tensor(train_df_sampled['Topic'].values)

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = TextDataset(train_encodings, train_labels)

# 加载BERT模型
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_))

# 设置训练参数
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="no",  # 禁用评估
)

# 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
)

# 训练模型
trainer.train()

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

# 标签编码
test_df['Topic'] = label_encoder.transform(test_df['Topic'])

# 编码测试数据
test_encodings = tokenizer(list(test_df['Text']), padding=True, truncation=True, max_length=512, return_tensors='pt')
test_labels = torch.tensor(test_df['Topic'].values)

test_dataset = TextDataset(test_encodings, test_labels)

# 预测测试集
predictions = trainer.predict(test_dataset)
y_pred = torch.argmax(torch.tensor(predictions.predictions), axis=1)

# 反向转换标签
y_test = label_encoder.inverse_transform(test_labels.numpy())
y_pred = label_encoder.inverse_transform(y_pred.numpy())

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
classification_rep = classification_report(y_test, y_pred, digits=4)

# 保存结果到文本文件
with open(output_txt_path, 'w', encoding='utf-8') as output_file:
    output_file.write("Classification Report:\n")
    output_file.write(classification_rep)
    output_file.write("\n")
    output_file.write(f"Accuracy Score: {accuracy:.4f}\n")
    output_file.write(f"Error Rate: {error_rate:.4f}\n\n")
error_count = defaultdict(int)

# 遍历预测值和实际标签，统计错误个数
for true_label, pred_label in zip(y_test, y_pred):
    if true_label != pred_label:
        error_count[true_label] += 1

# 将错误统计结果保存到文本文件
with open(output_txt_path, 'a', encoding='utf-8') as output_file:  # 使用 'a' 模式以追加内容
    output_file.write("Error Count by Topic:\n")
    for topic, count in error_count.items():
        output_file.write(f"Topic: {topic} - Errors: {count}\n")

print(f"Error count by topic saved to {output_txt_path}")

# 保存DataFrame为CSV文件
train_df_sampled.to_csv(output_csv_path, index=False)

print(f"CSV file saved to {output_csv_path}")
print(f"Classification results saved to {output_txt_path}")
# 保存DataFrame为CSV文件
train_df_sampled.to_csv(output_csv_path, index=False)

print(f"CSV file saved to {output_csv_path}")
print(f"Classification results saved to {output_txt_path}")

# 保存更大测试集的结果到文本文件
with open('classification_results_10000.txt', 'w', encoding='utf-8') as output_file:
    output_file.write("Classification Report on Larger Test Set:\n")
    output_file.write(classification_rep)
    output_file.write("\n")
    output_file.write(f"Accuracy Score on Larger Test Set: {accuracy:.4f}\n")
    output_file.write(f"Error Rate on Larger Test Set: {error_rate:.4f}\n\n")
    output_file.write("Predictions on Larger Test Set:\n")
    for text, true_label, pred_label in zip(test_df['Text'], test_df['Topic'], y_pred):
        output_file.write(f"Text: {text}\nTrue Label: {true_label}\nPredicted Label: {pred_label}\n\n")

print(f"Classification results on larger test set saved to 'larger_test_classification_results.txt'")
