import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt

# 设置日志格式
logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 添加控制台输出
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# 加载数据
data_dir = "./c_stance_dataset/subtaskA"
train_data = pd.read_csv(os.path.join(data_dir, "raw_train_all_onecol.csv"))
val_data = pd.read_csv(os.path.join(data_dir, "raw_val_all_onecol.csv"))
test_data = pd.read_csv(os.path.join(data_dir, "raw_test_all_onecol.csv"))

# 预处理数据
tokenizer = AutoTokenizer.from_pretrained('./hfl/chinese-xlnet-mid', max_length=512)

def preprocess_data(data):
    encoded_inputs = tokenizer(
        data["Text"].tolist(),
        data["Target 1"].tolist(),
        data["Type"].tolist(),
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512
    )
    labels = torch.tensor([0 if stance == "支持" else 1 if stance == "中立" else 2 for stance in data["Stance 1"].tolist()])
    return encoded_inputs, labels

train_inputs, train_labels = preprocess_data(train_data)
val_inputs, val_labels = preprocess_data(val_data)
test_inputs, test_labels = preprocess_data(test_data)

# 构建数据加载器
train_dataset = TensorDataset(train_inputs.input_ids, train_inputs.attention_mask, train_labels)
val_dataset = TensorDataset(val_inputs.input_ids, val_inputs.attention_mask, val_labels)
test_dataset = TensorDataset(test_inputs.input_ids, test_inputs.attention_mask, test_labels)

train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=24)
test_loader = DataLoader(test_dataset, batch_size=24)

# 定义模型
model = AutoModelForSequenceClassification.from_pretrained("./hfl/chinese-xlnet-mid", num_labels=3)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)

model = AutoModelForSequenceClassification.from_pretrained("./train/my_finetuned_model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

train_losses = []
val_losses = []

test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for batch in tqdm(test_loader, desc=f"Testing"):
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        test_loss += loss.item() * input_ids.size(0)

        _, predicted = torch.max(outputs.logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_loss /= len(test_loader.dataset)
test_accuracy = correct / total

logging.info(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# 保存损失记录
loss_data = {"train_loss": train_losses, "val_loss": val_losses}
loss_df = pd.DataFrame(loss_data)
loss_df.to_csv("losses.csv", index=False)

# 绘制曲线图
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('loss_curve.png')
plt.show()
