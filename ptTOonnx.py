import torch
from transformers import AutoModelForSequenceClassification

# 加载PyTorch模型
model = AutoModelForSequenceClassification.from_pretrained("./train/my_finetuned_model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 创建一个虚拟输入
max_seq_length = 512
dummy_input = (torch.zeros(1, max_seq_length).long().to(device),
               torch.zeros(1, max_seq_length).long().to(device))

# 导出为ONNX格式
input_names = ["input_ids", "attention_mask"]
output_names = ["output"]

torch.onnx.export(model,
                  dummy_input,
                  "best_model.onnx",
                  input_names=input_names,
                  output_names=output_names,
                  dynamic_axes={"input_ids": {0: "batch_size"},
                                "attention_mask": {0: "batch_size"},
                                "output": {0: "batch_size"}})

print("ONNX模型已生成：best_model.onnx")
