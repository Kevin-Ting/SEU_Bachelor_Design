from flask import Flask, request, jsonify
import json
import onnxruntime
from transformers import AutoTokenizer
import torch
import numpy as np

app = Flask(__name__)

# 加载模型
ort_session = onnxruntime.InferenceSession("best_model.onnx")

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained("./train/my_finetuned_model")

# 接收JSON格式POST请求
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    processed_data = []

    # 遍历数据，对每个条目进行处理
    for item in data:
        # 获取text、target和type
        text = item["Text"]
        target = item["Target 1"]  # 将 "Target 1" 改为 "Target"
        type_ = item["Type"]      # 添加 "Type"

        # 对文本进行编码
        encoded_input = tokenizer(
            text,
            target,
            type_,
            padding='max_length',
            truncation=True,
            return_tensors="pt",
            max_length=512
        )

        # 构造预处理后的条目
        processed_data.append(encoded_input)

    # 对处理后的数据进行预处理
    input_ids = torch.cat([item["input_ids"] for item in processed_data], dim=0)
    attention_masks = torch.cat([item["attention_mask"] for item in processed_data], dim=0)

    # 推理
    ort_inputs = {
        ort_session.get_inputs()[0].name: input_ids.cpu().numpy(),
        ort_session.get_inputs()[1].name: attention_masks.cpu().numpy()
    }
    logits = ort_session.run(None, ort_inputs)[0]
    predicted_labels = np.argmax(logits, axis=1)

    # 构造响应，将文本解码为中文字符
    response = [{"Text": item["Text"],
                 "Target 1": item["Target 1"],
                 "Type": item["Type"],
                 "Stance": "支持" if label == 0 else "中立" if label == 1 else "反对"}
                for item, label in zip(data, predicted_labels)]

    # 返回 JSON 响应
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
