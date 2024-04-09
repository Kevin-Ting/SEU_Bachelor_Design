import json
import requests

# 读取测试数据文件
with open('./test_data.json', 'r', encoding='utf-8') as file:
    test_data = json.load(file)

# 构造请求头和URL
url = 'http://localhost:5001/predict'
headers = {'Content-Type': 'application/json'}

# 发送POST请求并获取响应
response = requests.post(url, json=test_data, headers=headers)

# 解码响应内容为中文字符
decoded_response = response.content.decode('unicode-escape')

# 打印解码后的响应内容到控制台
print(decoded_response)