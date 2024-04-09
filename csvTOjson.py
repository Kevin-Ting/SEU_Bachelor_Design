import csv
import json

csv_file = './c_stance_dataset/subtaskA/raw_val_all_onecol.csv'
json_file = 'output.json'

data = []
with open(csv_file, 'r', encoding='utf-8-sig') as file:  # 使用 utf-8-sig 来去除 UTF-8-BOM
    csv_reader = csv.DictReader(file)
    for index, row in enumerate(csv_reader):
        if index < 100:
            data.append(row)
        else:
            break

# 将数据保存为 JSON 文件
with open(json_file, 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

print("CSV 文件的前100行数据已转换为 JSON 格式，并保存到", json_file)
