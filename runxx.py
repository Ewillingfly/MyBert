import extract_feature
import json
bv = extract_feature.BertVector()
with open("data/test.json",'r',encoding='utf-8') as load_f:
    load_dict = json.load(load_f)
    for i in load_dict:
        print(i[0])
        print(type(bv.encode([i[0]])))