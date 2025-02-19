#coding=utf-8
import sys
import os
import json
import re
import time

# real id and names
id_name_dict = {}
name_id_dict = {}
# real id and virtual ids
id_dict = {}
# read name and virtual names
name_dict = {}
# real id and interests
interests_dict = {}

file_path = "../../SAGraph/dataset/"
files = os.listdir(file_path)
for filename in files:
    filename = file_path + filename
    if filename.find("profile") != -1:
        print(filename)
        with open(filename, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for real_id in data:
                real_name = data[real_id]["user_name"]
                if real_id not in id_name_dict:
                    id_name_dict.setdefault(real_id, real_name)
                    name_id_dict.setdefault(real_name, real_id)
                else:
                    id_name_dict[real_id] = real_name
    if filename.find("interaction") != -1:
        with open(filename, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for real_id in data:
                if real_id not in id_name_dict:
                    id_name_dict.setdefault(real_id, "user_%s" % real_id)
                for interaction in data[real_id]:
                    if str(interaction["interact_id"]) not in id_name_dict:
                        id_name_dict.setdefault(str(interaction["interact_id"]), "user_%s" % interaction["interact_id"])
    if filename.find("interests") != -1:
        for line in open(filename, "r", encoding="utf-8"):
            line = line.strip()
            data = json.loads(line)
            infer_answer = data["infer_answer"]
            infer_answer = infer_answer.replace("```json", "")
            infer_answer = infer_answer.replace("```", "")
            infer_answer = infer_answer.replace("\n", "")
            try:
                infer_answer = json.loads(infer_answer)
                user_id = infer_answer["user_id"]
                interests = infer_answer["user_interests"]
                if user_id not in interests_dict:
                    interests_dict.setdefault(user_id, interests)
                else:
                    continue
            except:
                continue
                
print(len(id_name_dict))

virtual_id = 1
for real_id in id_name_dict:
    virtual_name = "user_%s" % virtual_id
    id_dict.setdefault(real_id, virtual_id)
    name_dict.setdefault(id_name_dict[real_id], virtual_name)
    virtual_id += 1

with open('id_name_dict.json', 'w', encoding='utf-8') as file:
    json.dump(id_name_dict, file, ensure_ascii=False, indent=4)

with open('id_dict.json', 'w', encoding='utf-8') as file:
    json.dump(id_dict, file, ensure_ascii=False, indent=4)

with open('name_dict.json', 'w', encoding='utf-8') as file:
    json.dump(name_dict, file, ensure_ascii=False, indent=4)

def anon_text(input_str):
    # anon name
    if input_str.find("@") != -1:
        pattern = r'@(\w+)(?=["\s<])'
        matches = re.findall(pattern, input_str)
        for match in matches:
            if match in name_dict:
                input_str = input_str.replace(match, name_dict[match])
    # anon phone
    phone_pattern = r'1[3-9]\d{9}'
    input_str = re.sub(phone_pattern, '[PHONE]', input_str)
    return input_str


for filename in files:
    filename = file_path + filename
    profile_data = {}
    interaction_data = {}
    if filename.find("profile") != -1:
        with open(filename, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for real_id in data:
                items = {}
                virtual_id = id_dict[real_id]
                virtual_name = name_dict[id_name_dict[real_id]]
                interests = []
                if real_id in interests_dict:
                    interests = interests_dict[real_id]
                items["user_id"] = virtual_id
                items["user_name"] = virtual_name
                items["user_followers"] = data[real_id]["user_followers"]
                items["user_friends"] = data[real_id]["user_friends"]
                items["user_interests"] = interests
                items["user_description"] = anon_text(data[real_id]["user_description"])
                profile_data.setdefault(virtual_id, items)
        with open(filename + ".anon", 'w', encoding='utf-8') as file:
            json.dump(profile_data, file, ensure_ascii=False, indent=4)
    if filename.find("interaction") != -1:
        with open(filename, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for real_id in data:
                new_interactions = []
                for interaction in data[real_id]:
                    interaction["interact_id"] = id_dict[str(interaction["interact_id"])]
                    interaction["text_raw"] = anon_text(interaction["text_raw"])
                    interaction["text_comment"] = anon_text(interaction["text_comment"])
                    new_interactions.append(interaction)
                interaction_data.setdefault(id_dict[real_id], new_interactions)
        with open(filename + ".anon", 'w', encoding='utf-8') as file:
            json.dump(interaction_data, file, ensure_ascii=False, indent=4)
    



