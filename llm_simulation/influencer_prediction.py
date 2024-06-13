#coding=utf-8
import sys
import openai
import time
import json
import random
import os
from retrying import retry
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from evaluate import ranking_evaluation
import itertools
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

openai.api_type = "open_ai"
openai.api_version=""
api_key_kimi = ""
api_base_kimi = ""
api_key_gpt = ""
api_base_gpt = ""

product = "spark_thinking"
# product = "electric_toothbrush"
# product = "intelligent_floor_scrubber"
# product = "ruby_face_cream"
# product = "abc_reading"
# product = "supor_boosted_showerhead"
static_file = "dataset/%s_r5.static.graph" % product
dynamic_file = "dataset/%s_r5.dynamic.graph" % product
feature_file = "dataset/%s.feature" % product

PRODUCT_DETAIL = {}
with open("dataset/product_info.jsonl", 'r', encoding='utf-8') as file:
        PRODUCT_DETAIL = json.load(file)

PROMPT_DICT = {
    "prompt4staticprofile": "每个标号后面是用户的个性签名，请以json列表的形式输出每个用户可能感兴趣的领域，id字段为标号，interests字段为领域，多个领域采用列表，输出中文。\n",
    "prompt4dynamicprofile": "每个标号后面是一个用户的评论，post为上文，reply为当前用户的回复，请以json列表的形式输出回复的每个用户可能感兴趣的领域，当前回复对上文的支持程度，以及和上文的相关性，id字段为标号，interests字段为领域，多个领域采用列表，support_score为支持程度，relative_score为相关性程度，输出中文，支持程度和相关性程度打分1-10。输出结果为可解析的json。\n",
    "prompt4behavior_stepbystep": "博主介绍：\n{blogger_profile}\n\n{user_count}个用户介绍： \n{followers_profile}\n\n产品介绍：\n{product_info}\n\n输出要求：每个用户的兴趣点在interest字段、支持度在support_score字段，浏览新产品{product_name}的介绍后，根据兴趣点和支持度模拟每个用户做出动作（忽略产品或评论产品）。\n1、动作包括忽略ignore或评论comment，输出在action字段；\n2、若动作为comment，则根据用户已知的兴趣点和支持度，来考虑用户对发布的{product_type}产品{product_name}是否有需求，根据需求来模拟用户的语气生成针对产品{product_name}的评论，输出在new_comment字段，15字内；\n3、若动作为comment，根据生成的评论预测每条评论购买{product_name}的倾向，输出在purchase_likelihood字段，打分范围：1（不可能购买）到10（很可能购买）；\n4、整体结果json格式输出，预测的action的个数和原有用户数保持一致，保留原有用户id，interest和support_score字段，请保证输出结果为可解析的json，如{\"2737421304\": {\"interests\": [\"家庭\", \"娱乐\"], \"support_score\": 7, \"action\": \"comment\", \"new_comment\": \"适合家庭亲子活动\", \"purchase_likelihood\": 7}。",
    "prompt4behavior_wo_profile": "用户历史评论：\n{followers_comments}\n\n产品介绍：\n{product_info}\n\n输出要求：浏览新产品{product_name}的介绍后，根据历史评论模拟每个用户做出动作（忽略产品或评论产品）。\n1、动作包括忽略ignore或评论comment，输出在action字段；\n2、若动作为comment，则根据用户历史评论，来考虑用户对发布的{product_type}产品{product_name}是否有需求，根据需求来模拟用户的语气生成针对产品{product_name}的评论，输出在new_comment字段，15字内；\n3、若动作为comment，根据生成的评论预测每条评论购买{product_name}的倾向，输出在purchase_likelihood字段，打分范围：1（不可能购买）到10（很可能购买）；\n4、整体结果json格式输出，预测的action的个数和原有用户数保持一致，保留原有用户id，请保证输出结果为可解析的json，如{\"2737421304\": {\"action\": \"comment\", \"new_comment\": \"适合家庭亲子活动\", \"purchase_likelihood\": 7}。",
    "prompt4behavior_wo_cot": "博主介绍：\n{blogger_profile}\n\n{user_count}个用户介绍： \n{followers_profile}\n\n产品介绍：\n{product_info}\n\n输出要求：对于每个用户，根据其兴趣点和支持度，在浏览新产品{product_name}介绍后进行模拟动作，包括忽略(ignore)或评论(comment)。动作信息记录在action字段。\n\n若动作为评论(comment)：\n- 根据用户已知的兴趣点和支持度，考虑用户对发布的{product_type}产品{product_name}是否有需求。\n- 模拟用户语气生成针对产品{product_name}的评论，评论内容记录在new_comment字段（15字内）。\n- 针对每条评论预测购买{product_name}的倾向，输出在purchase_likelihood字段，打分范围：1（不可能购买）到10（很可能购买）。\n\n整体结果以JSON格式输出，保持预测的动作个数与原有用户数一致，同时保留原有用户id、兴趣点(interests)和支持度(support_score)字段。如{\"2737421304\": {\"interests\": [\"家庭\", \"娱乐\"], \"support_score\": 7, \"action\": \"comment\", \"new_comment\": \"适合家庭亲子活动\", \"purchase_likelihood\": 7}。"
    
}


import re

def remove_angle_brackets(input_string):
    # 使用正则表达式匹配尖括号内的内容
    pattern = re.compile(r'<.*?>')
    # 使用 sub 方法将匹配到的内容替换为空字符串
    result = re.sub(pattern, '', input_string)
    if len(result.strip()) == 0:
        result = input_string
    return result

@retry(stop_max_attempt_number=10, wait_fixed=1000)
def get_response(text, model="gpt-3.5-turbo"):
    messages = []
    messages = [{"role": "user", "content": text}]
    if model == "gpt-3.5-turbo" or model == "gpt-4":
        openai.api_base = api_base_gpt
        openai.api_key = api_key_gpt
    else:
        openai.api_base = api_base_kimi
        openai.api_key = api_key_kimi
    result = openai.ChatCompletion.create(model=model, messages=messages, temperature=0.95)
    response = result['choices'][0]['message']['content']
    return response

# 加载大V
def load_seeds(round, static_data):
    global product
    seed_set = []
    for i in range(round+1):
        filename = "dataset/%s_r%s_uid.txt" % (product, i)
        for line in open(filename, "r", encoding="utf-8"):
            line = line.strip("\n")
            fields = line.split("\t")
            if len(fields) > 0:
                if str(fields[0]) not in seed_set and static_data[fields[0]]["user_followers"] > 100000:
                    seed_set.append(str(fields[0]))
        print(len(seed_set))
    return set(seed_set)

# 加载静态信息：用户ID，静态profile
def load_static_profile():
    global static_file
    with open(static_file, 'r', encoding='utf-8') as file:
        static_data = json.load(file)
    return static_data

# 加载动态信息：用户ID，历史评论等
def load_dynamic_profile():
    global dynamic_file
    with open(dynamic_file, 'r', encoding='utf-8') as file:
        dynamic_data = json.load(file)
    return dynamic_data

def load_features():
    global feature_file
    with open(feature_file, 'r', encoding='utf-8') as file:
        feature_data = json.load(file)
    return feature_data

# 所有大V采样至少20个用户，输出到文件user_id\tlabel 0:v,1:nv
def sample_user(static_profile, dynamic_profile, seed_set):
    v0_users = {}
    sample_f = open("./%s/step1.sample.user.res" % product, "w+", encoding="utf-8")
    for user_id in static_profile:
        if user_id not in seed_set:
            continue
        # if user_id in dynamic_profile and static_profile[user_id]["user_followers"] > 100000:
        if user_id in dynamic_profile and static_profile[user_id]["user_followers"] > 100000:
            interact_list = list(set([uttr["interact_id"] for uttr in dynamic_profile[user_id]]))
            reserved_list = []
            backup_list = []
            for interact_id in interact_list:
                if static_profile[str(interact_id)]["user_followers"] > 100000:
                    reserved_list.append(str(interact_id))
                else:
                    backup_list.append(str(interact_id))
            backup_list = random.sample(backup_list, min(len(backup_list), 20))
            if len(reserved_list) > 0:
                reserved_list.extend(backup_list)
            else:
                reserved_list = backup_list
            if user_id in v0_users:
                reserved_list = v0_users[user_id].split(":")
            else:
                print("not found %s" % user_id)
            reserved_list = random.sample(reserved_list, min(len(reserved_list), 20))
            sample_f.write("%s\t%s\n" % (user_id, ":".join(reserved_list)))
    return

# 选择influencer/随机or按模型选，输出到文件
def influencer_pre_selection(gt_ids, static_file, feature_dict):
    influencer_candidate = []
    for line in open("./%s/step1.sample.user.res" % product, "r", encoding="utf-8"):
        line = line.strip("\n")
        fields = line.split("\t")
        influencer_candidate.append(fields[0])
    sample_influencer_by_random = {user_id: static_file[user_id]["user_followers"] for user_id in influencer_candidate}
    sorted_sample_influencer_by_random = dict(sorted(sample_influencer_by_random.items(), key=lambda item: item[1], reverse=True))
    sample_influencer_by_random = list(itertools.islice(sorted_sample_influencer_by_random.keys(), 200))
    sample_influencer_by_random = [key + "\t" + static_file[key]["user_name"] + "\t" + str(static_file[key]["user_followers"]) for key in sample_influencer_by_random]
    filename = '../models/random_forest_model.joblib'
    rf_classifier = joblib.load(filename)
    feature_list = [feature_dict[user_id] for user_id in influencer_candidate]
    scores = rf_classifier.predict_proba(feature_list)
    scores = [score[1] for score in scores]
    pre_selection_scores = {influencer_candidate[i]: scores[i] for i in range(len(influencer_candidate))}
    pre_selection_scores = dict(sorted(pre_selection_scores.items(), key=lambda item: item[1], reverse=True))
    pre_selection_scores = list(pre_selection_scores.items())
    sample_influencer_by_model = [key + "\t" + static_file[key]["user_name"] + "\t" + str(value) for key, value in pre_selection_scores[0:200]]
    pre_selection_f1 = open("./%s/step2.influencer.pre.selection.by.sample" % product, "w+", encoding="utf-8")
    pre_selection_f1.write("\n".join(sample_influencer_by_random))
    pre_selection_f1.close()
    pre_selection_f2 = open("./%s/step2.influencer.pre.selection.by.model" % product, "w+", encoding="utf-8")
    pre_selection_f2.write("\n".join(sample_influencer_by_model))
    pre_selection_f2.close()
    return

def request_score(filename, data_map, prompt, request_batch_size):
    static_f = open(filename, "a+", encoding="utf-8")
    data_list = list(data_map.items())
    for i in range(int(len(data_list)/request_batch_size)+1):
        user_list = [item[0] for item in data_list[i*request_batch_size: (i+1)*request_batch_size]]
        description_list = [item[1] for item in data_list[i*request_batch_size: (i+1)*request_batch_size]]
        numbered_list = [(i + 1, item) for i, item in enumerate(description_list)]
        request_string = '\n'.join([f'{index}: {value}' for index, value in numbered_list])
        result_dict = {}
        try:
            response = get_response(prompt + request_string, model="kimi")
        except:
            try:
                response = get_response(prompt + request_string, model="kimi")
            except:
                try:
                    response = get_response(prompt + request_string, model="kimi")
                except:
                    print("request_score failed idx=%s" % i)
        static_f.write(json.dumps(user_list, ensure_ascii=False))
        static_f.write(response)
        static_f.write("\n")
        print("request_score process idx=%s" % i)
    static_f.close()

# 所有用户生成profile，输出到文件
def follower_profile_reasoning(static_file, request_type):
    #static profile/dynamic profile
    # user_id, user_gender, user_followers, user_friends, user_description
    # comments areas / support score / relative score
    interaction_map = {}
    existed_static_map = {}
    existed_dynamic_map = {}
    for line in open("./%s/step3.follower.profile.reasoning.dynamic" % product, "r", encoding="utf-8"):
        json_tokens = json.loads(line.strip())
        for hybrid_user in json_tokens:
            user_id = hybrid_user.split(":")[0]
            if user_id not in existed_dynamic_map:
                existed_dynamic_map.setdefault(user_id, 1)
    for line in open("./%s/step1.sample.user.res" % product, "r", encoding="utf-8"):
        line = line.strip("\n")
        fields = line.split("\t")
        interact_user = fields[1].split(":")
        interaction_map.setdefault(str(fields[0]), interact_user)
    static_map = {user_id: static_file[user_id]["user_description"] for user_id in interaction_map}
    print("request size for static: %s" % len(static_map))
    # request_score("./%s/step3.follower.profile.reasoning.static" % product, static_map, PROMPT_DICT["prompt4staticprofile"], 10)
    # return
    idx = 0
    for line in open("./%s/step3.follower.profile.reasoning.static" % product, "r", encoding="utf-8"):
        line = line.strip()
        json_tokens = json.loads(line)
        idx += 1
        for user_id in json_tokens:
            existed_static_map.setdefault(user_id, json_tokens[user_id])
    static_map = existed_static_map
    dynamic_map = {}
    for user_id in interaction_map:
        existed_users = interaction_map[user_id]
        print(existed_users)
        interact_dict = {str(uttr["interact_id"]): "post: " + remove_angle_brackets(uttr["text_raw"]) + " reply: " + remove_angle_brackets(uttr["text_comment"]) for uttr in
                     dynamic_profile[user_id] if str(uttr["interact_id"]) in existed_users}
        for interact_id in interact_dict:
            new_user_id = user_id + ":" + interact_id
            comments = interact_dict[interact_id]
            dynamic_map.setdefault(new_user_id, comments)
    print("request size for dynamic: %s" % len(dynamic_map))
    # request_score("./%s/step3.follower.profile.reasoning.dynamic" % product, dynamic_map, PROMPT_DICT["prompt4dynamicprofile"], 10)
    # return
    for line in open("./%s/step3.follower.profile.reasoning.dynamic" % product, 'r', encoding='utf-8'):
        line = line.strip()
        json_tokens = json.loads(line)
        for hybrid_user in json_tokens:
            if hybrid_user in dynamic_map and type(dynamic_map[hybrid_user]) == str:
                comment = dynamic_map[hybrid_user].split("reply:")[1]
                if type(json_tokens[hybrid_user]) == str:
                    dynamic_map[hybrid_user] = json.loads(json_tokens[hybrid_user])
                else:
                    dynamic_map[hybrid_user] = json_tokens[hybrid_user]
                
                dynamic_map[hybrid_user].setdefault("comment", comment)
    new_dynamic_map = {}
    for hybrid_user in dynamic_map:
        [user_id, interact_id] = hybrid_user.split(":")
        if type(dynamic_map[hybrid_user]) == str:
            continue
        interests = dynamic_map[hybrid_user]["interests"]
        relative_score = dynamic_map[hybrid_user]["relative_score"]
        supports_score = dynamic_map[hybrid_user]["support_score"]
        comment = dynamic_map[hybrid_user]["comment"]
        if user_id not in new_dynamic_map:
            item_interests = {interact_id: interests}
            item_supports = {interact_id: supports_score}
            item_relatives = {interact_id: relative_score}
            item_comments = {interact_id: comment}
            item = {}
            item["ids"] = [interact_id]
            item["interests"] = item_interests
            item["supports"] = item_supports
            item["relatives"] = item_relatives
            item["comments"] = item_comments
            new_dynamic_map.setdefault(user_id, item)
        else:
            item_ids = new_dynamic_map[user_id]["ids"]
            item_interests = new_dynamic_map[user_id]["interests"]
            item_supports = new_dynamic_map[user_id]["supports"]
            item_relatives = new_dynamic_map[user_id]["relatives"]
            item_comments = new_dynamic_map[user_id]["comments"]
            item_ids.append(interact_id)
            item_interests.setdefault(interact_id, interests)
            item_supports.setdefault(interact_id, supports_score)
            item_relatives.setdefault(interact_id, relative_score)
            item_comments.setdefault(interact_id, comment)
            new_dynamic_map[user_id]["ids"] = item_ids
            new_dynamic_map[user_id]["interests"] = item_interests
            new_dynamic_map[user_id]["supports"] = item_supports
            new_dynamic_map[user_id]["relatives"] = item_relatives
            new_dynamic_map[user_id]["comment"] = item_comments
    profile_dict = {}
    influencer_candidates = []
    for line in open("./%s/step2.influencer.pre.selection.by.%s" % (product, request_type), "r", encoding="utf-8"):
        influencer_candidates.append(line.strip().split("\t")[0])
    for user_id in influencer_candidates:
        item = {}
        item["user_name"] = static_file[user_id]["user_name"]
        item["user_gender"] = static_file[user_id]["user_gender"]
        item["user_followers"] = static_file[user_id]["user_followers"]
        item["user_friends"] = static_file[user_id]["user_friends"]
        item["interests"] = []
        item["followers_profiles"] = {}
        if user_id in static_map:
            item["interests"] = static_map[user_id]["interests"]
            if user_id in new_dynamic_map:
                followers_profiles = {}
                followers_interests = new_dynamic_map[user_id]["interests"]
                followers_supports = new_dynamic_map[user_id]["supports"]
                followers_relatives = new_dynamic_map[user_id]["relatives"]
                followers_comments = new_dynamic_map[user_id]["comments"]
                for interact_id in followers_interests:
                    interact_interests = followers_interests[interact_id]
                    interact_supports = followers_supports[interact_id]
                    interact_relatives = followers_relatives[interact_id]
                    interact_comments = followers_comments[interact_id]
                    interact_item = {}
                    interact_item["user_name"] = static_file[interact_id]["user_name"]
                    interact_item["user_gender"] = static_file[interact_id]["user_gender"]
                    interact_item["user_followers"] = static_file[interact_id]["user_followers"]
                    interact_item["user_friends"] = static_file[interact_id]["user_friends"]
                    interact_item["interests"] = interact_interests
                    interact_item["supports"] = interact_supports
                    interact_item["relatives"] = interact_relatives
                    interact_item["comments"] = interact_comments
                    followers_profiles.setdefault(interact_id, interact_item)
                item["followers_profiles"] = followers_profiles
        profile_dict.setdefault(user_id, item)
    profile_f = open("./%s/step3.follower.profile.%s" % (product, request_type), "w+", encoding="utf-8")
    profile_f.write(json.dumps(profile_dict, ensure_ascii=False, indent=2))
    profile_f.close()
    return

# 所有influencer candidates进行behavior预测/采纳profile和不采纳profile，输出到文件
def follower_behavior_prediction(request_type, prompt_type):
    result_dict = {}
    influencer_candidates = {}
    for line in open("./%s/step4.follower.behavior.prediction.%s" % (product, prompt_type), "r", encoding="utf-8"):
        line = line.strip()
        fields = line.split("\t")
        if fields[0] not in result_dict:
            result_dict.setdefault(fields[0], 1)
    result_f = open("./%s/step4.follower.behavior.prediction.%s" % (product, prompt_type), "a+", encoding="utf-8")
    with open("./%s/step3.follower.profile.%s" % (product, request_type), 'r', encoding='utf-8') as file:
        influencer_candidates = json.load(file)
    for user_id in influencer_candidates:
        if user_id in result_dict:
            continue
        item = {}
        item["user_name"] = influencer_candidates[user_id]["user_name"]
        item["user_gender"] = influencer_candidates[user_id]["user_name"]
        item["user_followers"] = influencer_candidates[user_id]["user_followers"]
        item["user_friends"] = influencer_candidates[user_id]["user_friends"]
        item["interests"] = influencer_candidates[user_id]["interests"]
        blogger_profile = json.dumps(item, ensure_ascii=False)
        user_count = len(influencer_candidates[user_id]["followers_profiles"])
        followers_profile = json.dumps(influencer_candidates[user_id]["followers_profiles"], ensure_ascii=False)
        followers_comments = {interact_id: influencer_candidates[user_id]["followers_profiles"][interact_id]["comments"] for interact_id in influencer_candidates[user_id]["followers_profiles"]}
        followers_comments = json.dumps(followers_comments, ensure_ascii=False)
        product_name = PRODUCT_DETAIL[product]["product_name"]
        product_type = PRODUCT_DETAIL[product]["product_type"]
        product_info = PRODUCT_DETAIL[product]["product_info"]
        prompt = PROMPT_DICT[prompt_type]
        prompt = prompt.replace("{blogger_profile}", blogger_profile)
        prompt = prompt.replace("{followers_profile}", followers_profile)
        prompt = prompt.replace("{followers_comments}", followers_comments)
        prompt = prompt.replace("{product_name}", product_name)
        prompt = prompt.replace("{product_type}", product_type)
        prompt = prompt.replace("{product_info}", product_info)
        prompt = prompt.replace("{user_count}", str(user_count))
        response = ""
        try:
            # response = get_response(prompt, model="gpt-4")
            response = get_response(prompt, model="kimi")
            response = response[response.find("{"):(response.rfind("}") + 1)]
            response = response.replace("\n", "")
        except:
            response = {}
        result_f.write("%s\t%s\n" % (user_id, response))
    result_f.close()
    return



if __name__ == "__main__":
    gt_ids = PRODUCT_DETAIL[product]["gt_ids"]
    static_profile = load_static_profile()
    print(len(static_profile))
    seed_set = load_seeds(5, static_profile)
    # print(seed_set)
    print(len(seed_set))
    dynamic_profile = load_dynamic_profile()
    feature_dict = load_features()
    gt_names = [static_profile[_id]["user_name"] for _id in gt_ids]

    # step1
    sample_user(static_profile, dynamic_profile, seed_set)

    # # step2
    influencer_pre_selection(gt_ids, static_profile, feature_dict)

    # # step3
    follower_profile_reasoning(static_profile, "sample")
    follower_profile_reasoning(static_profile, "model")

    # step4
    follower_behavior_prediction("model", "prompt4behavior_stepbystep")
    follower_behavior_prediction("model", "prompt4behavior_wo_profile")
    follower_behavior_prediction("model", "prompt4behavior_wo_cot")
