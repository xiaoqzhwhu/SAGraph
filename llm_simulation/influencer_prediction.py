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
feature_file = "dataset/%s.feature.v6" % product

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
    # print("messages")
    # print(messages)
    if model == "gpt-3.5-turbo" or model == "gpt-4":
        openai.api_base = api_base_gpt
        openai.api_key = api_key_gpt
    else:
        openai.api_base = api_base_kimi
        openai.api_key = api_key_kimi
    result = openai.ChatCompletion.create(model=model, messages=messages, temperature=0.95)
    response = result['choices'][0]['message']['content']
    print("response")
    print(response)
    return response


@retry(stop_max_attempt_number=10, wait_fixed=1000)
def get_response_gpt(text, model="gpt-3.5-turbo"):
    messages = []
    messages = [{"role": "user", "content": text}]
    # print("messages")
    # print(messages)
    result = openai.ChatCompletion.create(model=model, messages=messages, temperature=0.95)
    response = result['choices'][0]['message']['content']
    print("response")
    print(response)
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
    # for line in open("./%s.v0/step1.sample.user.res" % product, "r", encoding="utf-8"):
    #     line = line.strip()
    #     fields = line.split("\t")
    #     v0_users.setdefault(fields[0], fields[1])
    # print(v0_users)
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
                    # reserved_list.append(str(interact_id)+","+str(static_profile[str(interact_id)]["user_followers"]))
                    reserved_list.append(str(interact_id))
                    # reserved_list.append(str(interact_id)+"$$$"+str(interact_dict[str(interact_id)]))
                else:
                    # backup_list.append(str(interact_id)+","+str(static_profile[str(interact_id)]["user_followers"]))
                    backup_list.append(str(interact_id))
                    # backup_list.append(str(interact_id)+"$$$"+str(interact_dict[str(interact_id)]))
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
            # interact_list_detail = [uttr for uttr in dynamic_profile[user_id] if str(uttr["interact_id"]) in reserved_list]
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
    # sample_influencer_by_random[0:4] = gt_ids
    filename = '../models/random_forest_model_v9.joblib'
    rf_classifier = joblib.load(filename)
    feature_list = [feature_dict[user_id] for user_id in influencer_candidate]
    # random_feature_list = [feature_dict[user_id] for user_id in sample_influencer_by_random]
    # random_scores = rf_classifier.predict_proba(random_feature_list)
    # random_scores = [score[1] for score in random_scores]
    # random_selection_scores = {sample_influencer_by_random[i]: random_scores[i] for i in range(len(sample_influencer_by_random))}
    # random_selection_scores = dict(sorted(random_selection_scores.items(), key=lambda item: item[1], reverse=True))
    # random_selection_scores = list(random_selection_scores.items())
    # sample_influencer_by_random = [key + "\t" + static_file[key]["user_name"] + "\t" + str(value) for key, value in random_selection_scores[0:200]]
    scores = rf_classifier.predict_proba(feature_list)
    scores = [score[1] for score in scores]
    pre_selection_scores = {influencer_candidate[i]: scores[i] for i in range(len(influencer_candidate))}
    pre_selection_scores = dict(sorted(pre_selection_scores.items(), key=lambda item: item[1], reverse=True))
    pre_selection_scores = list(pre_selection_scores.items())
    # print(pre_selection_scores)
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
        # if i != 1528:
        #     continue
        user_list = [item[0] for item in data_list[i*request_batch_size: (i+1)*request_batch_size]]
        description_list = [item[1] for item in data_list[i*request_batch_size: (i+1)*request_batch_size]]
        numbered_list = [(i + 1, item) for i, item in enumerate(description_list)]
        request_string = '\n'.join([f'{index}: {value}' for index, value in numbered_list])
        result_dict = {}
        try:
            response = get_response(prompt + request_string, model="kimi")
            # scores = json.loads(get_response(prompt + request_string, model="gpt-4"))
            # scores = json.loads(get_response(prompt + request_string, model="kimi"))
            # result_dict = {user_list[j]: scores[j] for j in range(len(user_list))}
        except:
            try:
                response = get_response(prompt + request_string, model="kimi")
                # scores = json.loads(get_response(prompt + request_string, model="kimi"))
                # result_dict = {user_list[j]: scores[j] for j in range(len(user_list))}
            except:
                try:
                    response = get_response(prompt + request_string, model="kimi")
                    # scores = json.loads(get_response(prompt + request_string, model="kimi"))
                    # result_dict = {user_list[j]: scores[j] for j in range(len(user_list))}
                except:
                    print("request_score failed idx=%s" % i)
        # static_f.write(json.dumps(result_dict, ensure_ascii=False))
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
    print("len of existed_dynamic_map: %s" % len(existed_dynamic_map))
    # sample_user_f = open("./%s/step1.sample.user.res.sample.interact.res" % product, "w+", encoding="utf-8")
    for line in open("./%s/step1.sample.user.res" % product, "r", encoding="utf-8"):
        line = line.strip("\n")
        fields = line.split("\t")
        interact_user = fields[1].split(":")
        # if len(interact_user) > 20:
        #     interact_user = random.sample(interact_user, 20)
        # sample_user_f.write("%s\t%s\n" % (str(fields[0]), ":".join(interact_user)))
        # if str(fields[0]) not in existed_dynamic_map:
        interaction_map.setdefault(str(fields[0]), interact_user)
            # print("request user %s for interaction map" % str(fields[0]))
    # sample_user_f.close()
    static_map = {user_id: static_file[user_id]["user_description"] for user_id in interaction_map}
    print("request size for static: %s" % len(static_map))
    # request_score("./%s/step3.follower.profile.reasoning.static" % product, static_map, PROMPT_DICT["prompt4staticprofile"], 10)
    # return
    idx = 0
    for line in open("./%s/step3.follower.profile.reasoning.static" % product, "r", encoding="utf-8"):
        line = line.strip()
        json_tokens = json.loads(line)
        idx += 1
        # print(idx)
        for user_id in json_tokens:
            # print(user_id)
            # print(json_tokens[user_id])
            existed_static_map.setdefault(user_id, json_tokens[user_id])
    # print(len(existed_static_map))
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
    # print(dynamic_map)
    print("request size for dynamic: %s" % len(dynamic_map))
    # request_score("./%s/step3.follower.profile.reasoning.dynamic" % product, dynamic_map, PROMPT_DICT["prompt4dynamicprofile"], 10)
    # return
    for line in open("./%s/step3.follower.profile.reasoning.dynamic" % product, 'r', encoding='utf-8'):
        line = line.strip()
        # print(line)
        json_tokens = json.loads(line)
        # print(json_tokens)
        # print(type(json_tokens))
        for hybrid_user in json_tokens:
            if hybrid_user in dynamic_map and type(dynamic_map[hybrid_user]) == str:
                # print("hybrid_user")
                # print(dynamic_map[hybrid_user])
                comment = dynamic_map[hybrid_user].split("reply:")[1]
                # print("comment")
                # print(comment)
                if type(json_tokens[hybrid_user]) == str:
                    # print(json_tokens[hybrid_user])
                    dynamic_map[hybrid_user] = json.loads(json_tokens[hybrid_user])
                else:
                    dynamic_map[hybrid_user] = json_tokens[hybrid_user]
                # print(dynamic_map[hybrid_user])
                
                dynamic_map[hybrid_user].setdefault("comment", comment)
    # print(dynamic_map)
    # with open("./%s/step3.follower.profile.reasoning.dynamic" % product, 'r', encoding='utf-8') as file:
    #     dynamic_map = json.load(file)
    new_dynamic_map = {}
    for hybrid_user in dynamic_map:
        [user_id, interact_id] = hybrid_user.split(":")
        # print(dynamic_map[hybrid_user])
        # print(type(dynamic_map[hybrid_user]))
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
        print(user_id)
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
        print(prompt)
        try:
            # response = get_response(prompt, model="gpt-4")
            response = get_response(prompt, model="kimi")
            response = response[response.find("{"):(response.rfind("}") + 1)]
            response = response.replace("\n", "")
            print(response)
        except:
            response = {}
        result_f.write("%s\t%s\n" % (user_id, response))
    result_f.close()
    return

# 每个influencer计算交互weight，输出到文件
def influencer_ranking(static_profile, dynamic_file, request_type, prompt_type, top_k, sample_k, ranking_policy="simulation"):
    name2id = {}
    whole_users = []
    before_ranking_list = []
    before_ranking_ids = []
    idx = 0
    ranking_dict = {}
    for line in open("./%s/step2.influencer.pre.selection.by.sample" % (product)):
        line = line.strip("\n")
        fields = line.split("\t")
        user_id = fields[0]
        user_name = static_profile[user_id]["user_name"]
        if user_name.find("官方") != -1 or user_name.find("平台") != -1:
            continue
        whole_users.append(user_name)
    for line in open("./%s/step2.influencer.pre.selection.by.%s" % (product, request_type)):
        line = line.strip("\n")
        fields = line.split("\t")
        user_id = fields[0]
        user_name = static_profile[user_id]["user_name"]
        if user_name.find("官方") != -1 or user_name.find("平台") != -1:
            continue
        before_ranking_ids.append(user_id)
        before_ranking_list.append(user_name)
        # whole_users.append(user_name)
    # for line in open("./%s/step4.model.follower.behavior.prediction" % product, "r", encoding="utf-8"):
    #     line = line.strip("\n")
    #     fields = line.split("\t")
    #     user_id = fields[0]
    #     user_name = static_profile[user_id]["user_name"]
    #     whole_users.append(user_name)
    behavior_result = {}
    for line in open("./%s/step4.follower.behavior.prediction.%s" % (product, prompt_type), "r", encoding="utf-8"):
        line = line.strip("\n")
        # print(line)
        fields = line.split("\t")
        # print(fields[1])
        if len(fields[1].strip()) == 0:
            continue
        behavior_result.setdefault(fields[0], fields[1])
    for user_id in before_ranking_ids:
        if user_id not in behavior_result:
            continue
        user_name = static_profile[user_id]["user_name"]
        if user_name not in name2id:
            name2id.setdefault(user_name, user_id)
        continue_flag = 0
        followers_count = static_profile[user_id]["user_followers"]
        response = behavior_result[user_id]
        response = response.replace("null", "0")
        response = response.replace("...", "")
        response = response.replace(" ", "")
        # print(response)
        if len(response.split("：")) > 1 and response.find("：") < 10:
            response = "".join(response.split("：")[1:])
        if response[0:5] == "输出结果：":
            response = response[5:]
        if response[0:3] == "输出：":
            response = response[3:]
        try:
            comments = json.loads(response)
        except:
            try:
                comments = json.loads("[" + response + "]")
            except:
                comments = json.loads(response + "}")
        avg_score = 0
        effective_count = 0
        
        if type(comments) == list:
            comments = comments[0]
            comments_ids = random.sample(list(comments), min(sample_k, len(comments)))
            # print("sample k: %s len of comments: %s len of ids: %s" % (sample_k, len(comments), len(comments_ids)))
            # print(len(comments_ids))
            for interact_comment in comments:
                if interact_comment not in comments_ids:
                    continue
                # print("interact_comment")
                # print(interact_comment)
                comment = comments[interact_comment]
                # print("comment")
                # print(comment)
                if "action" in comment and comment["action"] == "comment" and "purchase_likelihood" in comment:
                    # print(comment["purchase_likelihood"])
                    if comment["purchase_likelihood"] == "" or comment["purchase_likelihood"] is None:
                        comment["purchase_likelihood"] = 0
                    if int(comment["purchase_likelihood"]) > 0:
                        effective_count += 1
                    avg_score += int(comment["purchase_likelihood"])
        else:
            comments_ids = random.sample(list(comments), min(sample_k, len(comments)))
            # print(len(comments_ids))
            # print("sample k: %s len of comments: %s len of ids: %s" % (sample_k, len(comments), len(comments_ids)))
            # print(comments)
            for interact_id in comments:
                if interact_id not in comments_ids:
                    continue
                # print(interact_id)
                comment = comments[interact_id]
                # print("comment")
                # print(comment)
                # if "purchase_likelihood" in comment:
                if "action" in comment and comment["action"] == "comment" and "purchase_likelihood" in comment:
                    # print(comment["purchase_likelihood"])
                    if comment["purchase_likelihood"] == "":
                        comment["purchase_likelihood"] = 0
                    if int(comment["purchase_likelihood"]) > 0:
                        effective_count += 1
                    avg_score += int(comment["purchase_likelihood"])
        # avg_score /= len(comments)
        # if effective_count > 0:
        #     avg_score /= effective_count
        # if len(list(uttr["text_raw"] for uttr in dynamic_file[user_id] if uttr["interact_type"] == "comment"))/len(list(set(uttr["text_raw"] for uttr in dynamic_file[user_id]))) < 10:
        #     continue
        # if idx > 0.5 * len(before_ranking_list):
        #     break
        if idx >= top_k:
            break
        if len(comments) > 0:
            if ranking_policy == "simulation":
                avg_score = avg_score * (len(list(uttr["text_raw"] for uttr in dynamic_file[user_id] if uttr["interact_type"] == "comment"))/(min(sample_k, len(comments))*len(list(set(uttr["text_raw"] for uttr in dynamic_file[user_id])))))
            else:
                avg_score = avg_score/min(sample_k, len(comments))
                # avg_score = (len(list(uttr["text_raw"] for uttr in dynamic_file[user_id] if uttr["interact_type"] == "comment"))/len(list(set(uttr["text_raw"] for uttr in dynamic_file[user_id]))))
        # print("用户名：%s\t平均分：%s\t互动数：%s\t评论总数：%s\t模拟评论数：%s\t有效评论数：%s\t无效评论数：%s\t发帖数：%s" % (user_name, avg_score, len(dynamic_file[user_id]), len(list(uttr["text_raw"] for uttr in dynamic_file[user_id] if uttr["interact_type"] == "comment")), len(comments), effective_count, len(comments)-effective_count, len(list(set(uttr["text_raw"] for uttr in dynamic_file[user_id])))))
        if user_name not in ranking_dict:
            ranking_dict.setdefault(user_name, avg_score)
        idx += 1
    #print(ranking_dict)
    print("len of ranking dict: %s" % len(ranking_dict))
    sorted_dict = dict(sorted(ranking_dict.items(), key=lambda item: item[1], reverse=True))
    # print(sorted_dict)
    # print(before_ranking_list)
    after_ranking_list = list(itertools.islice(sorted_dict.keys(), 200))
    ranking_ids = [name2id[user_name] for user_name in after_ranking_list]
    # sim_f = open("simulation.txt", "a+", encoding="utf-8")
    # sim_f.write("%s\t%s\n" % (product, json.dumps(ranking_ids, ensure_ascii=False)))
    # sim_f.close()
    return whole_users, before_ranking_list, after_ranking_list

def overall_evaluation(gt_names, predicted_items):
    print(gt_names)
    print(predicted_items)
    # print("after selection")
    # precision_1, recall_1, ndcg_1 = ranking_evaluation(gt_names, selection_names, 1)
    # precision_2, recall_2, ndcg_2 = ranking_evaluation(gt_names, selection_names, 2)
    # precision_5, recall_5, ndcg_5 = ranking_evaluation(gt_names, selection_names, 5)
    # precision_10, recall_10, ndcg_10 = ranking_evaluation(gt_names, selection_names, 10)
    # print("P@1=%.3f, R@1=%.3f, NDCG@1=%.3f" % (precision_1, recall_1, ndcg_1))
    # print("P@2=%.3f, R@2=%.3f, NDCG@2=%.3f" % (precision_2, recall_2, ndcg_2))
    # print("P@5=%.3f, R@5=%.3f, NDCG@5=%.3f" % (precision_5, recall_5, ndcg_5))
    # print("P@10=%.3f, R@10=%.3f, NDCG@10=%.3f" % (precision_10, recall_10, ndcg_10))
    # print("after ranking")
    precision_1, recall_1, ndcg_1 = ranking_evaluation(gt_names, predicted_items, 1)
    precision_2, recall_2, ndcg_2 = ranking_evaluation(gt_names, predicted_items, 2)
    precision_5, recall_5, ndcg_5 = ranking_evaluation(gt_names, predicted_items, 5)
    precision_10, recall_10, ndcg_10 = ranking_evaluation(gt_names, predicted_items, 10)
    print("P@1=%.3f, R@1=%.3f, NDCG@1=%.3f" % (precision_1, recall_1, ndcg_1))
    print("P@2=%.3f, R@2=%.3f, NDCG@2=%.3f" % (precision_2, recall_2, ndcg_2))
    print("P@5=%.3f, R@5=%.3f, NDCG@5=%.3f" % (precision_5, recall_5, ndcg_5))
    print("P@10=%.3f, R@10=%.3f, NDCG@10=%.3f" % (precision_10, recall_10, ndcg_10))
    return precision_5, precision_10, recall_5, recall_10, ndcg_5, ndcg_10
    # return

def display_domain_distribution(display_users):
    print("display users")
    print(display_users)
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    profile_data = {}
    with open("./%s/step3.follower.profile" % product, "r", encoding='utf-8') as file:
        profile_data = json.load(file)
    domain_dict = {}
    for user_id in profile_data:
        user_name = profile_data[user_id]["user_name"]
        if user_name in display_users:
            interests = profile_data[user_id]["interests"]
            for interest in interests:
                if interest == "旅游":
                    interest = "旅行"
                if interest not in domain_dict:
                    domain_dict.setdefault(interest, 1)
                else:
                    domain_dict[interest] += 1
    sorted_dict = dict(sorted(domain_dict.items(), key=lambda item: item[1], reverse=True))
    labels = []
    sizes = []
    total_cnt = 0
    idx = 0
    for domain in sorted_dict:
        if idx == 10:
            break
        cnt = sorted_dict[domain]
        print(domain)
        print(cnt)
        labels.append(domain)
        sizes.append(cnt)
        total_cnt += cnt
        idx += 1
    print(sizes)
    print(total_cnt)
    sizes = [int((i/total_cnt)*100) for i in sizes]
    print(labels)
    print(sizes)
    # 饼图的颜色
    # 美食 gold
    # 生活 lightcoral
    # 旅行 lightskyblue
    # 摄影 lightgreen
    # 育儿 orange
    # 写作 lightpink
    # 商务 cyan
    # 生活分享 lavender
    # 绘本 tomato
    # 母婴 lightsalmon
    # 好物分享 darkseagreen
    # 博主 darkcyan
    # 购物 skyblue
    # 二胎育儿 pink
    # 育婴 wheat 
    # 科学育儿 lightsteelblue
    # 媒体 coral
    # 教育 sandybrown
    # 亲子沟通 tab:green
    # 健康 tab:orange
    # 德国品牌 tab:blue
    # 地球探索 tab:cyan
    #['美食', '生活', '旅行', '摄影', '育儿', '写作', '商务', '生活分享', '绘本', '母婴']
    #['生活分享', '育儿', '摄影', '美食', '好物分享', '博主', '购物', '二胎育儿', '育婴', '科学育儿']
    #['育儿', '生活分享', '摄影', '媒体', '教育', '亲子沟通', '健康', '德国品牌', '旅游', '地球探索']
    colors_a = ['gold', 'lightcoral', 'lightskyblue', 'lightgreen', 'orange', 'lightpink', 'cyan', 'lavender', 'lightsalmon', 'lightblue']
    colors_b = ['lavender', 'orange', 'lightgreen', 'gold', 'darkseagreen', 'darkcyan', 'skyblue', 'pink', 'wheat', 'lightsteelblue']
    colors_c = ['orange', 'lavender', 'lightgreen', 'coral', 'sandybrown', 'tab:green', 'tab:orange', 'tab:blue', 'lightskyblue', 'tab:cyan']
    # 突出显示某个类别，可选
    explode = (0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    plt.rcParams['font.size'] = 12
    fig, ax = plt.subplots(figsize=(10, 10))
    # 绘制饼图
    ax.pie(sizes, explode=explode, labels=labels, colors=colors_a, autopct='%1.1f%%', shadow=True, startangle=140, radius=2)
    # 设置饼图的标题
    # plt.title("          领域分布         ")
    # 显示图例
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    # 显示饼图
    plt.axis('equal')  # 保证饼图是圆形而不是椭圆形
    plt.tight_layout()
    plt.savefig('pie_chart_a.pdf', dpi=500)
    plt.show()


def plot_pr(x, y, labels):

    # 画出曲线
    plt.figure(figsize=(8, 6))
    print(len(y))
    print(y)

    plt.plot(x, y[0], label=labels[0])
    plt.plot(x, y[1], label=labels[1])
    plt.plot(x, y[2], label=labels[2])
    plt.plot(x, y[3], label=labels[3])
    plt.plot(x, y[4], label=labels[4])
    plt.plot(x, y[5], label=labels[5])

    # 添加图例
    plt.legend()

    # 添加标题和坐标轴标签
    plt.title('Simulation Scale Comparison')
    plt.xlabel('sample-k')
    plt.ylabel('metrics')

    # 显示图形
    plt.show()

    


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
    # sample_user(static_profile, dynamic_profile, seed_set)

    # # step2
    # influencer_pre_selection(gt_ids, static_profile, feature_dict)

    # # step3
    # follower_profile_reasoning(static_profile, "sample")
    # follower_profile_reasoning(static_profile, "model")

    # step4
    follower_behavior_prediction("model", "prompt4behavior_stepbystep")
    # follower_behavior_prediction("model", "prompt4behavior_wo_profile")
    # follower_behavior_prediction("model", "prompt4behavior_wo_cot")


    # step5
    # eval for main method
    top_k = 13
    sample_k = 20
    whole_users, selection_names, predicted_names = influencer_ranking(static_profile, dynamic_profile, "model", "prompt4behavior_stepbystep", top_k, sample_k)
    # print(gt_names)
    # print(selection_names)
    # print(predicted_names)
    print("Main Method: ")
    overall_evaluation(gt_names, predicted_names)
    
    print("Pre-Selection: ")
    overall_evaluation(gt_names, selection_names[:10])

    whole_users, selection_names, predicted_names = influencer_ranking(static_profile, dynamic_profile, "model", "prompt4behavior_stepbystep", top_k, sample_k, "otherwise")
    # print(gt_names)
    # print(selection_names)
    # print(predicted_names)
    print("wo Simulation: ")
    overall_evaluation(gt_names, predicted_names)

    # # step5
    # eval for wo pre-selection
    whole_users, selection_names, predicted_names = influencer_ranking(static_profile, dynamic_profile, "sample", "prompt4behavior_stepbystep", top_k, sample_k)
    # step6
    # print(gt_names)
    # print(selection_names)
    # print(predicted_names)
    print("wo Pre-selection: ")
    overall_evaluation(gt_names, predicted_names)

    # step5
    # eval for wo profile
    whole_users, selection_names, predicted_names = influencer_ranking(static_profile, dynamic_profile, "model", "prompt4behavior_wo_profile", top_k, sample_k)
    # step6
    # print(gt_names)
    # print(selection_names)
    # print(predicted_names)
    print("wo Profile: ")
    overall_evaluation(gt_names, predicted_names)

    # # # step5
    # # eval for wo cot
    whole_users, selection_names, predicted_names = influencer_ranking(static_profile, dynamic_profile, "model", "prompt4behavior_wo_cot", top_k, sample_k)
    # step6
    # print(gt_names)
    # print(selection_names)
    # print(predicted_names)
    print("wo CoT: ")
    overall_evaluation(gt_names, predicted_names)

    # # step7 analysis
    # # print(whole_users)
    # # print(selection_names)
    # # print(predicted_names)
    # # display_domain_distribution(whole_users[:200])
    # # display_domain_distribution(selection_names[:20])
    # # display_domain_distribution(predicted_names[:10])

    # pr curve
    # write_f = open("pr.txt", "w+", encoding="utf-8")
    # top_k = 10
    # sample_k = 20
    # x = np.arange(1, 51)
    # y = [[]]*6
    # p_5 = []
    # p_10 = []
    # r_5 = []
    # r_10 = []
    # g_5 = []
    # g_10 = []
    # labels = ["P@5", "P@10", "R@5", "R@10", "G@5", "G@10"]
    # for top_k in range(1, 51):
    #     precision_5_all, precision_10_all, recall_5_all, recall_10_all, ndcg_5_all, ndcg_10_all = 0, 0, 0, 0, 0, 0
    #     for k in range(10):

    #         whole_users, selection_names, predicted_names = influencer_ranking(static_profile, dynamic_profile, "model", "prompt4behavior_stepbystep", top_k, sample_k)
    
    #         # step6
    #         # print(gt_names)
    #         # print(selection_names)
    #         # print(predicted_names)
    #         #print("Main Method: ")
    #         precision_5, precision_10, recall_5, recall_10, ndcg_5, ndcg_10 = overall_evaluation(gt_names, predicted_names)
    #         precision_5_all += precision_5
    #         precision_10_all += precision_10
    #         recall_5_all += recall_5
    #         recall_10_all += recall_10
    #         ndcg_5_all += ndcg_5
    #         ndcg_10_all += ndcg_10
    #     write_f.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (top_k, precision_5_all/10, precision_10_all/10, recall_5_all/10, recall_10_all/10, ndcg_5_all/10, ndcg_10_all/10))
    #     p_5.append(precision_5_all/10)
    #     p_10.append(precision_10_all/10)
    #     r_5.append(recall_5_all/10)
    #     r_10.append(recall_10_all/10)
    #     g_5.append(ndcg_5_all/10)
    #     g_10.append(ndcg_10_all/10)
    # write_f.close()
    # plot_pr(x, [p_5, p_10, r_5, r_10, g_5, g_10], labels)


