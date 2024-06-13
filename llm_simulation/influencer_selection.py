#coding=utf-8

import json
import random
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib



product = "spark_thinking"
gt_user = ["林小暖bella", "小土大橙子", "小兽睡睡", "机器人家"]
# product = "abc_reading"
# gt_user = ["周周anais", "苒妈镜镜Mirror", "大小仲育儿时光", "万物心选·Kids"]
# product = "electric_toothbrush"
# gt_user = ["牧宸", "我是HYK", "搞机小公主", "数码君", "飞扬发髻"]
# product = "ruby_face_cream"
# gt_user = ["小丁的备忘录哟", "差点失控了", "娜娜酱肉噗噗", "十三块五", "一只可爱珣", "吃掉小熊掌", "懒狐狸cindy"]
# product= "intelligent_floor_scrubber"
# gt_user = ["吴小杰WJie", "结衣衣君", "爱范儿", "驴立领"]
# product= "supor_boosted_showerhead"
# gt_user = ["月亮碎片kakera-", "甜甜甜宝_", "废话小梦", "坡诶佩"]


static_file = "dataset/%s_r5.static.graph" % product
dynamic_file = "dataset/%s_r5.dynamic.graph" % product
feature_file = "dataset/%s.feature" % product
train_file = "dataset/train.data"

def random_sampling(graph, n):
    return random.sample(graph.nodes(), n)

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

# 每个用户计算个人影响力
def calculate_persona_influence(static_data, dynamic_data):
    # 粉丝数、关注数、发帖数
    influence_dict = {}
    for user_id in static_data:
        followers_count = static_data[user_id]["user_followers"]
        friends_count = static_data[user_id]["user_friends"]
        posts_count = 0
        if user_id in dynamic_data:
            posts_count = len(set([i["text_raw"] for i in dynamic_data[user_id]]))
        influence_dict.setdefault(user_id, [followers_count, friends_count, posts_count])
    return influence_dict

# 每个用户计算互动频率
def calculate_interaction_frequency(static_data, dynamic_data):
    # 回复用户数、回复评论数、被回复用户数、被回复评论数、回复大V数、回复大V评论数、被回复大V数、被回复大V评论数；
    frequency_dict = {}
    frequency_by_dict = {}
    for user_id in dynamic_data:
        for item in dynamic_data[user_id]:
            interact_id = str(item["interact_id"])
            if interact_id not in frequency_by_dict:
                frequency_by_dict.setdefault(interact_id, {user_id: 1})
            else:
                if user_id in frequency_by_dict[interact_id]:
                    frequency_by_dict[interact_id][user_id] += 1
                else:
                    frequency_by_dict[interact_id].setdefault(user_id, 1)
    for user_id in static_data:
        interact_user_cnt = 0
        interact_comment_cnt = 0
        interact_vip_cnt = 0
        interact_vip_comment_cnt = 0
        interact_by_user_cnt = 0
        interact_by_comment_cnt = 0
        interact_by_vip_cnt = 0
        interact_by_vip_comment_cnt = 0
        if user_id in dynamic_data:
            interact_user_cnt = len(set([i["interact_id"] for i in dynamic_data[user_id]]))
            interact_comment_cnt = len(dynamic_data[user_id])
            interact_vip_cnt = len(set([i["interact_id"] for i in dynamic_data[user_id] if static_data[str(i["interact_id"])]["user_followers"] > 100000]))
            interact_vip_comment_cnt = len([i["interact_id"] for i in dynamic_data[user_id] if static_data[str(i["interact_id"])]["user_followers"] > 100000])
        if user_id in frequency_by_dict:
            interact_by_user_cnt = len(frequency_by_dict[user_id])
            interact_by_comment_cnt = sum([frequency_by_dict[user_id][u] for u in frequency_by_dict[user_id]])
            interact_by_vip_cnt = len([u for u in frequency_by_dict[user_id] if static_data[user_id]["user_followers"] > 100000])
            interact_by_vip_comment_cnt = sum([frequency_by_dict[user_id][u] for u in frequency_by_dict[user_id] if static_data[user_id]["user_followers"] > 100000])
        frequency_dict.setdefault(user_id, [interact_user_cnt, interact_comment_cnt, interact_vip_cnt, interact_vip_comment_cnt, interact_by_user_cnt, interact_by_comment_cnt, interact_by_vip_cnt, interact_by_vip_comment_cnt])
    return frequency_dict

# 每个用户计算信息传播量
def calculate_information_broadcast(static_data, dynamic_data):
    # 转发总数、评论总数、平均转发数、平均评论数
    broadcast_dict = {}
    for user_id in static_data:
        reposts_cnt = 0
        comments_cnt = 0
        avg_reposts_cnt = 0
        avg_comments_cnt = 0
        comments_list = []
        if user_id in dynamic_data:
            comments_list = dynamic_data[user_id]
            posts_count = len(set([i["text_raw"] for i in comments_list]))
            for comment in comments_list:
                if comment["interact_type"] == "comment":
                    comments_cnt += 1
                else:
                    reposts_cnt += 1
            avg_reposts_cnt = reposts_cnt / posts_count
            avg_comments_cnt = comments_cnt / posts_count
        broadcast_dict.setdefault(user_id, [reposts_cnt, avg_reposts_cnt, comments_cnt, avg_comments_cnt])
    return broadcast_dict

# 每个用户计算社交网络属性：度中心性、接近中心性、介数中心性
def calculate_social_network_centrality(static_data, dynamic_data):
    centrality_dict = {}
    G = nx.Graph()
    idx = 1
    vertex_dict = {}
    vertex2userdict = {}
    for user_id in static_data:
        vertex_dict.setdefault(user_id, idx)
        vertex2userdict.setdefault(idx, user_id)
        idx += 1
    # idx = 1
    idx = 1
    for user_id in dynamic_data:
        edges = []
        for item in dynamic_data[user_id]:
            from_vertex = vertex_dict[user_id]
            to_vertex = vertex_dict[str(item["interact_id"])]
            edges.append((from_vertex, to_vertex))
        G.add_edges_from(edges)
    #     idx += 1
    #     if idx % 100 == 0:
    #         print("load %s user done" % idx)
    # print("after load vertex")
    # 可视化图
    # #nx.draw(G, with_labels=True, font_weight='bold', node_color='skyblue', node_size=10, arrows=True)
    # pos = nx.spring_layout(G)
    # nx.draw(G, pos, font_weight='bold', node_color='skyblue', node_size=10, arrows=True)
    # plt.show()

    # 计算度中心性
    degree_centrality = nx.degree_centrality(G)
    # print("度中心性：", degree_centrality)
    # print("度中心性 done")

    num_samples = 10
    closeness_centralities = []
    betweenness_centralities = []

    for _ in range(num_samples):
        # 生成子图
        sampled_nodes = random_sampling(G, 5000)
        subgraph = G.subgraph(sampled_nodes)

        # 计算接近中心性
        closeness_centrality = nx.closeness_centrality(subgraph)
        closeness_centralities.append(closeness_centrality)
        betweenness_centrality = nx.betweenness_centrality(subgraph)
        betweenness_centralities.append(betweenness_centrality)

    # 计算平均接近中心性
    average_closeness_centrality = {}
    average_betweenness_centrality = {}
    for node in G.nodes():
        sum_score = 0
        between_score = 0
        for closeness in closeness_centralities:
            if node in closeness:
                sum_score += closeness[node]
        for closeness in betweenness_centralities:
            if node in closeness:
                between_score += closeness[node]
        average_closeness_centrality.setdefault(node, sum_score/num_samples)
        average_betweenness_centrality.setdefault(node, between_score/num_samples)
    # average_closeness_centrality = {node: sum(closeness[node] for closeness in closeness_centralities) / num_samples for node in G.nodes()}

    # print(average_closeness_centrality)

    # # 计算接近中心性
    # # closeness_centrality = nx.closeness_centrality(G)
    # # # closeness_centrality = nx.fast_closeness_centrality(G)
    # print("接近中心性：", average_closeness_centrality)
    # print("接近中心性 done")
    #
    # # 计算介数中心性
    # # betweenness_centrality = nx.betweenness_centrality(G)
    # print("介数中心性：", average_betweenness_centrality)
    # print("介数中心性 done")

    for user_id in static_data:
        vertex = vertex_dict[user_id]
        degree_score = 0
        closeness_score = 0
        between_score = 0
        if vertex in degree_centrality:
            degree_score = degree_centrality[vertex]
        if vertex in average_closeness_centrality:
            closeness_score = average_closeness_centrality[vertex]
        if vertex in average_betweenness_centrality:
            between_score = average_betweenness_centrality[vertex]
        centrality_dict.setdefault(user_id, [degree_score, closeness_score, between_score])
    # for user_id in centrality_dict:
    #     print("%s\t%s" % (user_id, centrality_dict[user_id]))
    return centrality_dict


def merge_feature(static_data, influence_dict, broadcast_dict, frequency_dict, centrality_dict):
    global product
    feature_dict = {}
    feature_max = [0] * 18
    train_data = []
    labels = []
    global gt_user
    seed_set = load_seeds(5, static_data)
    print(len(seed_set))
    for user_id in influence_dict:
        feature = influence_dict[user_id]
        feature.extend(broadcast_dict[user_id])
        feature.extend(frequency_dict[user_id])
        feature.extend(centrality_dict[user_id])
        # print(feature)
        for i in range(len(feature)):
            if feature[i] > feature_max[i]:
                feature_max[i] = feature[i]
        feature_dict.setdefault(user_id, feature)
    for user_id in feature_dict:
        user_name = static_data[user_id]["user_name"]
        label = 0
        if user_name in gt_user:
            print("find user")
            label = 1
        for i in range(len(feature_dict[user_id])):
            # print(i)
            # print(feature_max)
            # print(feature_dict[user_id])
            if feature_max[i] > 0:
                feature_dict[user_id][i] /= feature_max[i]
        if user_id in seed_set:
            print(user_id)
            train_data.append(feature_dict[user_id])
            labels.append(label)
        # if label == 1:
        #     train_data.append(feature_dict[user_id])
        #     labels.append(label)
        # elif len(train_data) >= 4 and len(train_data) < 400 and static_data[user_id]["user_followers"] > 100000:
        #     train_data.append(feature_dict[user_id])
        #     labels.append(label)
    result_list = [sublist + [item] for sublist, item in zip(train_data, labels)]
    print(len(train_data))
    print(len(result_list))
    train_data_f = open(train_file, "a+", encoding="utf-8")
    for line in result_list:
        train_data_f.write("%s\n" % json.dumps(line, ensure_ascii=False))
    train_data_f.close()
    feature_f = open(feature_file, "w+", encoding="utf-8")
    feature_f.write(json.dumps(feature_dict, ensure_ascii=False))
    feature_f.close()
    return feature_dict, train_data, labels
#
def model_train(train_data, labels):
    # dump model
    X, y = train_data, labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)
    print("test data")
    print(y_test)
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    filename = '../models/random_forest_model_v18.joblib'
    joblib.dump(rf_classifier, filename)
    feature_importances = rf_classifier.feature_importances_
    print("Feature Importances:")
    for i, importance in enumerate(feature_importances):
        print(f"Feature {i + 1}: {importance}")
    plt.bar(range(len(feature_importances)), feature_importances)
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Importance')
    plt.show()
    return rf_classifier
#
def rank(feature_dict, static_data, weights=[0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]):
    score_dict = {}
    filename = '../models/random_forest_model.joblib'
    rf_classifier = joblib.load(filename)
    for user_id in feature_dict:
        feature = feature_dict[user_id]
        # score = 0
        # for i in range(len(feature)):
        #     score += feature[i] * weights[i]
        score = 0
        if static_data[user_id]["user_followers"] > 100000:
            score = rf_classifier.predict_proba([feature])
            score = score[0][1]
            if score <= 0:
                score = 0.001

        score_dict.setdefault(user_id, score)
    sorted_dict = dict(sorted(score_dict.items(), key=lambda item: item[1], reverse=True))
    # print(sorted_dict)
    return sorted_dict


if __name__ == "__main__":
    static_profile = load_static_profile()
    dynamic_profile = load_dynamic_profile()
    influence_dict = calculate_persona_influence(static_profile, dynamic_profile)
    # print(influence_dict)
    frequency_dict = calculate_interaction_frequency(static_profile, dynamic_profile)
    # print(frequency_dict)
    broadcast_dict = calculate_information_broadcast(static_profile, dynamic_profile)
    # print(broadcast_dict)
    centrality_dict = calculate_social_network_centrality(static_profile, dynamic_profile)
    feature_dict, train_data, labels = merge_feature(static_profile, influence_dict, broadcast_dict, frequency_dict, centrality_dict)
    train_data, labels = [], []
    for line in open(train_file, "r", encoding="utf-8"):
        sample = json.loads(line.strip())
        train_data.append(sample[0:18])
        labels.append(sample[18])
    model = model_train(train_data, labels)
    sorted_dict = rank(feature_dict, static_profile)
    rank_idx = 1
    for user_id in sorted_dict:
        print("%s\t%s\t%s\t%s" % (rank_idx, user_id, static_profile[user_id]["user_name"], sorted_dict[user_id]))
        rank_idx += 1
