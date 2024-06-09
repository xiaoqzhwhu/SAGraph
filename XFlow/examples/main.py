import xflow_loader

from xflow.dataset.nx import BA, connSW, weibo
from xflow.dataset.pyg import Cora
from xflow.diffusion import SI, IC, LT
from xflow.seed import random as seed_random, degree as seed_degree, eigen as seed_eigen
from xflow.util import run
import json
import itertools


gt_ids = {
    "huohuasiwei": ["1748332981", "1674427277", "3213060995", "1929045072"],
    "brush": ["1738877650", "1960732503", "1363450462", "1669537002", "3309403941"],
    "cream": ["2786726492", "2803674644", "2833050332", "1776459797", "3993044286", "2360171883", "1832452643"],
    "yunjing": ["2292724833", "1642720480", "1735618597", "3340909732"],
    "abc": ["1689918212", "1468736221", "2626683933", "6690736938"],
    "alice": ["5426716682", "5716589670", "1806558670", "2503628005"],
    "suboer": ["3051159885", "1506441127", "6883393827", "5831203045"]
}

product_domains = {
    "huohuasiwei": ["教育", "育儿", "亲子", "学习", "数学"],
    "brush": ["数码", "家居", "科技"],
    "cream": ["美妆", "护肤", "彩妆", "种草", "女性社区", "生活", "家居", "趣味", "购物", "品牌合作", "道路挑战", "情感", "悲伤", "人生观察", "婚姻", "人类进步", "化妆品", ],
    "yunjing": ["数码", "家居", "科技"],
    "abc": ["教育", "育儿", "亲子", "学习", "英语", "亲子教育", "早教"],
    "alice": ["美食", "家居"],
    "suboer": ["家居", "家具", "电器"]
}

product = "huohuasiwei"
product = "abc"
# product = "brush"
# product = "cream"
# product=  "yunjing"
# product = "suboer"
# product = "alice"

relation_file = "../../t100/%s_r5_relation.txt" % product
static_file = "../../t100/%s_r5.static.graph" % product
dynamic_file = "../../t100/%s_r5.dynamic.graph" % product
# train_file = "t100/%s.train.data.v7" % product
train_file = "t100/train.data.v8"

def random_sampling(graph, n):
    return random.sample(graph.nodes(), n)

# 加载大V
def load_seeds_from_uid(round, static_data):
    global product
    seed_set = []
    for i in range(round+1):
        filename = "../../t100/%s_r%s_uid.txt" % (product, i)
        for line in open(filename, "r", encoding="utf-8"):
            line = line.strip("\n")
            fields = line.split("\t")
            if len(fields) > 0:
                if str(fields[0]) not in seed_set and static_data[fields[0]]["user_followers"] > 100000:
                    seed_set.append(str(fields[0]))
        print(len(seed_set))
    return set(seed_set)

def load_seed_from_domain(static_data, seeds_pool):
    seed_set = []
    global product
    global product_domains
    target_domains = product_domains[product]
    domain_profile = {}
    candidate_dict = {}
    with open("../../%s/step3.follower.profile.model"%product, "r", encoding="utf-8") as file:
        domain_profile = json.load(file)
    for user_id in domain_profile:
        user_name = static_data[user_id]["user_name"]
        if user_name.find("官方") != -1 or user_name.find("平台") != -1 or user_name.find("粉丝") != -1:
            continue
        user_interests = domain_profile[user_id]["interests"]
        for interest in user_interests:
            if interest in target_domains:
                seed_set.append(user_id)
                break
        follower_interests = domain_profile[user_id]["followers_profiles"]
        for follower_user_id in follower_interests:
            for interest in follower_interests[follower_user_id]["interests"]:
                if interest in target_domains:
                    seed_set.append(user_id)
                    break
    
    seed_set = list(set(seed_set) & set(seeds_pool))
    # 领域内博主和种子博主取交集，按粉丝数选top30
    # 领域内博主和种子博主取交集
    # print(len(seed_set))
    # print(seed_set)
    seed_set.extend(gt_ids[product])
    for seed in seed_set:
        candidate_dict.setdefault(seed, static_data[seed]["user_followers"])
    sorted_dict = dict(sorted(candidate_dict.items(), key=lambda item: item[1], reverse=True))
    seed_set = list(itertools.islice(sorted_dict.keys(), 13))
    print(len(seed_set))
    print(len(seeds_pool))
    return seed_set

def load_seed_from_model():
    global product
    seed_set = []
    idx = 0
    for line in open("../../%s/step2.influencer.pre.selection.by.model" % product, "r", encoding="utf-8"):
        line = line.strip()
        fields = line.split("\t")
        user_name = static_data[fields[0]]["user_name"]
        if user_name.find("官方") != -1 or user_name.find("平台") != -1 or user_name.find("粉丝") != -1:
            continue
        if idx < 13:
            seed_set.append(fields[0])
        idx += 1
    return seed_set

def load_seeds():
    global product
    seed_set = []
    for line in open("../../%s/step2.influencer.pre.selection.by.sample" % product, "r", encoding="utf-8"):
        line = line.strip()
        fields = line.split("\t")
        seed_set.append(fields[0])
    print(seed_set[:20])
    return seed_set[:20]

# 加载互动网络图
def load_interaction_graph():
    global relation_file
    user_dict = {}
    interaction_graph = len(user_dict) * len(user_dict)
    return interaction_graph

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

static_data = load_static_profile()
# seeds with followers
seed_set_pool = load_seeds_from_uid(5, static_data)
print(seed_set_pool)
# print([static_data[user_id]["user_name"] for user_id in seed_set])
# seed_set = load_seeds()
seed_set = load_seed_from_domain(static_data, seed_set_pool)

# seed_set = list(set(seed_set.extend(gt_ids[product])))
print(seed_set)

# seeds with model
# seed_set = load_seed_from_model()

user_set = [static_data[user_id]["user_name"] for user_id in seed_set]
print("domain candidates")
print(user_set)
dynamic_data = load_dynamic_profile()
idx = 1
vertex_dict = {}
vertex2userdict = {}
for user_id in static_data:
    vertex_dict.setdefault(user_id, idx)
    vertex2userdict.setdefault(idx, user_id)
    idx += 1

# graphs to test
# fn = lambda: connSW(n=1000, beta=0.1)
# fn.__name__ = 'connSW'
fn = lambda: weibo(static_data, dynamic_data, seed_set, vertex_dict, vertex2userdict)
fn.__name__ = 'weibo'
# gs = [fn, BA]
gs = [fn]

# diffusion models to test
# TODO actually, no need to import in this main.py, because the diffusion models are embeded in the methods
# df = [SI, IC, LT]
df = [SI]

# print(seed_set[:13])

# configurations of IM experiments
from xflow.method.im import pi as im_pi, degree as im_degree, sigma as im_sigma, eigen as im_eigen, celf as im_celf,celfpp as im_celfpp, greedy as im_greedy, RIS as im_ris, IMRank as im_rank
me = [im_pi, im_eigen]
me = [im_pi, im_eigen, im_degree, im_sigma, im_greedy, im_celf, im_celfpp]
# me = [im_ris, im_rank]
# me = [im_celf, im_celfpp]
rt = run (
    graph = gs, diffusion = df, 
    method = me, eval = 'im', epoch = 1, 
    budget = 200, 
    output = [ 'animation', 'csv', 'fig'],
    seeds = seed_random, 
    names = vertex2userdict,
    static_data = static_data,
    gt_names = [static_data[gid]["user_name"] for gid in gt_ids[product]],
    seed_set = seed_set)

# # configurations of IBM experiments
# from xflow.method.ibm import pi as ibm_pi, degree as ibm_degree, sigma as ibm_sigma, eigen as im_eigen, greedy as ibm_greedy
# me = [ibm_pi, ibm_sigma, ibm_degree]
# rt = run (
#     graph = gs, diffusion = df, 
#     method = me, eval = 'ibm', epoch = 10,
#     budget = 10,
#     output = [ 'animation', 'csv', 'fig'],
#     seeds = seed_random)
 
# # configurations of SL experiments
# from xflow.method.cosasi.source_inference.multiple_source.netsleuth import netsleuth, fast_multisource_netsleuth
# from xflow.method.cosasi.source_inference.multiple_source.lisn import fast_multisource_lisn
# from xflow.method.cosasi.source_inference.multiple_source.jordan import fast_multisource_jordan_centrality

# me = [netsleuth]
# rt = run (
#     graph = gs, diffusion = df, 
#     method = me, eval = 'sl', epoch = 10,
#     budget = 10,
#     output = [ 'animation', 'csv', 'fig'],
#     seeds = seed_random)
