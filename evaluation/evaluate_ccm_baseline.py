from evaluate import ranking_evaluation
import json

def overall_evaluation(gt_names, predicted_items):
    print("gt_names: %s" % gt_names)
    print("predicted_names: %s" % predicted_items)
    precision_1, recall_1, ndcg_1 = ranking_evaluation(gt_names, predicted_items, 1)
    precision_2, recall_2, ndcg_2 = ranking_evaluation(gt_names, predicted_items, 2)
    precision_5, recall_5, ndcg_5 = ranking_evaluation(gt_names, predicted_items, 5)
    precision_10, recall_10, ndcg_10 = ranking_evaluation(gt_names, predicted_items, 10)
    print("P@1=%.3f, R@1=%.3f, NDCG@1=%.3f" % (precision_1, recall_1, ndcg_1))
    print("P@2=%.3f, R@2=%.3f, NDCG@2=%.3f" % (precision_2, recall_2, ndcg_2))
    print("P@5=%.3f, R@5=%.3f, NDCG@5=%.3f" % (precision_5, recall_5, ndcg_5))
    print("P@10=%.3f, R@10=%.3f, NDCG@10=%.3f" % (precision_10, recall_10, ndcg_10))
    return

gt_ids = {
    "huohuasiwei": ["1748332981", "1674427277", "3213060995", "1929045072"],
    "brush": ["1738877650", "1960732503", "1363450462", "1669537002", "3309403941"],
    "cream": ["1776459797", "3993044286", "2360171883", "1832452643", "2833050332"],
    "yunjing": ["2292724833", "1642720480", "1735618597", "3340909732"],
    "abc": ["1689918212", "1468736221", "2626683933", "6690736938"],
    "alice": ["5426716682", "5716589670", "1806558670", "2503628005"]
}

product = "huohuasiwei"
# product = "abc"
# product = "alice"
# product = "brush"
# product = "cream"
# product= "yunjing"
# product = "abc"
# product = "alice"

relation_file = "./t100/%s_r5_relation.txt" % product
static_file = "./t100/%s_r5.static.graph" % product
dynamic_file = "./t100/%s_r5.dynamic.graph" % product
cim_file = "./ccm/%s.v0.ids" % product

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

# load cim ids
def load_cim_ids():
    global cim_file
    with open(cim_file, 'r', encoding='utf-8') as file:
        cim_data = json.load(file)
    return cim_data

static_data = load_static_profile()
dynamic_data = load_dynamic_profile()
cim_data = load_cim_ids()

for line in open("cim.txt", "r", encoding="utf-8"):
    line = line.strip()
    fields = line.split(" ")
    if fields[0] != product:
        continue
    algorithm = fields[1]
    r4maxinter = [static_data[cim_data[_id]]["user_name"] for _id in fields[2].split(",")]
    r4maxfollowers = [static_data[cim_data[_id]]["user_name"] for _id in fields[3].split(",")]
    gt_names = [static_data[_id]["user_name"] for _id in gt_ids[product]]
    print(algorithm)
    print("MAXInteraction")
    overall_evaluation(gt_names, r4maxinter)
    print("MAXFollowers")
    overall_evaluation(gt_names, r4maxfollowers)