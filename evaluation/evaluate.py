#coding=utf-8
from math import log2

product = "spark_thinking"
# product = "electric_toothbrush"
# product = "intelligent_floor_scrubber"
# product = "ruby_face_cream"
# product = "abc_reading"
# product = "supor_boosted_showerhead"

def precision_at_k(actual, predicted, k):
    """
    计算准确率（Precision@k）

    Parameters:
    - actual: 实际的项目列表
    - predicted: 预测的项目列表
    - k: Top k 推荐

    Returns:
    - Precision@k
    """
    actual_set = set(actual)
    predicted_set = set(predicted[:k])
    intersection = actual_set.intersection(predicted_set)
    return len(intersection) / k if k != 0 else 0

def recall_at_k(actual, predicted, k):
    """
    计算召回率（Recall@k）

    Parameters:
    - actual: 实际的项目列表
    - predicted: 预测的项目列表
    - k: Top k 推荐

    Returns:
    - Recall@k
    """
    actual_set = set(actual)
    predicted_set = set(predicted[:k])
    intersection = actual_set.intersection(predicted_set)
    return len(intersection) / len(actual_set) if len(actual_set) != 0 else 0

def ndcg_at_k(actual, predicted, k):
    """
    计算归一化折损累积（NDCG@k）

    Parameters:
    - actual: 实际的项目列表
    - predicted: 预测的项目列表
    - k: Top k 推荐

    Returns:
    - NDCG@k
    """
    dcg = sum(1 / (log2(i + 2)) if item in actual else 0 for i, item in enumerate(predicted[:k]))
    idcg = sum(1 / (log2(i + 2)) for i in range(min(k, len(actual))))
    return dcg / idcg if idcg != 0 else 0


def ranking_evaluation(actual_items, predicted_items, k):
    precision_k = precision_at_k(actual_items, predicted_items, k)
    recall_k = recall_at_k(actual_items, predicted_items, k)
    ndcg_k = ndcg_at_k(actual_items, predicted_items, k)
    return precision_k, recall_k, ndcg_k

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
        labels.append(domain)
        sizes.append(cnt)
        total_cnt += cnt
        idx += 1
    sizes = [int((i/total_cnt)*100) for i in sizes]
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


def eval_llm():
    # eval for main method
    top_k = 13
    sample_k = 20
    whole_users, selection_names, predicted_names = influencer_ranking(static_profile, dynamic_profile, "model", "prompt4behavior_stepbystep", top_k, sample_k)
    
    print("Main Method: ")
    overall_evaluation(gt_names, predicted_names)
    
    print("Pre-Selection: ")
    overall_evaluation(gt_names, selection_names[:10])

    whole_users, selection_names, predicted_names = influencer_ranking(static_profile, dynamic_profile, "model", "prompt4behavior_stepbystep", top_k, sample_k, "otherwise")
    
    print("wo Simulation: ")
    overall_evaluation(gt_names, predicted_names)

    # eval for wo pre-selection
    whole_users, selection_names, predicted_names = influencer_ranking(static_profile, dynamic_profile, "sample", "prompt4behavior_stepbystep", top_k, sample_k)
    print("wo Pre-selection: ")
    overall_evaluation(gt_names, predicted_names)

    # eval for wo profile
    whole_users, selection_names, predicted_names = influencer_ranking(static_profile, dynamic_profile, "model", "prompt4behavior_wo_profile", top_k, sample_k)
    
    print("wo Profile: ")
    overall_evaluation(gt_names, predicted_names)

    # # eval for wo cot
    whole_users, selection_names, predicted_names = influencer_ranking(static_profile, dynamic_profile, "model", "prompt4behavior_wo_cot", top_k, sample_k)
    
    print("wo CoT: ")
    overall_evaluation(gt_names, predicted_names)


if __name__ == "__main__":

    if sys.argv[1] == "llm":
        eval_llm()
    
    elif sys.argv[1] == "classic":
        print("Switch to the XFlow projects for the classic results: ")
        print("cd ../XFlow/examples/")
        print("python main.py")

    