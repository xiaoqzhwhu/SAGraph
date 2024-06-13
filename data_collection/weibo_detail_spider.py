# coding=utf-8

import datetime
import json
import re
import threading
import time
import urllib.parse
from multiprocessing import Process
import requests
from weibo_list_spider import base_path, get_response, RedisClient, loggings

proxy_pool = "127.0.0.1:2222"

def write_to_redis(uid):
    """
    将列表页获取到的url列表 存入redis
    :param url_list:
    :return:
    """
    a = RedisClient().conn
    if a.scard("talk:weibo_bozhu_uid") > 2000000:
        return
    r = a.sadd("talk:weibo_bozhu_uid",uid)
    if r == 0:
        return
    a.sadd("talk:weibo_bozhu_uid_queue", uid)


def write_json(data, round):
    json_data = json.dumps(data,ensure_ascii=False)
    time_str = datetime.datetime.now().strftime("%Y-%m-%d")
    with open(base_path + "weibodata_"+time_str+"_"+str(round)+".json","a+",encoding="utf-8") as f:
    # with open("1.json","a+",encoding="utf-8") as f:
        f.write(json_data)
        f.write("\n")


def get_weibo_comments_replies(user_id,comment_id):
    # https://weibo.com/ajax/statuses/buildComments?is_reload=1&id=4728019363301240&is_show_bulletin=2&is_mix=1&fetch_level=1&max_id=0&count=20&uid=2803301701
    base_url = "https://weibo.com/ajax/statuses/buildComments?"
    get_data = {
        "flow": "1",
        "is_reload": "1",
        "id": comment_id,
        "is_show_bulletin": "2",
        "is_mix": "1",
        "fetch_level": "1",
        "max_id": "0",
        "count": "100",
        "uid": user_id,
    }
    start = 0
    end_page = 1000
    max_id = ""
    comment_list = []
    while True:  # TODO 正常运行时放开
    # for i in range(5):  # TODO 正常运行时只是掉 打开上一个循环
        if start+1 > end_page:
            break
        this_get_data = get_data
        if start != 0:
            this_get_data["max_id"] = max_id
        loggings.info(f"采集微博 回应 user_id={user_id} comment_id={comment_id} list page 第 {start} 页")
        url = base_url + urllib.parse.urlencode(this_get_data)
        datas = {}
        for i in range(6):
            try:
                response = get_response(url)
                # with open("test1.txt","w",encoding="utf-8") as f:
                #     f.write(response.text)
                datas = response.json()
                if datas:
                    break
                else:
                    continue
            except Exception as e:
                loggings.error(f"第{i}次获取微博 回应 报错：{repr(e)}")
        if not datas:
            loggings.error(f"微博 回应 user_id={user_id} comment_id={comment_id} list page 第 {start} 页 失败！翻页结束！")
            break
        max_id = datas["max_id"]
        loggings.debug(f"获取微博 回应 数量 len(datas['data']) = {len(datas['data'])}，max_id={datas['max_id']}")
        if not datas["data"]:
            break
        for data in datas["data"]:
            # loggings.debug(f"data == {data}")
            # com_com_dict = {}
            # com_com_dict["user_name"] = data["user"]["name"]
            # com_com_dict["id"] = data["id"]
            # com_com_dict["text"] = data["text_raw"]
            # com_com_dict["ref_comment"] = {}
            # com_com_dict["ref_comment"]["user_name"] = data.get("reply_comment", {}).get("user", {}).get("name", "")
            # com_com_dict["ref_comment"]["id"] = data.get("reply_comment", {}).get("id", "")
            # com_com_dict["ref_comment"]["text"] = data.get("reply_comment", {}).get("text", "")
            # comment_list.append(com_com_dict)
            user_id = data["user"]["id"]
            write_to_redis(user_id)
        comment_list.extend(datas["data"])
        if datas["max_id"] == 0:break
        start += 1
    # loggings.info(f"微博评论 user_id={user_id} comment_id={comment_id} 评论页采集完毕, 采集到：{start}")
    return comment_list


def get_weibo_comments_url(user_id,article_id):
    # https://weibo.com/ajax/statuses/buildComments?is_reload=1&id=4728019052135281&is_show_bulletin=2&is_mix=0&count=10&uid=2803301701
    base_url = "https://weibo.com/ajax/statuses/buildComments?"
    get_data = {
        "is_reload": "1",
        "id": article_id,
        "is_show_bulletin": "2",
        "is_mix": "0",
        "count": "200",
        "uid": user_id,
    }
    start = 0
    end_page = 1000
    max_id = ""
    comment_list = []
    while True:  # TODO 正常运行时放开
    # for i in range(2):  # TODO 正常运行时只是掉 打开上一个循环
        if start+1 > end_page:
            break
        this_get_data = get_data
        if start != 0:
            this_get_data["flow"] = 0
            this_get_data["max_id"] = max_id
        loggings.info(f"微博 评论 user_id={user_id} article_id={article_id} list page 第 {start} 页")
        url = base_url + urllib.parse.urlencode(this_get_data)
        datas = {}
        for i in range(6):
            try:
                response = get_response(url)
                # with open("test1.txt","w",encoding="utf-8") as f:
                #     f.write(response.text)
                datas = response.json()
                if datas:
                    break
                else:
                    continue
            except Exception as e:
                loggings.error(f"第{i}次获取微博 评论 报错：{repr(e)}")
        if not datas:
            loggings.error(f"微博 评论 user_id={user_id} article_id={article_id} list page 第 {start} 页 失败！翻页结束！")
            break
        max_id = datas["max_id"]
        loggings.debug(f"获取微博 评论 数量 len(datas['data']) = {len(datas['data'])}，max_id={datas['max_id']}")
        if not datas["data"]:
            break
        for data in datas["data"]:
            # loggings.debug(f"data == {data}")
            # comment_data = {}
            # comment_data["user_name"] = data["user"]["name"]
            # comment_data["text"] = data["text_raw"]
            # comment_data["id"] = data["id"]
            user_id =  data["user"]["id"]
            write_to_redis(user_id)
            total_number = data["total_number"]
            if total_number>0:
                # comment_data["replies"] = []
                yu_commet_num = len(data["comments"])
                if 0 < yu_commet_num and total_number <= yu_commet_num:
                    for com_com in data["comments"]:
                        # com_com_dict = {}
                        # com_com_dict["user_name"] = com_com["user"]["name"]
                        # com_com_dict["id"] = com_com["id"]
                        # com_com_dict["text"] = com_com["text_raw"]
                        # com_com_dict["ref_comment"] = {}
                        # com_com_dict["ref_comment"]["user_name"] = com_com.get("reply_comment",{}).get("user",{}).get("name","")
                        # com_com_dict["ref_comment"]["id"] = com_com.get("reply_comment",{}).get("id","")
                        # com_com_dict["ref_comment"]["text"] = com_com.get("reply_comment",{}).get("text","")
                        # comment_data["replies"].append(com_com_dict)
                        user_id = com_com["user"]["id"]
                        write_to_redis(user_id)
                elif total_number > yu_commet_num:
                    comment_id = data["id"]
                    replies_list = get_weibo_comments_replies(user_id,comment_id)
                    data["comments"].extend(replies_list)
            comment_list.append(data)
        if datas["max_id"] == 0: break
        start += 1
    loggings.info(f"微博 评论 user_id={user_id} article_id={article_id} 评论页采集完毕, 采集到：{start}")
    return comment_list


def get_weibo_reposts(comment_id):
    # https://weibo.com/ajax/statuses/repostTimeline?id=4728010400596670&page=1&moduleID=feed&count=10
    base_url = "https://weibo.com/ajax/statuses/repostTimeline?"
    get_data = {
        "id": comment_id,
        "page": "1",
        "moduleID": "feed",
        "count": "200",
    }
    start = 1
    end_page = 1000
    max_id = ""
    comment_list = []
    while True:  # TODO 正常运行时放开
    # for i in range(5):  # TODO 正常运行时只是掉 打开上一个循环
        if start+1 > end_page:
            break
        this_get_data = get_data
        if start != 0:
            # this_get_data["count"] = 20
            this_get_data["page"] = start
        loggings.info(f"微博 转发 获取 comment_id={comment_id} list page 第 {start} 页")
        url = base_url + urllib.parse.urlencode(this_get_data)
        datas = {}
        for i in range(6):
            try:
                response = get_response(url)
                # with open("test1.txt","w",encoding="utf-8") as f:
                #     f.write(response.text)
                datas = response.json()
                if datas:
                    break
                else:
                    continue
            except Exception as e:
                loggings.error(f"第{i}次获取微博 转发 列表页报错：{repr(e)}")
        if not datas:
            start += 1
            continue
        # max_id = datas["max_id"]
        loggings.debug(f"获取微博 转发 数量 len(datas['data']) = {len(datas['data'])}")
        if not datas["data"]:
            break
        for data in datas["data"]:
            # loggings.debug(f"data == {data}")
            # com_com_dict = {}
            # com_com_dict["user_name"] = data["user"]["screen_name"]
            # com_com_dict["text"] = data["text_raw"]
            # comment_list.append(com_com_dict)
            user_id = data["user"]["id"]
            write_to_redis(user_id)
        comment_list.extend(datas["data"])
        # if datas["max_id"] == 0:break
        start += 1
    loggings.info(f"微博 转发 comment_id={comment_id} 评论页采集完毕, 采集到：{start}")
    return comment_list


def get_longtext(mblogid):
    global proxy_pool
    for i in range(10):
        try:
            proxy_str = requests.get("http://%s/vpsgn_get_proxy"%proxy_pool,timeout=10).text
            proxy = {"http": proxy_str, "https": proxy_str}
            visitor_url = "https://passport.weibo.com/visitor/genvisitor2"
            post_data = {
                "cb": "visitor_gray_callback",
                "tid": "",
                "from": "weibo",
            }
            headers = {
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            }
            visitor_response = requests.post(visitor_url, data=post_data, headers=headers, proxies=proxy)
            subs = re.findall(r'"sub":"(.*?)","', visitor_response.text)
            # usbps = re.findall(r'","subp":"(.*?)","', visitor_response.text)
            cookies = {
                # 'SUB': '_2AkMSOOx6f8NxqwFRmfoUyGnraIR-zAnEieKkZB2hJRMxHRl-yT9vqkoetRB6ObjClb6Eju3ciY1K94-4jjawrGAst-JD',
                'SUB': subs[0] if subs else "",
                # 'SUB': "_2AkMSOOX_f8NxqwFRmfoUyGnraIR-zAnEieKkZBQkJRMxHRl-yT9vqk1etRB6ObjLEMTbzrcwc2VszK-C5fiWEuFiF1rb",
                # 'SUBP': usbps[0] if usbps else "",
            }
            response = requests.get(f"https://weibo.com/ajax/statuses/longtext?id={mblogid}", cookies=cookies,
                                    headers=headers, proxies=proxy)
            return response.json()
        except Exception as e:
            loggings.error(f"获取longtext报错：{repr(e)}")


def controller_start():
    while True:
    # for j in range(1): # TODO 正常运行需要注释掉
        round = 5
        redis_cnn = RedisClient().conn
        rid = redis_cnn.spop("talk:weibo_article_mblogid_queue")
        if not rid:
            loggings.info(f"微博评论采集结束，队列中rid已经消耗完了！休息15分钟！！")
            time.sleep(900)
            continue
        mblogid = rid.decode()
        # mblogid = "Nu6fw1rmm"
        show_url = "https://weibo.com/ajax/statuses/show?id=" + str(mblogid)
        all_comment_data={}
        for i in range(10):
            try:
                response = get_response(show_url)
                all_comment_data = response.json()
                if all_comment_data:
                    break
                else:
                    continue
            except Exception as e:
                loggings.error(f"第{i}次获微博详情页报错：{repr(e)}")
        if not all_comment_data:
            loggings.error(f"获取微博详情页失败！放弃！")
            continue
        all_comment_data["longtext_data"] = get_longtext(mblogid)
        # loggings.debug(f"all_comment_data = {all_comment_data}")

        # all_comment_data = {}
        # all_comment_data["user_id"] = datas.get("user",{}).get("id",'')  # 2803301701
        # all_comment_data["user_name"] = datas.get("user",{}).get("screen_name",'')
        # all_comment_data["article_id"] = datas.get("id","")  # 4728019052135281
        # all_comment_data["mblogid"] = datas.get("mblogid","")
        # all_comment_data["text_raw"] = datas.get("text_raw","")
        # all_comment_data["comments_count"] = datas.get("comments_count","")
        # all_comment_data["reposts_count"] = datas.get("reposts_count","")

        if all_comment_data.get("user",{}).get("id",'') and all_comment_data.get("id",""):
            # 获取品论信息
            if all_comment_data["comments_count"] > 0:
                try:
                    comment_data = get_weibo_comments_url(user_id=all_comment_data.get("user",{}).get("id",''),article_id=all_comment_data.get("id",""))
                except Exception as e:
                    loggings.error(f"微博 评论 采集最大限度报错 userid={all_comment_data['user_id']} article_id={all_comment_data['article_id']} 报错信息：{repr(e)}")
                    continue
                all_comment_data["comments"] = comment_data
            # 获取转发数据
            if all_comment_data["reposts_count"] > 0:
                try:
                    reposts_data = get_weibo_reposts(all_comment_data["id"])
                except Exception as e:
                    loggings.error(f"微博 转发 采集最大限度报错 userid={all_comment_data['user']['id']} article_id={all_comment_data['id']} 报错信息：{repr(e)}")
                    continue
                all_comment_data["reposts"] = reposts_data

            # if all_comment_data["reposts_count"] >0 or all_comment_data["comments_count"] > 0:
            write_json(all_comment_data, round)
            # loggings.debug(f"all_comment_data = {all_comment_data}")

        else:
            loggings.error(f"获微博详情页报错，没有获取到user_id={all_comment_data['user_id']} 或者 article_id={all_comment_data['article_id']} 放弃！")



def get_detail_url():
    """
    获取详情页url
    :return:
    """
    loggings.info("线程开始")
    t_list = []
    for i in range(10):
        t1 = threading.Thread(target=controller_start)
        t1.setDaemon(True)
        t1.start()
        t_list.append(t1)

    for i in t_list:
        i.join()
    loggings.info("++++线程结束")

def proc():
    loggings.info("------start")
    process_list = []
    for i in range(5):  # 开启5个子进程执行fun1函数
        # p = Process(target=m, args=('Python',))  # 实例化进程对象
        p = Process(target=get_detail_url)  # 实例化进程对象
        p.start()
        process_list.append(p)
    for i in process_list:
        i.join()
    loggings.info("--------end")


if __name__ == '__main__':
    # controller_start()
    # get_longtext("NsFxnmkaA")
    # get_longtext("Nu6fw1rmm")
    # get_longtext("NuuTxoOtg")
    # get_weibo_comments_url(1674427277,4972644575222553)
    proc()
