# coding=utf-8

import re
import threading
import time
import urllib.parse
from multiprocessing import Process
from pathlib import Path
import redis
import requests
from loguru import logger
# from Redis_DB import RedisClient, loggings


REDIS_HOST = "127.0.0.1"
REDIS_PORT = 6379
#base_path = "/mnt/datadisk1/house_video/takl_data/weibo/"
base_path = "./"


HDADERS = {
    # "accept":'*/*',
    # "accept-encoding":'gzip, deflate, br',
    # "accept-language":'en-US,en;q=0.9',
    # "content-type":'application/x-www-form-urlencoded',
    #"cookie":"SUB=_2A25PG2G3DeRhGeFJ6FEQ8irFyz-IHXVsUdR_rDV8PUNbmtCOLXCnkW9NfCSfdkTX1FCNXr_bCtEAhtwJiPwdyHMa; ",
    #"cookie":"SUB=_2A25IcVCfDeRhGeNJ71oX8CvLwj-IHXVrD-xXrDV8PUNbmtANLW7mkW9NS9Kvd1vIh4OIh2SbtekuEksmMuBj6J5t; ",
    "cookie":"SUB=_2A25In-N8DeRhGeNJ71oX8CvLwj-IHXVr1Xq0rDV_PUNbm9ANLW7ukW9NS9Kvd2nRjI7i3Q9OAIAT3tS0UNjJhUjw; ",
    # "cookie":"SINAGLOBAL=6663743473455.534.1642649104065; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9WFOsKrUFyD4nTqfSo5NfG8i5JpX5KMhUgL.FoMNe0epeoB4ehe2dJLoIpMLxK-LBKBLBKMLxK.LBKeL12HkSh.c1Btt; ULV=1642731841944:2:2:2:9455606836418.162.1642731841941:1642649104071; ALF=1674532151; SSOLoginState=1642996151; SCF=ApdufH1eSffSWc2UvJ50Z8c7qC4wX4PUvF7VDnPgl4dC2z-E3NEQ4os_QfOaQMhCyVJ6wefX76J7JRQlLYlDbL4.; SUB=_2A25M6lHnDeRhGeFJ6FEQ8irFyz-IHXVvnsQvrDV8PUNbmtAKLRL-kW9NfCSfdoTPM6MAwuKDPtkxfYotu93Nj2Jd; XSRF-TOKEN=VeR2e4I3KdL8jY9eFtQyLRgy; WBPSESS=Pv6yU4DwlA2iWG5i2Eu5hKD-nrIN_KXJqocfvWGunudBVp50OZm5_jUxThFFwMKLqWS_0G-sBRorQza-EyGESlLXkS03EzAJcH4dthYFAzOGuNkDg5G88XE3BgDi3sNVE9D9Y7Wdkfk4LvYZkKzgdQ==",
    # "referer":'https://d.weibo.com/102803_ctg1_4288_-_ctg1_4288?from=faxian_hot&mod=fenlei',
    # "sec-ch-ua":'" Not A;Brand";v="99", "Chromium";v="96", "Google Chrome";v="96"',
    # "sec-ch-ua-mobile":'?0',
    # "sec-ch-ua-platform":'"Windows"',
    # "sec-fetch-dest":'empty',
    # "sec-fetch-mode":'cors',
    # "sec-fetch-site":'same-origin',
    "user-agent":'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36',
    # "x-requested-with":'XMLHttpRequest',
}

class Singleton(type):
    """
    单例模式 - 元类. 单例类可以通过参数 'metaclass=Singleton' 继承此元类.
    传递参数不同，则生成不同的单例
    """
    _instance_lock = threading.Lock()
    _instances = {}
    def __call__(cls, *args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        if cls not in cls._instances:
            cls._instances[cls] = {}
        if key not in cls._instances[cls]:
            with Singleton._instance_lock:
                if key not in cls._instances[cls]:
                    cls._instances[cls][key] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls][key]

class Loggings(metaclass=Singleton):

    def __init__(self):
        # project_path = Path.cwd().parent
        project_path = Path.cwd()
        log_path = Path(project_path, "log")
        t = time.strftime("%Y_%m_%d")
        logger.add(f"{log_path}/weibo_log_{t}.log", rotation="500MB", encoding="utf-8", enqueue=True,
               retention="10 days")

Loggings()
loggings = logger


class RedisClient(metaclass=Singleton):

    # def __init__(self,host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD,db=REDIS_DB):
    #     self.pool = redis.ConnectionPool(host=host, port=port, password=password,db=db)

    def __init__(self,host=REDIS_HOST, port=REDIS_PORT):
        self.pool = redis.ConnectionPool(host=host, port=port,db=0)

    @property
    def conn(self):
        if not hasattr(self, '_conn'):
            self.getConnection()
        return self._conn

    def getConnection(self):
        self._conn = redis.Redis(connection_pool = self.pool)


def write_to_redis_article(url_list):
    """
    将列表页获取到的url列表 存入redis
    :param url_list:
    :return:
    """
    result = 0
    a = RedisClient().conn
    for url in url_list:
        print('add weibo_article_mblogid %s' % url)
        r = a.sadd("talk:weibo_article_mblogid",url)
        if r == 0:
            continue
        a.sadd("talk:weibo_article_mblogid_queue", url)
        result += 1
    return result

def get_response(url:str):
    """
    获取请求
    :param url:
    :return:
    """
    spider_error = ''
    for i in range(6):
        try:
            ip_str = requests.get("http://172.24.99.255:9528/vpsgn_get_proxy",timeout=10).text
            if ip_str:
                ip_str = ip_str.strip()
                pattern = re.compile(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d+$')
                if not pattern.match(ip_str):
                    loggings.error(f"get proxy error ip_str:{ip_str}")
                    time.sleep(10)
                    continue
            proxy = {"http": ip_str, "https": ip_str}
            print('proxy')
            print(proxy)
            response = requests.get(url=url,headers=HDADERS,proxies=proxy,timeout=10)
            print('response')
            print(response)
            # response = requests.get(url=url,headers=headers,timeout=10)
            if response and response.status_code != 200: raise Exception
            return response
        except Exception as e:
            spider_error = e
            # loggings.error(f"第{i}次下载报错：url = {url}, {repr(e)}")
        # time.sleep(2)
    loggings.error(f"下载报错：url = {url}, {repr(spider_error)}")
    return None

def get_auth_list(group_type):
    # https://weibo.com/ajax/feed/hottimeline?since_id=0&refresh=1&group_id=1028032222&containerid=102803_2222&extparam=discover%7Cnew_feed&max_id=0&count=10
    base_url = "https://weibo.com/ajax/statuses/mymblog?"
    get_data = {
        "uid":group_type,
        "page":"1",
        "feature":"0",
    }
    start = 0
    end_page = 1
    since_id = ""
    while True:  # TODO 正常运行时放开
    # for i in range(5):
        if start+1 > end_page:# for i in range(2): # TODO 正常运行时只是掉 打开上一个循环
            break
        this_get_data = get_data
        if start != 0:
            this_get_data["since_id"] = since_id
        loggings.info(f"get 博主 uid={group_type} list page 第 {start} 页")
        url = base_url + urllib.parse.urlencode(this_get_data)
        loggings.debug(f"url == {url}")
        datas = {}
        for i in range(6):
            try:
                response = get_response(url)
                if "-->帐号登录" in response.text:
                    # loggings.error(f"博主列表页采集cookie过期。请使用新的cookie！！！！！！！！！！！！")
                    return True
                with open("test.txt","w",encoding="utf-8") as f:
                    f.write(response.text)
                datas = response.json()
                if datas:
                    break
                else:
                    continue
            except Exception as e:
                loggings.error(f"第{i}次获取博主列表页报错：{repr(e)}")
        if not datas:
            loggings.error(f"get 博主 uid={group_type} list page 第 {start} 页 失败！翻页结束！")
            break
        loggings.debug(f"datas = {datas}")
        since_id = datas["data"]["since_id"]
        mblogid_list = []
        if not datas["data"]["list"]:
            break
        loggings.debug(f'获取到数据：{len(datas["data"]["list"])}')
        for data in datas["data"]["list"]:
            # loggings.debug(f"data == {data}")
            mblogid_list.append(data["mblogid"])
        resulr = write_to_redis_article(mblogid_list)
        if resulr <= 5:
            break
        start += 1
    loggings.info(f"get 博主 uid={group_type} 表页采集完毕, 采集到：{start}")


def controller_start():
    while True:
    # for j in range(1): # TODO 正常运行需要注释掉
        redis_cnn = RedisClient().conn
        rid = redis_cnn.spop("talk:weibo_bozhu_uid_queue")
        if not rid:
            loggings.info(f"博主微博采集结束，队列中rid已经消耗完了！休息15分钟！！")
            time.sleep(900)
            continue
        rid = rid.decode()
        # rid = "47445653"
        # loggings.info(f"spider rid = {rid}, page=0")
        try:
            result = get_auth_list(group_type=rid)
            if result:
                loggings.error(f"博主列表页采集cookie过期。请使用新的cookie！！！！！！！！！！！！")
                break
        except Exception as e:
            loggings.error(f"博主微博采集最大限度报错 userid == {rid}：{repr(e)}")
            continue


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
    # test_group = "1674427277"
    # get_move_detail_url(test_group)
    proc()
