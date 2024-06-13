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
    "cookie":"SUB=_2A25PG2G3DeRhGeFJ6FEQ8irFyz-IHXVsUdR_rDV8PUNbmtCOLXCnkW9NfCSfdkTX1FCNXr_bCtEAhtwJiPwdyHMa; ",
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

def write_to_redis(uid):
    """
    将列表页获取到的url列表 存入redis
    :param url_list:
    :return:
    """
    a = RedisClient().conn
    if a.scard("talk:weibo_bozhu_uid") > 100000:
        return
    r = a.sadd("talk:weibo_bozhu_uid",uid)
    if r == 0:
        print('%s in redis' % uid)
        # rid = a.spop("talk:weibo_bozhu_uid")
        # if not rid:
        #     print(rid)
        # print(rid)
        return
    a.sadd("talk:weibo_bozhu_uid_queue", uid)
    print('add %s success' % uid)


def init_redis(filename):
    for line in open(filename, 'r', encoding='utf-8'):
        uid = line.strip('\n').split('\t')[0]
        write_to_redis(uid)



if __name__ == '__main__':
    init_redis("dataset/spark_thinking_r0_uid.txt")
