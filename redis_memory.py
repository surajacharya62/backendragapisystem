import redis
from langchain.vectorstores.redis import Redis


url = "redis-cli -u redis://default:KZMQMTMNlbdXMpcYOFb1qrggDb2Khxcs@redis-11998.crce178.ap-east-1-1.ec2.redns.redis-cloud.com:11998"
host = "redis-11998.crce178.ap-east-1-1.ec2.redns.redis-cloud.com"
password = "KZMQMTMNlbdXMpcYOFb1qrggDb2Khxcs"
port = 11998

def redis_connect():
    try:
        connect_redis = redis.Redis(host=host,password=password,port=port)
        print(connect_redis.ping())
    except Exception as e:
        print(e)



