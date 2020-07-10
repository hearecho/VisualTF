"""
基于物品的协同过滤，使用修正后的余弦相似度.
"""
from math import sqrt
users = {"David": {"Imagine Dragons": 3, "Daft Punk": 5,
                   "Lorde": 4, "Fall Out Boy": 1},
         "Matt": {"Imagine Dragons": 3, "Daft Punk": 4,
                  "Lorde": 4, "Fall Out Boy": 1},
         "Ben": {"Kacey Musgraves": 4, "Imagine Dragons": 3,
                 "Lorde": 3, "Fall Out Boy": 1},
         "Chris": {"Kacey Musgraves": 4, "Imagine Dragons": 4,
                   "Daft Punk": 4, "Lorde": 3, "Fall Out Boy": 1},
         "Tori": {"Kacey Musgraves": 5, "Imagine Dragons": 4,
                  "Daft Punk": 5, "Fall Out Boy": 3}}


def cos_update(th1, th2, ratings):
    """
    修正后的余弦相似度
    1. 计算用户评价的平均分
    2. 计算分子，两个物品均评价过的用户的修正过的评分汇总
    :param th1: 物品1
    :param th2: 物品2
    :param ratings: 用户的评分
    :return:
    """
    average = {}
    for k, rating in ratings.items():
        average[k] = (float(sum(rating.values()))) / len(rating.values())
    num = 0
    dem1 = 0
    dem2 = 0
    for user, rating in ratings.items():
        if th1 in rating and th2 in rating:
            avg = average[user]
            num += (rating[th1] - avg) * (rating[th2] - avg)
            dem1 += (rating[th1] - avg) ** 2
            dem2 += (rating[th2] - avg) ** 2
    if dem1 ==0 or dem2 == 0:
        return 0
    return num/(sqrt(dem1)*sqrt(dem2))


def RToNR(min,max,rate):
    return (2*(rate-min)-(max-min))/(max-min)
def NRToR(min,max,rate):
    return (rate+1)*(max-min)/2 +min

def pred(ratings,similar):
    num = 0
    dem = 0
    for (k,v) in similar:
        num += ratings[k]*v
        dem += abs(v)
    return NRToR(1,5,num/dem)
    pass
def recommend(user,thing,users):
    """
    通过上述计算相似度算法，预测对于某个物品的评分
    1. 规范化评分
    2. 计算该物品和用户评价中物品计算相似度
    3. 计算规范化预测评分
    :param user: 用户
    :param thing: 预测评价对象
    :return:
    """
    ratings = {}
    similar = []
    for k,v in users[user].items():
        ratings[k] = RToNR(1,5,v)
        if k != thing:
            rate = cos_update(k,thing,users)
            if rate != 0:
                similar.append((k,rate))
    return pred(ratings,similar)


print(recommend("David","Kacey Musgraves",users))