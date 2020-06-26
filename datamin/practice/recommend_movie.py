import codecs
from math import sqrt

class recommendMovie():

    def __init__(self,data={},k=1,n=5,metric='pearson'):
        self.k = k
        self.n = n
        self.metric = metric
        self.names = []
        self.ratings = {}
        if self.metric == 'pearson':
            self.funcName = self.pearson
        else:
            self.funcName = self.cosDis
        if type(data).__name__ == 'dict':
            self.data = data

    def loadData(self,path):
        with codecs.open("../data/Movie_Ratings.csv", 'r', 'utf8') as f:
            for line in f:
                items = line.strip().split(',')
                if self.ratings == {}:
                    for item in items:
                        if item != '':
                            item = item.strip('"')
                            self.ratings[item] = {}
                            self.names.append(item)
                else:
                    movie = items[0].strip('"')
                    for i in range(1, len(items)):
                        if items[i] == '':
                            continue
                        # 评分
                        self.ratings[self.names[i - 1]][movie] = int(items[i])

    def pearson(self,r1,r2):
        """皮尔逊相关系数估计值
        r1 为推荐对象
        """
        sum_xy = 0
        sum_x = 0
        sum_y = 0
        sum_x2 = 0
        sum_y2 = 0
        n = 0
        for k in r1:
            if k in r2:
                x = r1[k]
                y = r2[k]
                n += 1
                sum_xy += x * y
                sum_x += x
                sum_y += y
                sum_x2 += pow(x, 2)
                sum_y2 += pow(y, 2)
        if n == 0:
            return 0
        deno = sqrt(sum_x2 - pow(sum_x, 2) / n) * sqrt(sum_y2 - pow(sum_y, 2) / n)
        if deno == 0:
            return 0
        else:
            return (sum_xy - sum_x * sum_y / n) / deno
        pass

    def cosDis(self,r1,r2):
        """余弦相似度，适用于稀疏数据，排除0的影响"""
        sum_xy = 0
        sum_x2 = 0
        sum_y2 = 0
        for k in r1:
            x = r1[k]
            y = r2[k]
            sum_xy = x*y
            sum_x2 = pow(x,2)
            sum_y2 = pow(y,2)
        deno = sqrt(sum_x2)*sqrt(sum_y2)
        if deno == 0:
            return 0
        return sum_xy/deno

    def nearest(self,username):
        distances = []
        for k in self.ratings:
            if username != k:
                d = self.funcName(self.ratings[username],self.ratings[k])
                distances.append((k,d))
        distances.sort(key=lambda t:t[1],reverse=True)
        return distances

    def recommend(self,username):
        #获取K个近邻
        nearest = self.nearest(username)[:self.k]
        #获取总的分数
        total = 0
        for item in nearest:
            total += item[1]
        userRatings = self.ratings[username]
        recommends = {}
        #计算推荐人没有看过的电影的推荐分数
        for item in nearest:
            #当前用户的权重
            weight = item[1] / total
            #当前用户评价过的电影
            otherRatings = self.ratings[item[0]]
            for o in otherRatings:
                if o not in userRatings:
                    #如果这个电影不在用户评价过的电影中
                    if o not in recommends:
                        recommends[o] = (otherRatings[o]*weight)
                    else:
                        recommends[o] = (recommends[o]+otherRatings[o]*weight)
        recommends = list(recommends.items())
        recommends.sort(key=lambda t:t[1],reverse=True)
        return recommends
    pass

if __name__ == '__main__':
    r = recommendMovie(k=3)
    r.loadData("../data/Movie_Ratings.csv")
    dis = r.recommend('Patrick C')
    print(dis)
    pass