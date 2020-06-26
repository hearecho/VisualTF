import codecs
from math import sqrt

users = {"Angelica": {"Blues Traveler": 3.5, "Broken Bells": 2.0, "Norah Jones": 4.5, "Phoenix": 5.0,
                      "Slightly Stoopid": 1.5, "The Strokes": 2.5, "Vampire Weekend": 2.0},
         "Bill": {"Blues Traveler": 2.0, "Broken Bells": 3.5, "Deadmau5": 4.0, "Phoenix": 2.0, "Slightly Stoopid": 3.5,
                  "Vampire Weekend": 3.0},
         "Chan": {"Blues Traveler": 5.0, "Broken Bells": 1.0, "Deadmau5": 1.0, "NorahJones": 3.0, "Phoenix": 5,
                  "Slightly Stoopid": 1.0},
         "Dan": {"Blues Traveler": 3.0, "Broken Bells": 4.0, "Deadmau5": 4.5, "Phoenix": 3.0, "Slightly Stoopid": 4.5,
                 "The Strokes": 4.0, "Vampire Weekend": 2.0},
         "Hailey": {"Broken Bells": 4.0, "Deadmau5": 1.0, "Norah Jones": 4.0, "The Strokes": 4.0,
                    "Vampire Weekend": 1.0},
         "Jordyn": {"Broken Bells": 4.5, "Deadmau5": 4.0, "Norah Jones": 5.0, "Phoenix": 5.0, "Slightly Stoopid": 4.5,
                    "The Strokes": 4.0, "Vampire Weekend": 4.0},
         "Sam": {"Blues Traveler": 5.0, "Broken Bells": 2.0, "Norah Jones": 3.0, "Phoenix": 5.0,
                 "Slightly Stoopid": 4.0, "The Strokes": 5.0},
         "Veronica": {"Blues Traveler": 3.0, "Norah Jones": 5.0, "Phoenix": 4.0, "Slightly Stoopid": 2.5,
                      "The Strokes": 3.0}
         }


class recommender:
    def __init__(self, data, k=1, metric='pearson', n=5):
        """
        初始化推荐模块
        :param data:    训练数据
        :param k:       K近邻算法中的K个邻居
        :param metric:  使用的计算相似度的方式
        :param n:       推荐结果的数量
        """
        self.k = k
        self.n = n
        self.username2id = {}
        self.userid2name = {}
        self.productid2name = {}
        self.metric = metric
        if self.metric == 'pearson':
            self.fn = self.pearson
        if type(data).__name__ == 'dict':
            self.data = data

    def convProductId2name(self, key):
        """
        通过产品ID获取名称
        :param key:  产品ID
        :return:
        """
        if id in self.productid2name:
            return self.productid2name[key]
        else:
            return key

    def userRatings(self, key, n):
        """返回该用户评分最高的n个产品"""
        print("Rating for " + self.userid2name[key])
        ratings = self.data[key]
        ratings = list(ratings.items())
        ratings = [(self.convProductId2name(k), v) for (k, v) in ratings]
        # 排序
        ratings.sort(key=lambda t: t[1], reverse=True)
        ratings = ratings[:n]
        for rating in ratings:
            print("{}\t{}".format(rating[0], rating[1]))
        return ratings

    def loadData(self, path=''):
        """加载数据"""
        self.data = {}
        i = 0
        f = codecs.open(path + "BX-Book-Ratings.csv", 'r', 'utf8')
        for line in f:
            i += 1
            fields = line.split(';')
            user = fields[0].strip('"')
            book = fields[1].strip('"')
            rating = int(fields[2].strip().strip('"'))
            if user in self.data:
                currentRatings = self.data[user]
            else:
                currentRatings = {}
            currentRatings[book] = rating
            self.data[user] = currentRatings
        f.close()
        # 存储书籍信息
        f = codecs.open(path + "BX-Books.csv", 'r', 'utf8')
        for line in f:
            i += 1
            fields = line.split(';')
            isbn = fields[0].strip('"')
            title = fields[1].strip('"')
            author = fields[2].strip().strip('"')
            title = title + ' by ' + author
            self.productid2name[isbn] = title
        f.close()
        # 存储用户信息
        f = codecs.open(path + "BX-Users.csv", 'r', 'utf8')
        for line in f:
            i += 1
            # print(line)
            # separate line into fields
            fields = line.split(';')
            userid = fields[0].strip('"')
            location = fields[1].strip('"')
            if len(fields) > 3:
                age = fields[2].strip().strip('"')
            else:
                age = 'NULL'
            if age != 'NULL':
                value = location + ' (age: ' + age + ')'
            else:
                value = location
            self.userid2name[userid] = value
            self.username2id[location] = userid
        f.close()

    def pearson(self, r1, r2):
        """皮尔逊相关系数"""
        sum_xy = 0
        sum_x = 0
        sum_y = 0
        sum_x2 = 0
        sum_y2 = 0
        n = 0
        for k in r1:
            if k in r2:
                n += 1
                x = r1[k]
                y = r2[k]
                sum_xy += x * y
                sum_x += x
                sum_y += y
                sum_x2 += pow(x, 2)
                sum_y2 += pow(y, 2)
        if n == 0:
            return 0
        deno = sqrt(sum_x2 - pow(sum_x, 2) / n) * sqrt(sum_y2 - pow(sum_y, 2) / n)
        if (deno == 0):
            return 0
        else:
            return (sum_xy - sum_x * sum_y / n) / deno
        pass

    def computNearestNeighbor(self, username):
        """获取邻近用户"""
        dis = []
        for instance in self.data:
            if instance != username:
                d = self.fn(self.data[username], self.data[instance])
                dis.append((instance, d))
        # 按照距离排序，相似度高的在前面
        dis.sort(key=lambda t: t[1], reverse=True)
        return dis

    def recommend(self, user):
        recommendList = {}
        nearest = self.computNearestNeighbor(user)
        userRatings = self.data[user]
        totalDis = 0.0
        for i in range(self.k):
            totalDis += nearest[i][1]
        # 计算每个邻近用户所占的百分比
        for i in range(self.k):
            weight = nearest[i][1] / totalDis
            # 获取用户名称
            name = nearest[i][0]
            # 获取用户评分
            neighborRatings = self.data[name]
            # 获得没有评价过的商品
            for artist in neighborRatings:
                if artist not in userRatings:
                    if artist not in recommendList:
                        recommendList[artist] = (neighborRatings[artist] * weight)
                    else:
                        recommendList[artist] = (recommendList[artist] + neighborRatings[artist] * weight)

        recommendList = list(recommendList.items())
        recommendList = [(self.convProductId2name(k), v)
                         for (k, v) in recommendList]
        # 排序并返回
        recommendList.sort(key=lambda artistTuple: artistTuple[1],
                           reverse=True)
        # 返回前n个结果
        return recommendList[:self.n]


if __name__ == '__main__':
    r = recommender(users)
    print(r.recommend('Jordyn'))
