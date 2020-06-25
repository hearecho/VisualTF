from math import sqrt
'''
曼哈顿距离
'''
def Manhattan(x1,y1,x2,y2):
    return abs(x1-x2)+abs(y1-y2)
'''
欧几里得距离
'''
def Euclid(x1,y1,x2,y2):
    return pow(pow(x1-x2,2)+pow(y1-y2,2),2)



users = {"Angelica": {"Blues Traveler": 3.5, "Broken Bells": 2.0, "Norah Jones": 4.5,"Phoenix": 5.0, "Slightly Stoopid": 1.5, "The Strokes": 2.5, "Vampire Weekend": 2.0},
         "Bill":{"Blues Traveler": 2.0, "Broken Bells": 3.5, "Deadmau5": 4.0, "Phoenix": 2.0, "Slightly Stoopid": 3.5, "Vampire Weekend": 3.0},
         "Chan": {"Blues Traveler": 5.0, "Broken Bells": 1.0, "Deadmau5": 1.0, "NorahJones": 3.0, "Phoenix": 5, "Slightly Stoopid": 1.0},
         "Dan": {"Blues Traveler": 3.0, "Broken Bells": 4.0, "Deadmau5": 4.5, "Phoenix": 3.0, "Slightly Stoopid": 4.5, "The Strokes": 4.0, "Vampire Weekend": 2.0},
         "Hailey": {"Broken Bells": 4.0, "Deadmau5": 1.0, "Norah Jones": 4.0, "The Strokes": 4.0, "Vampire Weekend": 1.0},
         "Jordyn": {"Broken Bells": 4.5, "Deadmau5": 4.0, "Norah Jones": 5.0, "Phoenix": 5.0, "Slightly Stoopid": 4.5, "The Strokes": 4.0, "Vampire Weekend": 4.0},
         "Sam": {"Blues Traveler": 5.0, "Broken Bells": 2.0, "Norah Jones": 3.0, "Phoenix": 5.0, "Slightly Stoopid": 4.0, "The Strokes": 5.0},
         "Veronica": {"Blues Traveler": 3.0, "Norah Jones": 5.0, "Phoenix": 4.0, "Slightly Stoopid": 2.5, "The Strokes": 3.0}
}
'''
皮尔森相关系数的近似值计算方式
'''
def pearson(r1,r2):
    sum_xy = 0
    sum_x =0
    sum_y = 0
    sum_x2 = 0
    sum_y2 = 0
    n = 0
    for k in r1:
        if k in r2:
            n += 1
            x = r1[k]
            y = r2[k]
            sum_xy += x*y
            sum_x += x
            sum_y += y
            sum_x2 += pow(x,2)
            sum_y2 += pow(y,2)
    deno = sqrt(sum_x2-pow(sum_x,2)/n)*sqrt(sum_y2-pow(sum_y,2)/n)
    if(deno == 0):
        return 0
    else:
        return (sum_xy-sum_x*sum_y/n)/deno


'''
皮尔森相关系数的准确计算方法
'''
def corrPearson(r1,r2):
    sum_x = 0
    sum_y = 0
    aver_x = 0
    aver_y = 0
    sum_aver_xy = 0
    sum_aver_x2 = 0
    sum_aver_y2 = 0
    n = 0
    for k in r1:
        if k in r2:
            n += 1
            x = r1[k]
            y = r2[k]
            sum_x += x
            sum_y += y
    aver_x = sum_x / n
    aver_y = sum_y / n
    for k in r1:
        if k in r2:
            x = r1[k]
            y = r2[k]
            sum_aver_xy += (x-aver_x)*(y-aver_y)
            sum_aver_x2 += pow(x-aver_x,2)
            sum_aver_y2 += pow(y-aver_y,2)
    deno = sqrt(sum_aver_x2)*sqrt(sum_aver_y2)
    if(deno == 0):
        return 0
    else:
        return sum_aver_xy/deno
    pass

'''
K最相邻算法
'''
