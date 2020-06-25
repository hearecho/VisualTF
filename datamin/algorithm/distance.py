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
