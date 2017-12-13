from scipy import optimize
import re
from dicts import Dict
import numpy as np

# with open("train1.wtag", 'r') as f:
#     for line in f:
#         match = re.findall('[^ ]*|[^_]*', line)
#         print(match)
#         break
# zz = 'xcvbnm'
# z = "sdsff"
# print(zz[-4:])


# a = np.ones(5)
# b= np.ones(5)*2
# a+=b
# print(a)

d = Dict("train.wtag")
print(d.max_tri)

# ll = [[2,4],[5,7,3],[1,2]]
# print(ll[5][1])


