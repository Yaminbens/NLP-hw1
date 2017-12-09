from scipy import optimize
import re
from dicts import Dict
import numpy as np

# with open("train1.wtag", 'r') as f:
#     for line in f:
#         match = re.split(' |_*', line)
#         print(match)
#         break

a = np.ones(5)
b= np.ones(5)*2
a+=b
print(a)