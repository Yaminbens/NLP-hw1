from scipy import optimize
import re
from Dict import Dict
import numpy as np
from Basic2 import *
import pickle

d = Dict("train.wtag")
model = Basic2(d)
model.vec = pickle.load(open('weights_vec/v_basic2_unlimited', 'rb'))

print(model.calc_f_v("The","DT","*","*",model.vec))
print(model.calc_f_v("The","NPP","*","*",model.vec))


