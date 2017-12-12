from dicts import *
from poptim import *
from scipy import optimize
import time
import pickle
from Basic2 import *
from Complex import *
from Inference import *
import time


d = Dict("train.wtag")
model = Basic2(d)
model.vec = pickle.load(open('weights_vec/v_basic2_unlimited', 'rb'))

t = time.time()
res = Inferece(model)
res.eval(model)
print("time: ",time.time()-t)