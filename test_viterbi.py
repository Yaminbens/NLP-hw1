from dicts import *
from poptim import *
from scipy import optimize
import time
import pickle
from Basic2 import *
from Complex import *
from Inference import *

d = Dict("train.wtag")
model = Basic2(d)
model.vec = pickle.load(open('v_basic_train1', 'rb'))

res = Inferece(model)
res.eval(model)
