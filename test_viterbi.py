from dicts import *
from poptim import *
from scipy import optimize
import time
import pickle
from Basic2 import *
from Complex import *
from Inference import *
import time
from Parser import *
#complex

# d = Dict("train.wtag")
# model = Complex(d)
# model.vec = pickle.load(open('weights_vec/v_complex_30_10', 'rb'))
#
# t = time.time()
# res = Inferece(model)
# res.eval_train(model,"vec_results_train/v_complex_30_10_alltags.txt")
# print("time: ",time.time()-t)

#basic

# d = Dict("train.wtag")
# model = Basic2(d)
# model.vec = pickle.load(open('weights_vec/v_basic2_15', 'rb'))
#
# t = time.time()
# res = Inferece(model)
# res.eval(model,"vec_results_train/v_basic2_15.txt")
# print("time: ",time.time()-t)

#test

d = Dict("train.wtag")
# print(d.tags_dist_sorted)
model = Complex(d)
model.vec = pickle.load(open('weights_vec/v_complex_30_10', 'rb'))
#
t = time.time()
parsed = Parser("test.wtag")
# print(parsed.tag_sentence)
res = Inferece(model,parsed)

res.eval_test(parsed,"vec_results_test/v_complex_30_10_opt_1.txt")
print("time: ",time.time()-t)