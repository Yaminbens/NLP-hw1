from dicts import *
from poptim import *
from scipy import optimize
import time
import pickle
from Basic2 import *
from Complex import *
from Inference import *

def main():
    t = time.time()
    d = Dict("train.wtag")

    model = Basic2(d)
    v = np.ones(model.feat_vec_len)

    # model = Complex(d)
    # v = np.ones(model.feat_vec_len)
    # print(d.tags_list)
    # zzz = L_v_func(v, "train1.wtag", 0.1, c)
    # zzz = dL_func(v,"train1.wtag", 0.1,c)
    # print(np.shape(zzz))
    # print(zzz)
    # print(time.time()-t)
    # zzzzz = optimize.check_grad(L_v_func,dL_func,[v], "train1.wtag",1,d)
    model.vec, f, diccc = optimize.fmin_l_bfgs_b(L_v_func,v, dL_func,("train1.wtag",1,model))#, maxiter=30,factr=10.0)
    # iprint = 99, factr=10.0, maxiter=15)
    print(model.vec)
    print(f)
    print(diccc)
    pickle.dump(model.vec, open("v_basic_train1", 'wb'))

    # for xx in list(x):
    #     print(xx)


        # # print(d.words_idx)

    # for w in dictionary.word_tag.keys():
    #     print(w," ",dictionary.word_tag[w])
    # print(dictionary.tags_idx)


if __name__ == "__main__":
    main()