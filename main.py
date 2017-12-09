from dicts import *
from poptim import *
from scipy import optimize
import time
import pickle

def main():
    t = time.time()
    d = Dict("train.wtag")
    v = np.ones(d.feat_vec_len)
    # print(d.tags_list)
    # zzz = L_v_func(v, "train1.wtag", 0.1, d)
    # # zzz = dL_func(v,"train1.wtag", 0.1,d)
    # print(np.shape(zzz))
    # print(zzz)
    # print(time.time()-t)
    # zzzzz = optimize.check_grad(L_v_func,dL_func,[v], "train1.wtag",1,d)
    x, f, diccc = optimize.fmin_l_bfgs_b(L_v_func,v, dL_func,("train.wtag",0.1,d))
    print(x)
    print(f)
    print(diccc)
    pickle.dump(x, open("v_weights", 'wb'))


    # # print(d.words_idx)

    # for w in dictionary.word_tag.keys():
    #     print(w," ",dictionary.word_tag[w])
    # print(dictionary.tags_idx)


if __name__ == "__main__":
    main()