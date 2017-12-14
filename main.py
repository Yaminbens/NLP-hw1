from Dict import *
from LLMoptim import *
from scipy import optimize
import pickle
from Basic import *
from Complex import *
from Inference import *
from utils import *
from time import time

def train_basic(dic):
    t = time()

    model_basic = Basic(dic)
    v = np.ones(model_basic.feat_vec_len)
    model_basic.vec, f, dicto = optimize.fmin_l_bfgs_b(L_v_func, v, dL_func, #TODO edit parameters
                                                          ("train.wtag", LAMBDA, model_basic), maxiter=MAXITER, factr=FACTR)
                                                       # ("train1.wtag", LAMBDA, model_basic))
    print(model_basic.vec)
    print(f)
    print(dicto)
    pickle.dump(model_basic.vec, open("basic_vec_nolimits", 'wb'))
    print(time()-t)
    return model_basic


def train_complex(dic):
    model_complex = Complex(dic)
    v = np.ones(model_complex.feat_vec_len)
    model_complex.vec, f, dicto = optimize.fmin_l_bfgs_b(L_v_func, v, dL_func, #TODO edit parameters
                                                         ("train.wtag", LAMBDA, model_complex), maxiter=MAXITER, factr=FACTR)
    print(model_complex.vec)
    print(f)
    print(dicto)
    pickle.dump(model_complex.vec, open("complex_vec_"+MAXITER+"_"+FACTR, 'wb'))
    return model_complex

def main():

    #Parsing and initiating dictionaries for features vector
    dictionary = Dict("train.wtag")

    # Training - Basic Model
    model_basic = train_basic(dictionary)

    #Training - Complex Model
    # model_complex = train_complex(dictionary)

    #


if __name__ == "__main__":
    main()