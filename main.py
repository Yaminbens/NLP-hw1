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
    model_basic = Basic(dic)
    v = np.ones(model_basic.feat_vec_len)
    model_basic.vec, f, dicto = optimize.fmin_l_bfgs_b(L_v_func, v, dL_func, #TODO edit parameters
                                                          ("train.wtag", LAMBDAb, model_basic), maxiter=MAXITERb, factr=FACTRb)
                                                       # ("train1.wtag", LAMBDA, model_basic))
    print(model_basic.vec)
    print(f)
    print(dicto)
    pickle.dump(model_basic.vec, open("weights_vec/"+VECSAVEb, 'wb'))
    return model_basic

def test_basic(model_basic):
    parsed = Parser("test.wtag")
    result_basic = Inferece(model_basic, parsed)
    result_basic.eval_test(VECSAVEb)
    result_basic.print_confusion("basic_conf")

def competition_basic(model_basic):
    parsed = Parser("comp.wtag")
    result_basic = Inferece(model_basic, parsed)
    result_basic.tag_text("comp_m1_305056293.wtag")



def train_complex(dic):
    model_complex = Complex(dic)
    v = np.ones(model_complex.feat_vec_len)
    model_complex.vec, f, dicto = optimize.fmin_l_bfgs_b(L_v_func, v, dL_func, #TODO edit parameters
                                                         ("train.wtag", LAMBDAc, model_complex), maxiter=MAXITERc, factr=FACTRc)

    print(model_complex.vec)
    print(f)
    print(dicto)
    pickle.dump(model_complex.vec, open("weights_vec/"+VECSAVEc, 'wb'))
    return model_complex

def test_complex(model_complex):
    parsed = Parser("test.wtag")
    result_basic = Inferece(model_complex, parsed)
    result_basic.eval_test(VECSAVEc)
    result_basic.print_confusion("complex_conf")

def competition_complex(model_complex):
    parsed = Parser("comp.wtag")
    result_basic = Inferece(model_complex, parsed)
    result_basic.tag_text("comp_m2_305056293.wtag")




def main():

    #Parsing and initiating dictionaries for features vector
    dictionary = Dict("train.wtag")

    # # Training - Basic Model
    # model_basic = train_basic(dictionary)
    # # Test - Basic Model
    # test_basic(model_basic)
    # # Competition
    # # competition_basic(model_basic)

    # Training - Complex Model
    model_complex = train_complex(dictionary)
    # Test - Basic Model
    test_complex(model_complex)
    # Competition
    # competition_complex(model_basic)



if __name__ == "__main__":
    main()