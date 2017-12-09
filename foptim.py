from scipy import optimize
import re
from dicts import Dict
import numpy as np


def sig(v, dic, w0, t, t_1, t_2, sum_z):
    '''
    :param v: wight vec
    :param dic: dictionary
    :param w0: current word
    :param t: current tag
    :param t_1: last tag
    :param t_2: second last tag
    :param sum_z: sum on exponents
    :return: sigmoid
    '''
    return np.exp(Dict.calc_f_v(dic, w0, t, t_1, t_2, v))/sum_z


def L_v_func(v, file_name, lamda, dic):
    v_f = 0.0
    e_v_f = 0.0
    lam_v_2 = 0.0

    with open(file_name, 'r') as f:
        for line in f:
            t_1 = "*"
            t_2 = "*"

            match = re.split("\\s+", line)
            for w in match:

                if w == "":
                    continue

                word = re.findall("[^_]*", w)

                e_v_f = np.add(e_v_f, np.log(np.sum([np.exp(Dict.calc_f_v(dic,word[0],y,t_1,t_2,v))
                                                     for y in dic.tags_list])))
                v_f = np.add(v_f, Dict.calc_f_v(dic,word[0],word[2],t_1,t_2,v))

                #update last tags
                t_2 = t_1
                t_1 = word[2]

    return -(v_f-e_v_f-(0.5*lamda*np.dot(v,v)))

def dL_func(v, file_name, lamda, dic):
    '''

    :param v:
    :param file_name:
    :param lamda:
    :param dic:
    :return: grads vector
    '''
    f_f = np.zeros(len(v))
    f_p = np.zeros(len(v))

    with open(file_name, 'r') as f:
        for line in f:
            t_1 = "*"
            t_2 = "*"

            match = re.split("\\s+", line)
            for w in match:

                if w == "":
                    continue

                word = re.findall("[^_]*", w)

                # print(Dict.feat_vec(dic,word[0],word[2],t_1,t_2))
                f_f = np.add(f_f,Dict.feat_vec(dic,word[0],word[2],t_1,t_2))
                sum_z = np.sum([np.exp(Dict.feat_vec(dic, word[0], z, t_1, t_2)) for z in dic.tags_list])
                f_p = np.add(f_p,np.sum(
                    [np.dot(Dict.feat_vec(dic,word[0],y,t_1,t_2),sig(v,dic,word[0],word[2],t_1,t_2,sum_z))
                     for y in dic.tags_list]))

                # update last tags
                t_2 = t_1
                t_1 = word[2]

    return f_f - f_p - lamda * v





def main():
    dictionary = Dict("train.wtag")
    v = np.zeros()
