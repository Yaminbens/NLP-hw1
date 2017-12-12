import _pickle
import re
import numpy as np


class Basic:

    def __init__(self, dic):
        self.words_len = dic.words_len
        self.tags_len = dic.tags_len
        self.words_idx = dic.words_idx
        self.tags_idx = dic.tags_idx

        self.feat_vec_len = self.words_len + 3 * self.tags_len
        self.tags_list = dic.tags_idx.keys()

        self.words_idx_w = dic.words_idx
        self.tags_idx_t = {}
        self.tags_idx_t1 = {}
        self.tags_idx_t2 = {}

        for t in dic.tags_idx.keys():
            self.tags_idx_t.update({t: dic.tags_idx[t] + dic.words_len})
            self.tags_idx_t1.update({t: dic.tags_idx[t] + dic.words_len + dic.tags_len})
            self.tags_idx_t2.update({t: dic.tags_idx[t] + dic.words_len + 2 * dic.tags_len})

    def feat_vec(self, w1, t, t_1, t_2):
        '''
        return feature vector for basic model f100,f103,f104
        '''

        f0w = np.zeros(self.words_len)
        try:
            f0w[self.words_idx[w1]]=1
        except:
            pass

        f0t = np.zeros(self.tags_len)
        try:
            f0t[self.tags_idx[t]]= 1
        except:
            pass

        f1t = np.zeros(self.tags_len)
        try:
            f1t[self.tags_idx[t_1]]= 1
        except:
            pass

        f2t = np.zeros(self.tags_len)
        try:
            f2t [self.tags_idx[t_2]] = 1
        except:
            pass
        # print("right",np.shape(np.concatenate((f0w, f0t, f3, f4), axis=0)))
        return np.concatenate((f0w,f0t,f1t,f2t), axis=0)

    def calc_f_v(self, w1, t, t_1, t_2,v):
        """
        return v*f(x,y) for basic model
        """

        try:
            v1 = v[self.words_idx_w[w1]]
        except:
            v1 = 0.0

        try:
            v2 = v[self.tags_idx_t[t]]
        except:
            v2 = 0.0

        try:
            v3 = v[self.tags_idx_t1[t_1]]
        except:
            v3 = 0.0

        try:
            v4 = v[self.tags_idx_t2[t_2]]
        except:
            v4 = 0.0

        return v1+v2+v3+v4