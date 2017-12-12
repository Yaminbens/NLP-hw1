import re
import numpy as np


class Basic2:

    def __init__(self, dic):
        self.word_tag_len = dic.word_tag_len
        self.tags_len = dic.tags_len
        self.words_idx = dic.words_idx
        self.tags_idx = dic.tags_idx
        self.word_tag_idx = dic.word_tag_idx
        self.word_seen_tags = dic.word_seen_tags

        self.feat_vec_len = self.word_tag_len + 2 * self.tags_len
        self.tags_list = dic.tags_idx.keys()

        self.words_tag_idx = dic.words_idx
        self.tags_idx_t1 = {}
        self.tags_idx_t2 = {}
        self.vec = []
        self.tags_dist = dic.tags_dist
        self.max_tag = dic.max_tag
        self.word_sentence = dic.word_sentence
        self.tag_sentence = dic.tag_sentence

        for t in dic.tags_idx.keys():
            self.tags_idx_t1.update({t: dic.tags_idx[t] + dic.word_tag_len})
            self.tags_idx_t2.update({t: dic.tags_idx[t] + dic.word_tag_len + dic.tags_len})

    def feat_vec(self, w1, t, t_1, t_2):
        '''
        return feature vector for basic model f100,f103,f104
        '''

        f0w = np.zeros(self.word_tag_len)
        try:
            f0w[self.word_tag_idx[w1+t]] = 1
        except:
            pass

        f1t = np.zeros(self.tags_len)
        try:
            f1t[self.tags_idx[t_1]] = 1
        except:
            pass

        f2t = np.zeros(self.tags_len)

        try:
            f2t[self.tags_idx[t_2]] = 1
        except:
            pass
        # print("right",np.shape(np.concatenate((f0w, f0t, f3, f4), axis=0)))
        return np.concatenate((f0w, f1t, f2t), axis=0)

    def calc_f_v(self, w1, t, t_1, t_2, v):
        """
        return v*f(x,y) for basic model
        """

        try:
            v1 = v[self.word_tag_idx[w1 + t]]
        except:
            v1 = 0.0

        try:
            v2 = v[self.tags_idx_t1[t_1]]
        except:
            v2 = 0.0

        try:
            v3 = v[self.tags_idx_t2[t_2]]
        except:
            v3 = 0.0

        return v1 + v2 + v3

    def calc_denom(self,w, t, t_1, set_t_2):
        """
        return sum(exp(v*f(x,y)) for all tags for basic model
        """
        sum = 0.0
        for t_2 in set_t_2:
            sum += np.exp(self.calc_f_v(w,t, t_1,t_2,self.vec))
        return sum

