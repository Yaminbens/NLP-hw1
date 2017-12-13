import re
import numpy as np


class Complex:

    def __init__(self, dic):
        self.word_tag_len = dic.word_tag_len
        self.tags_len = dic.tags_len
        self.words_idx = dic.words_idx
        self.tags_idx = dic.tags_idx
        self.word_tag_idx = dic.word_tag_idx
        self.word_seen_tags = dic.word_seen_tags


        self.tags_list = dic.tags_idx.keys()

        self.words_tag_idx = dic.words_idx
        self.tags_idx_t1 = {}
        self.tags_idx_t2 = {}
        self.tags_idx_t = {}
        self.vec = []
        self.tags_dist = dic.tags_dist
        self.max_tag = dic.max_tag
        self.word_sentence = dic.word_sentence
        self.tag_sentence = dic.tag_sentence
        self.tags_dist_sorted = dic.tags_dist_sorted

        #shift in vector calculation
        for t in dic.tags_idx.keys():
            self.tags_idx_t1.update({t: dic.tags_idx[t] + dic.word_tag_len}) #f103
            self.tags_idx_t2.update({t: dic.tags_idx[t] + dic.word_tag_len + dic.tags_len}) #f104
            self.tags_idx_t.update({t: dic.tags_idx[t] + dic.word_tag_len + 2*dic.tags_len}) #f105

        self.prefix_idx = {}
        self.suffix_idx = {}
        self.word_prefix_len = dic.word_prefix_len
        self.word_suffix_len = dic.word_suffix_len

        # shift in vector calculation
        for w in dic.word_prefix:
            self.prefix_idx.update({w: dic.word_prefix[w] + dic.word_tag_len + 3*dic.tags_len}) #f101

        # shift in vector calculation
        for w in dic.word_suffix:
            self.suffix_idx.update({w: dic.word_suffix[w]+ dic.word_prefix_len + dic.word_tag_len + 3*dic.tags_len}) #f102

        self.feat_vec_len = dic.word_suffix_len + dic.word_prefix_len + dic.word_tag_len + 3*dic.tags_len

    def feat_vec(self, w1, t, t_1, t_2):
        '''
        return feature vector for complex model f100,f103,f104, f105, f101, f102
        '''

        fw_t = np.zeros(self.word_tag_len)
        try:
            fw_t[self.word_tag_idx[w1+t]] = 1
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

        ft = np.zeros(self.tags_len)
        try:
            ft[self.tags_idx[t]] = 1
        except:
            pass

        fpre = np.zeros(self.word_prefix_len)
        try:
            if len(w1) > 4:
                fpre[self.prefix_idx[w1[:4]]] = 1
            else:
                fpre[self.prefix_idx[w1]] = 1
        except:
            pass

        fsuf = np.zeros(self.word_suffix_len)
        try:
            if len(w1) > 4:
                fsuf[self.suffix_idx[w1[-4:]]] = 1
            else:
                fsuf[self.suffix_idx[w1]] = 1
        except:
            pass
        # print("right",np.shape(np.concatenate((f0w, f0t, f3, f4), axis=0)))
        return np.concatenate((fw_t, f1t, f2t, ft,fpre, fsuf), axis=0)

    def calc_f_v(self, w1, t, t_1, t_2, v):
        """
        return v*f(x,y) for complex model
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

        try:
            v4 = v[self.word_tag_idx[t]]
        except:
            v4 = 0.0

        try:
            if len(w1) > 4:
                v5 = v[self.prefix_idx[w1[:4]]]
            else:
                v5 = v[self.prefix_idx[w1]]
        except:
            v5 = 0.0

        try:
            if len(w1) > 4:
                v6 = v[self.suffix_idx[w1[-4:]]]
            else:
                v6 = v[self.suffix_idx[w1]]
        except:
            v6 = 0.0

        return v1 + v2 + v3 + v4 +v5 +v6

    def calc_denom(self, w, set_t, t_1, t_2):
        """
        return sum(exp(v*f(x,y)) for all tags for complex model
        """
        sum = 0.0
        for t in set_t:
            sum += np.exp(self.calc_f_v(w, t, t_1, t_2, self.vec))
        return sum