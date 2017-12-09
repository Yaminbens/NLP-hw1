# words 121816

import _pickle
import re
import numpy as np


class Dict:

    def __init__(self, file):
        cntr_w = 0
        cntr_t = 0
        with open(file, 'r') as f:
            self.words_idx = {}
            self.words_cnt = {}
            self.tags_idx = {}
            self.word_tag = {}

            for line in f:

                match = re.split("\\s+", line)
                for w in match:
                    if w == "":
                        continue

                    word = re.findall("[^_]*", w)
                    # print(word)
                    # break
                    #f100
                    if word[0] not in self.words_idx:
                        self.words_idx.update({word[0]: cntr_w})
                        self.words_cnt.update({word[0]: 1})
                        cntr_w += 1
                    else:
                        self.words_cnt[word[0]] += 1
                    #tags
                    if word[2] not in self.tags_idx:
                        self.tags_idx.update({word[2]: cntr_t})
                        cntr_t += 1

                    if word[0] not in self.word_tag:
                        self.word_tag.update({word[0]: [word[2]]})
                    elif word[2] not in self.word_tag[word[0]]:
                        self.word_tag[word[0]].extend([word[2]])

            self.tags_idx.update({"*": cntr_t})
            # self.tags_idx.update({"STOP": cntr_t+1})
            self.words_idx.update({"*": cntr_t})
            # self.words_idx.update({"STOP": cntr_t + 1})
            self.words_len = len(self.words_idx)
            # print("word_len",self.words_len)
            self.tags_len = len(self.tags_idx)
            # print("tags_len", self.tags_len)
            self.feat_vec_len = self.words_len+3*self.tags_len
            self.tags_list = self.tags_idx.keys()

            self.tags_idx_s1 = {}
            self.tags_idx_s2 = {}
            self.tags_idx_s3 = {}

            for t in self.tags_idx.keys():
                self.tags_idx_s1.update({t : self.tags_idx[t]+self.words_len})
                self.tags_idx_s2.update({t : self.tags_idx[t] + self.words_len+self.tags_len})
                self.tags_idx_s3.update({t : self.tags_idx[t] + self.words_len + 2*self.tags_len})

    #returns feature vector for basic model f100,f103,f104
    def feat_vec(self, w1, t, t_1, t_2):

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

        f3 = np.zeros(self.tags_len)
        try:
            f3[self.tags_idx[t_1]]= 1
        except:
            pass

        f4 = np.zeros(self.tags_len)
        try:
            f4 [self.tags_idx[t_2]] = 1
        except:
            pass
        # print("right",np.shape(np.concatenate((f0w, f0t, f3, f4), axis=0)))
        return np.concatenate((f0w,f0t,f3,f4), axis=0)

    def calc_f_v(self, w1, t, t_1, t_2,v):
        """
        return v*f(x,y)
        """

        try:
            v1 = v[self.words_idx[w1]]
        except:
            print("103")
            v1 = 0.0

        try:
            v2 = v[self.tags_idx_s1[t]]
        except:
            print("109")
            v2 = 0.0

        try:
            v3 = v[self.tags_idx_s2[t_1]]
        except:
            print("115")
            v3 = 0.0

        try:
            v4 = v[self.tags_idx_s3[t_2]]
        except:
            print("121")
            v4 = 0.0

        return v1+v2+v3+v4