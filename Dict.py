# words 121816

import re
import numpy as np
import operator
from Parser import *

class Dict:

    def __init__(self, file):
        cntr_w = 0
        cntr_t = 0
        cntr_wt = 0
        cntr_prefix = 0
        cntr_suffix = 0

        parse = Parser(file)
        parse.prefix_suffix_dist()
        self.prefix_dist = parse.prefix_dist
        self.suffix_dist = parse.suffix_dist

        parse.filter_pre_suf()
        self.prefix_filtered = parse.prefix_filtered
        self.suffix_filtered = parse.suffix_filtered

        parse.word_tag_distrib()
        self.word_tag_dist = parse.word_tag_dist

        parse.filter_word_tag()
        self.word_tag_filtered = parse.word_tag_filtered

        with open(file, 'r') as f:
            self.words_idx = {}
            self.words_cnt = {}
            self.tags_idx = {}
            self.word_seen_tags = {}
            self.word_tag_idx = {}

            self.word_prefix = {}
            self.word_suffix = {}
            self.word_prefix_dist = {}
            self.word_suffix_dist = {}

            self.word_sentence = []
            self.tag_sentence = []
            self.tags_dist = {} #tag distribution

            self.tags_tri_dist = {}

            for line in f:

                tmp_word_sentence = []
                tmp_tag_sentence = []
                t_2 = "*"
                t_1 = "*"
                match = re.split("\\s+", line)
                for i,w in enumerate(match):
                    if w == "":
                        continue

                    word = re.findall("[^_]*", w)
                    # print(word)
                    # break

                    #sentences
                    tmp_word_sentence.append(word[0])
                    tmp_tag_sentence.append(word[2])


                    #words
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

                    #dict of words and their seen tags

                    if word[0] not in self.word_seen_tags:
                        self.word_seen_tags.update({word[0]: [word[2]]})
                    elif word[2] not in self.word_seen_tags[word[0]]:
                        self.word_seen_tags[word[0]].extend([word[2]])

                    #dict of words and tags
                    if word[0]+word[2] not in self.word_tag_idx and word[0]+word[2] in self.word_tag_filtered:
                        self.word_tag_idx.update({word[0]+word[2]: cntr_wt})
                        cntr_wt += 1

                    #dict of suffixes and prefixes
                    if len(word[0]) > 4:
                        for i in range(1, 4):
                            if word[0][:i]+word[2] not in self.word_prefix and word[0][:i] in self.prefix_filtered:
                                self.word_prefix.update({word[0][:i]+word[2]: cntr_prefix})
                                cntr_prefix+=1
                            if word[0][-i:]+word[2] not in self.word_suffix and word[0][:i] in self.suffix_filtered:
                                self.word_suffix.update({word[0][-i:]+word[2]: cntr_suffix})
                                cntr_suffix += 1
                    else:
                        for i in range(1, len(word[0])):
                            if word[0][:i]+word[2] not in self.word_prefix and word[0][:i] in self.prefix_filtered:
                                self.word_prefix.update({word[0][:i]+word[2]: cntr_prefix})
                                cntr_prefix+=1
                            if word[0][-i:]+word[2] not in self.word_suffix and word[0][:i] in self.suffix_filtered:
                                self.word_suffix.update({word[0][-i:]+word[2]: cntr_suffix})
                                cntr_suffix += 1

                    #tags distribution
                    if word[2] not in self.tags_dist:
                        self.tags_dist.update({word[2]: 1})
                    else:
                        self.tags_dist[word[2]] += 1

                    #tags trigrams dist
                    if t_2 not in self.tags_tri_dist:
                        self.tags_tri_dist.update({t_2:{}})
                    if t_1 not in self.tags_tri_dist[t_2]:
                        self.tags_tri_dist[t_2].update({t_1: {}})
                    if word[2] not in self.tags_tri_dist[t_2][t_1]:
                        self.tags_tri_dist[t_2][t_1].update({word[2]:1})
                    else:
                        self.tags_tri_dist[t_2][t_1][word[2]] += 1

                    t_2 = t_1
                    t_1 = word[2]






                self.word_sentence.append(tmp_word_sentence)
                self.tag_sentence.append(tmp_tag_sentence)

            #update START
            self.tags_idx.update({"*": cntr_t})
            self.words_idx.update({"*": cntr_t})
            self.word_tag_idx.update({"**": cntr_wt})
            # print(len(self.word_tag_idx))

            #calculate lengths
            self.words_len = len(self.words_idx)
            self.tags_len = len(self.tags_idx)
            self.word_tag_len = len(self.word_tag_idx)
            self.word_prefix_len = len(self.word_prefix)
            self.word_suffix_len =len(self.word_suffix)

            self.tags_dist_sorted = list(x[0] for x in sorted(self.tags_dist.items(), key=operator.itemgetter(1)))


            #calculate maximals
            self.max_tag = max(self.tags_dist.items(), key=operator.itemgetter(1))[0]
            self.max_tri = {}
            for t_2 in self.tags_tri_dist:
                for t_1 in self.tags_tri_dist[t_2]:
                    if t_2 not in self.max_tri:
                        self.max_tri.update({t_2:{}})
                    if t_1 not in self.max_tri[t_2]:
                        self.max_tri[t_2].update({t_1: ""})
                    self.max_tri[t_2][t_1] = max(self.tags_tri_dist[t_2][t_1].items(), key=operator.itemgetter(1))[0]





