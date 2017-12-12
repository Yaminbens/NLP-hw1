# words 121816

import _pickle
import re
import numpy as np
import operator

class Dict:

    def __init__(self, file):
        cntr_w = 0
        cntr_t = 0
        cntr_wt = 0
        cntr_prefix = 0
        cntr_suffix = 0
        with open(file, 'r') as f:
            self.words_idx = {}
            self.words_cnt = {}
            self.tags_idx = {}
            self.word_seen_tags = {}
            self.word_tag_idx = {}
            self.word_prefix = {}
            self.word_suffix = {}
            self.word_sentence = []
            self.tag_sentence = []
            self.tags_dist = {} #tag distribution

            for line in f:

                tmp_word_sentence = []
                tmp_tag_sentence = []
                match = re.split("\\s+", line)
                for w in match:
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
                    if word[0]+word[2] not in self.word_tag_idx:
                        self.word_tag_idx.update({word[0]+word[2]: cntr_wt})
                        cntr_wt += 1

                    #dict of suffixes and prefixes
                    if len(word[0]) > 4:
                        if word[0][:4]+word[2] not in self.word_prefix:
                            self.word_prefix.update({word[0][:4]+word[2]: cntr_prefix})
                            cntr_prefix+=1
                        if word[0][-4:]+word[2] not in self.word_suffix:
                            self.word_suffix.update({word[0][-4:]+word[2]: cntr_suffix})
                            cntr_suffix += 1
                    else:
                        if word[0]+word[2] not in self.word_prefix:
                            self.word_prefix.update({word[0]+word[2]: cntr_prefix})
                            cntr_prefix+=1
                        if word[0]+word[2] not in self.word_suffix:
                            self.word_suffix.update({word[0]+word[2]: cntr_suffix})
                            cntr_suffix+=1

                    #tags distribution
                    if word[2] not in self.tags_dist:
                        self.tags_dist.update({word[2]: 1})
                    else:
                        self.tags_dist[word[2]] += 1


                self.word_sentence.append(tmp_word_sentence)
                self.tag_sentence.append(tmp_tag_sentence)

            #update START
            self.tags_idx.update({"*": cntr_t})
            self.words_idx.update({"*": cntr_t})
            self.word_tag_idx.update({"**": cntr_wt})

            #calculate lengths
            self.words_len = len(self.words_idx)
            self.tags_len = len(self.tags_idx)
            self.word_tag_len = len(self.word_tag_idx)
            self.word_prefix_len = len(self.word_prefix)
            self.word_suffix_len =len(self.word_suffix)
            self.max_tag = max(self.tags_dist.items(), key=operator.itemgetter(1))[0]




