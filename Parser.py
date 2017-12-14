import re

class Parser:

    def __init__(self, file):

        with open(file, 'r') as f:
            #for prefix suffix dist
            self.prefix_dist = {}
            self.suffix_dist = {}

            self.prefix_filtered = []
            self.suffix_filtered = []

            self.word_tag_dist = {}
            self.word_tag_filtered = []

            self.word_sentence = []
            self.tag_sentence = []



            for line in f:
                tmp_word_sentence = []
                tmp_tag_sentence = []

                match = re.split("\\s+", line)
                for i, w in enumerate(match):
                    if w == "":
                        continue

                    word = re.findall("[^_]*", w)

                    tmp_word_sentence.append(word[0])
                    tmp_tag_sentence.append(word[2])

                self.word_sentence.append(tmp_word_sentence)
                self.tag_sentence.append(tmp_tag_sentence)

    def word_tag_distrib(self):
        for i,sentence in enumerate(self.word_sentence):
            for j,word in enumerate(sentence):
                if self.word_sentence[i][j]+self.tag_sentence[i][j] not in self.word_tag_dist:
                    self.word_tag_dist.update({self.word_sentence[i][j]+self.tag_sentence[i][j] : 1})
                else:
                    self.word_tag_dist[self.word_sentence[i][j]+self.tag_sentence[i][j]] += 1

    def filter_word_tag(self):
        for wt in self.word_tag_dist.keys():
            if self.word_tag_dist[wt] > 4:
                self.word_tag_filtered.append(wt)


    def prefix_suffix_dist(self):
        for sentence in self.word_sentence:
            for word in sentence:
                if len(word) > 4:
                    for i in range(1,4):
                        if word[:i] not in self.prefix_dist:
                            self.prefix_dist.update({word[:i]: 1})
                        else:
                            self.prefix_dist[word[:i]] += 1
                        if word[-i:] not in self.suffix_dist:
                            self.suffix_dist.update({word[-i:]: 1})
                        else:
                            self.suffix_dist[word[-i:]] += 1
                else:
                    for i in range(1,len(word)):
                        if word[:i] not in self.prefix_dist:
                            self.prefix_dist.update({word[:i]: 1})
                        else:
                            self.prefix_dist[word[:i]] += 1
                        if word[-i:] not in self.suffix_dist:
                            self.suffix_dist.update({word[-i:]: 1})
                        else:
                            self.suffix_dist[word[-i:]] += 1

    def filter_pre_suf(self):
        for pre in self.prefix_dist.keys():
            if self.prefix_dist[pre] < 8:
                self.prefix_filtered.append(pre)
        for suf in self.suffix_dist.keys():
            if self.suffix_dist[suf] < 8:
                self.suffix_filtered.append(suf)

