import _pickle
import re
import numpy as np
import operator

class Parser:

    def __init__(self, file):

        with open(file, 'r') as f:
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

