import numpy as np
from Basic2 import *
from Complex import *
import csv
import pandas as pd

#need to supply calc_prob,word_tags see explanation in mail

class Inferece:
    def __init__(self,model,parser):
        self.confusion = {}
        self.results = []
        self.parser = parser
        self.model = model
        self.N = 0
        for sentence in parser.word_sentence:
            self.results.append(self.viterbi(sentence,model))


    def calc_set(self,word):
        if word in self.model.word_seen_tags:
            set_tag = self.model.word_seen_tags[word]
        elif word[0].isdigit():
            set_tag = ['CD']
        else:
            set_tag = self.model.tags_dist_sorted[-self.N:]
            # set_v.append('NNS')
            # set_v.append('CD')
        return set_tag

    def viterbi(self,sentence, model):
        self.N = 3
        if sentence == []:
            return sentence
        pi = {}
        back_pointer = []
        #for k in range(len(sentence) - 1):
            #back_pointer.append({})
        for k in range(len(sentence)):
            if k>1:
                back_pointer.append({})
            pi_aux = {}

            set_v = self.calc_set(sentence[k])

            for v in set_v:  # todo: need to limit to 3-4 max tags
                if k == 0:
                    set_u = ['*']
                else:
                    set_u = self.calc_set(sentence[k-1])
                for u in set_u:
                    max_value, max_tag = 0, "dummy"
                    if k == 0 or k == 1:
                        set_t = ['*']
                    else:
                        set_t = self.calc_set(sentence[k-2])

                    for t in set_t:
                        denom = model.calc_denom(sentence[k], set_v, u, t)
                        tmp_calc = model.calc_f_v(sentence[k], v, u, t,model.vec)
                        result = np.exp(tmp_calc)/denom  # todo calc_prob
                        if k > 0:
                            # try:
                            #     result *= pi[(t,u)]
                            # except:
                            #     pass
                            # print("k:",k)
                            result *= pi[(t, u)]
                        # find max over t
                        if result > max_value:
                            max_value = result
                            max_tag = t
                    pi_aux[(u,v)] = max_value
                    if k > 1:
                        back_pointer[k - 2][u] = max_tag
            pi = pi_aux
        # finding the path with bp
        tn_prev, tn = max(pi, key=pi.get)
        if len(sentence) == 1:
            return [tn]
        else:
            res = []
            current_tag = tn_prev
            for k in range(len(sentence) - 3, -1, -1): #length-3 because (prev,prev_tn) already extracted
                current_tag = back_pointer[k][current_tag]
                res.insert(0, current_tag)
            return res + [tn_prev] + [tn]

    # def eval_train(self,parser,filename):
    #
    #     accuracy = 0.0
    #     missed = 0.0
    #
    #     file = open(filename, 'w')
    #
    #     for i,sentence in enumerate(parser.tag_sentence):
    #         for j,word in enumerate(sentence):
    #             if self.results[i][j] == parser.tag_sentence[i][j]:
    #                 accuracy+=1
    #             else:
    #                 missed+=1
    #                 if parser.word_sentence[i][j] in parser.word_sentence:
    #                     seen = 'seen'
    #                 else:
    #                     seen = 'unseen'
    #
    #                 file.write("guess: "+self.results[i][j]+" true: "+parser.tag_sentence[i][j]+ " word: "+parser.word_sentence[i][j]+seen+"\n",'+w')
    #
    #     print("correct: ", 100*accuracy/(accuracy+missed))

    def eval_test(self,filename):

        accuracy = 0.0
        missed = 0.0

        cap_let = 0.0
        cap_word = 0.0
        num = 0.0

        cap_let_l = []
        cap_word_l = []
        num_l = []


        file = open(filename, 'w')

        for i,sentence in enumerate(self.parser.tag_sentence):
            for j,word in enumerate(sentence):
                if self.results[i][j] == self.parser.tag_sentence[i][j]:
                    accuracy+=1
                else:
                    missed+=1
                    if self.parser.word_sentence[i][j] in self.parser.word_sentence:
                        seen = 'seen'
                    else:
                        seen = 'unseen'
                    file.write("guess: "+self.results[i][j]+" true: "+self.parser.tag_sentence[i][j]+ " word: "+self.parser.word_sentence[i][j]+" "+seen+"\n")

                    #confusion matrix
                    if self.parser.tag_sentence[i][j] not in self.confusion:
                        self.confusion.update({self.parser.tag_sentence[i][j]:{}})
                    if self.results[i][j] not in self.confusion[self.parser.tag_sentence[i][j]]:
                        self.confusion[self.parser.tag_sentence[i][j]].update({self.results[i][j]:1})
                    else:
                        self.confusion[self.parser.tag_sentence[i][j]][self.results[i][j]]+=1

                    if len(self.parser.word_sentence[i][j]) == 1:
                        if self.parser.word_sentence[i][j][0].isupper():
                            cap_let +=1
                            cap_word += 1
                            cap_let_l.append(self.parser.word_sentence[i][j]+" | guess: "+self.results[i][j] +" | true: "+ self.parser.tag_sentence[i][j])
                            cap_word_l.append(self.parser.word_sentence[i][j]+" | guess: "+self.results[i][j] +" | true: "+ self.parser.tag_sentence[i][j])
                    elif len(self.parser.word_sentence[i][j]) > 1:
                        if self.parser.word_sentence[i][j][0].isupper():
                            cap_let += 1
                            cap_let_l.append(self.parser.word_sentence[i][j]+" | guess: "+self.results[i][j] +" | true: "+ self.parser.tag_sentence[i][j])
                            if self.parser.word_sentence[i][j][1].isupper():
                                cap_word += 1
                                cap_word_l.append(
                                    self.parser.word_sentence[i][j] + " | guess: " + self.results[i][j] + " | true: " +
                                    self.parser.tag_sentence[i][j])
                        if self.parser.word_sentence[i][j][0].isdigit():
                            num += 1
                            num_l.append(
                                    self.parser.word_sentence[i][j] + " | guess: " + self.results[i][j] + " | true: " +
                                    self.parser.tag_sentence[i][j])
        print("capital letters:",cap_let)
        print(cap_let_l)
        print("capital word: ", cap_word)
        print(cap_word_l)
        print("number: ",num)
        print(num_l)

        print("correct: ", 100*accuracy/(accuracy+missed))


    def print_confusion(self):

        # with open('conf.csv', 'w') as f:  # Just use 'w' mode in 3.x
        #     for true_tag in self.confusion:
        #         w = csv.DictWriter(f, self.confusion[true_tag].keys())
        #         w.writerow(true_tag)
        #         w.writeheader()
        #         w.writerow(self.confusion[true_tag])

        df = pd.DataFrame.from_dict(self.confusion)
        df = df.transpose()
        df.to_csv("conf.csv", encoding='utf-8')
        # file = open("confusion_matrix","w")
        # file.write("\nConfusion Matrix\n\n\n")
        # for true_tag in self.confusion:
        #     file.write("for tag "+true_tag+" false inferences are:\n\n")
        #     for false_tag in self.confusion[true_tag].keys():
        #         file.write(false_tag+": "+str(self.confusion[true_tag][false_tag])+"\n")
        #     file.write("\n\n")


    def tag_text(self,filename):

        with open(filename, 'w') as file:

            for i,sentence in enumerate(self.parser.word_sentence):
                for j,word in enumerate(sentence):
                    file.write(self.parser.word_sentence[i][j]+"_"+self.results[i][j]+" ")
                file.write("\n")

