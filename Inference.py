import numpy as np
from Basic2 import *
from Complex import *
#need to supply calc_prob,word_tags see explanation in mail

class Inferece:
    def __init__(self,model,parser):

        self.results = []
        for sentence in parser.word_sentence:
            self.results.append(self.viterbi(sentence,model))

    def viterbi(self,sentence, model):
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

            if sentence[k] in model.word_seen_tags:
                set_v = model.word_seen_tags[sentence[k]]
            else:
                set_v = model.tags_dist_sorted[-3:]
            for v in set_v:  # todo: need to limit to 3-4 max tags
                if k == 0:
                    set_u = ['*']
                else:
                    if sentence[k - 1]in model.word_seen_tags:
                        set_u = model.word_seen_tags[sentence[k - 1]]  # todo: word_tags has to take into account: 1)unseen words 2)numbers,3)seen words
                    else:
                        set_u = model.tags_dist_sorted[-3:]
                    # set_u = model.tags_list
                for u in set_u:
                    max_value, max_tag = 0, "dummy"
                    if k == 0 or k == 1:
                        set_t = ['*']
                    else:
                        if sentence[k - 2] in model.word_seen_tags:
                            set_t = model.word_seen_tags[sentence[k - 2]]
                        else:
                            set_t = model.tags_dist_sorted[-3:]
                        # set_t = model.tags_list

                    for t in set_t:
                        denom = model.calc_denom(sentence[k], set_v, u, t)
                        tmp_calc = model.calc_f_v(sentence[k], v, u, t,model.vec)
                        result = np.exp(tmp_calc)/denom  # todo calc_prob
                        if k > 0:
                            try:
                                result *= pi[(t,u)]
                            except:
                                pass
                            # print("k:",k)
                            # result *= pi[(t, u)]
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

    def eval_test(self,parser,filename):

        accuracy = 0.0
        missed = 0.0

        file = open(filename, 'w')

        for i,sentence in enumerate(parser.tag_sentence):
            for j,word in enumerate(sentence):
                if self.results[i][j] == parser.tag_sentence[i][j]:
                    accuracy+=1
                else:
                    missed+=1
                    if parser.word_sentence[i][j] in parser.word_sentence:
                        seen = 'seen'
                    else:
                        seen = 'unseen'

                    file.write("guess: "+self.results[i][j]+" true: "+parser.tag_sentence[i][j]+ " word: "+parser.word_sentence[i][j]+" "+seen+"\n")

        print("correct: ", 100*accuracy/(accuracy+missed))

