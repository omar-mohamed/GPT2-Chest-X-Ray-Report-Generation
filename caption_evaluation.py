from nltk.translate import bleu_score
import nltk.translate.gleu_score as gleu
from nlgeval import NLGEval


def get_evalutation_scores(hypothesis, refrences, testing_mode=False):
    gleu_scores = {"Gleu_1": gleu.corpus_gleu(refrences, hypothesis, min_len=1, max_len=1),
                   "Gleu_2": gleu.corpus_gleu(refrences, hypothesis, min_len=1, max_len=2),
                   "Gleu_3": gleu.corpus_gleu(refrences, hypothesis, min_len=1, max_len=3),
                   "Gleu_4": gleu.corpus_gleu(refrences, hypothesis, min_len=1, max_len=4)
                   }

    if testing_mode:
        for i in range(len(hypothesis)):
            hypothesis[i] = ' '.join(hypothesis[i])

        refs = [[]]
        for i in range(len(refrences)):
            refs[0].append(' '.join(refrences[i][0]))
            if refs[0][-1] == "":
                refs[0][-1] = "no"
        refrences = refs

        n = NLGEval()
        scores = n.compute_metrics(ref_list=refrences, hyp_list=hypothesis)
    else:
        scores = {"Bleu_1": bleu_score.corpus_bleu(refrences, hypothesis, weights=[1.0]),
                  "Bleu_2": bleu_score.corpus_bleu(refrences, hypothesis, weights=[1. / 2, 1. / 2]),
                  "Bleu_3": bleu_score.corpus_bleu(refrences, hypothesis, weights=[1. / 3, 1. / 3, 1. / 3]),
                  "Bleu_4": bleu_score.corpus_bleu(refrences, hypothesis, weights=[1. / 4, 1. / 4, 1. / 4, 1. / 4])}

    for key, val in gleu_scores.items():
        scores[key] = val
    return scores
