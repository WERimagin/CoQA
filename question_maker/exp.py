import warnings
warnings.filterwarnings("ignore")
import nltk

from nltk.translate.bleu_score import sentence_bleu
reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
candidate = ['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']
score = sentence_bleu(reference, candidate)
print(score)

from nltk.translate.bleu_score import sentence_bleu
reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
candidate = ['the', 'fast', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']
score = sentence_bleu(reference, candidate)
print(score)

# two words different
from nltk.translate.bleu_score import sentence_bleu
reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
candidate = ['the', 'fast', 'brown', 'fox', 'jumped', 'over', 'the', 'sleepy', 'dog']
score = sentence_bleu(reference, candidate)
print(score)

from nltk.translate.bleu_score import sentence_bleu
reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
candidate = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'lazy',"over"]
score = sentence_bleu(reference, candidate)
print(score)


# very short
from nltk.translate.bleu_score import sentence_bleu
reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
candidate = ['the', 'quick',"brown","fox"]
score = sentence_bleu(reference, candidate)
print(score)

text1="An individual N-gram score is the evaluation of just matching grams"
text2="An individual N-gram score is the evaluation of just matching grams and my end"
text3="An individual N-gram scosre is the evaluation of just mastching grams and my end"

text1=text1.split()
text2=text2.split()
text3=text3.split()
print(nltk.bleu_score.sentence_bleu([text1],text2))
print(nltk.bleu_score.sentence_bleu([text1],text3))

#nltk.bleu_score.sentence_bleu([re,re],hy)


#nltk.bleu_score.sentence_bleu([[1,2,3]],[1,2,4,5,6])

print(nltk.bleu_score.sentence_bleu([[1,2,3]],[1,2,4,5,6]))

print(nltk.bleu_score.sentence_bleu([[1,2,3,4,5,5,5,5,5]],[1,2,4,5,6],weights=(1,0,0,0)))
print(nltk.bleu_score.sentence_bleu([[1,2,3,4]],[1,2,4,5,6],weights=(0,1,0,0)))
print(nltk.bleu_score.sentence_bleu([[1,2,3,4]],[1,2,4,5,6],weights=(0,0,1,0)))
print(nltk.bleu_score.sentence_bleu([[1,2,3,4]],[1,2,4,5,6],weights=(0,0,0,1)))

print(nltk.bleu_score.sentence_bleu([[1,2,3,4,5]],[1,2,3,7,4,6]))

print(nltk.bleu_score.sentence_bleu([[1,2,6,3,4,5]],[1,2,3,7,4,6,10]))

print(nltk.bleu_score.sentence_bleu([[1,2,3,4,5]],[1,2,3,7,4,6]))

#nltk.bleu_score.sentence_bleu([[1,2,3,4,5]],[1,2,3,4,4,6])
#print(nltk.bleu_score.sentence_bleu([re],hy))
