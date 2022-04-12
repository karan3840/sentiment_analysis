
import pandas as pd
import nltk
nltk.download('opinion_lexicon')
dataset = pd.read_csv("https://raw.githubusercontent.com/karan3840/sentiment_analysis/main/tripadvisor_hotel_reviews.csv")

from nltk.tokenize import treebank
from nltk.corpus import opinion_lexicon

pos_list=set(opinion_lexicon.positive())
neg_list=set(opinion_lexicon.negative())

tokenizer = treebank.TreebankWordTokenizer()

def sentiment(sentence):
    senti=0
    words = [word.lower() for word in tokenizer.tokenize(sentence)]
    for word in words:
        if word in pos_list:
            senti += 1
        elif word in neg_list:
            senti -= 1
    return senti

dataset['sentiment']=dataset['Review'].apply(sentiment)
p_count,n_count,z_count =0,0,0
for i in dataset['sentiment']:
    if i>0:
        p_count += 1
    elif i<0:
        n_count += 1
    else:
        z_count += 1

print("The total positive reviews in dataset are: ",p_count)
print("The total negative reviews in dataset are: ",n_count)
print("The total neutral reviews in dataset are: ",z_count)


import matplotlib.pyplot as plt
fig, ax1 = plt.subplots()
ax1.bar(['Positive', 'Negative','Neutral'], [p_count, n_count,z_count])
fig.autofmt_xdate()

plt.show()