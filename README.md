# CollaboSentGen

1. Overview
This dataset is for the hackathon at <em>2018 TTIC Workshop: Collaborative & Knowledge-backed Language Generation</em>. 
It is divided into train/valid/test(70%/10%/20%) randomly,

2. Dataset Details

Each file is a csv file with following columns.

```storyid, title, sent1, sent2, sent3, sent4, sent5, missing_sent, missing_sent_len, keywords, keywords_pos```

- Out of 5 sentences, one sentence is selected randomly among [sent2, sent3, sent4] and marked as “\<MISSING\>”. This sentence is placed under <strong>"missing_sent”</strong> column. 
	- The first and last sentences were excluded to give it pre/post context. 
- <strong>"missing_sent_len"</strong> is the # of tokens of the missing sentence, including punctuations. 
- <strong>keywords</strong> contains the tokens of the missing sentence, excluding punctuations. (delimiter= “||”)
  - Punctuations are removed to put focus on words
 
- <stron>keywords_pos</strong> contrains the POS tags of the tokens listed in <strong>keywords</strong> file. 


3. Example data entry

Below is one example instance:

```
storyid	      16735298-ee95-42a2-9c08-87c066f89d61
title	        Better Socks
sent1	        I always used to buy the cheapest socks available.
sent2	        I figured it was inconsequential.
sent3	        <MISSING>
sent4	        I upgraded to softer and more expensive sock varieties.
sent5	        Now I can never go back.
missing_sent_len	10
missing_sent	But I started walking to work and getting blisters.
keywords	blisters||started||work||I||getting||and||But||to||walking
keywords_pos	NOUN||VERB||NOUN||PRON||VERB||CCONJ||CCONJ||ADP||VERB
```



4. Dataset size

Below is the count of each type:

train:  36,865/ 52,665(70%) 
valid:   5,266/ 52,665(10%) 
test:  10,533/ 52,665(20%) 

