# CollaboSentGen

1. Overview
This dataset is for the hackathon at <em>2018 TTIC Workshop: Collaborative & Knowledge-backed Language Generation</em>. 
It is divided into train/valid/test(70%/10%/20%) randomly,

2. Dataset Details

Each file is a csv file with following columns.

```storyid, title, sent1, sent2, sent3, sent4, sent5, missing_sent_len, missing_sent, accepted_words, keywords```

- Out of 5 sentences, one sentence is selected randomly among [sent2, sent3, sent4] and marked as “\<MISSING\>”. This sentence is placed under <strong>"missing_sent”</strong> column. 
	- The first and last sentences were excluded to give it pre/post context. 
- <strong>"missing_sent_len"</strong> is the # of tokens of the missing sentence, including punctuations. 
- <strong>"accepted_words"</strong> is a list of (word:position), delimited by "||". ex) understand:8||working:3
	- position starts from 0
- <strong>"keywords"</strong> contains the tokens of the missing sentence, excluding punctuations. (delimiter= “||”)
	- Punctuations are removed to put focus on words
 


3. Example data entry

Below is one example instance:

```
storyid:	fa0b6f65-fdfc-4df7-a5c4-fcaf463097db
title:		New Flavors
sent1:		Sam was extremely picky.
sent2:		He would only eat pizza and fries.
sent3:		His mother would try to get him to eat other things but he refused.
sent4:		<MISSING>
sent5:		Sam promised to try all kinds of new things.
missing_sent_len: 15
missing_sent:	Until one day she made a lasagna and Sam tried it and liked it.
accepted_words:	it:10||and:11||day:2||sam:8
keywords:	tried||one||and
```


4. Data Sources
* ROCStories from: http://cs.rochester.edu/nlp/rocstories/


5. Dataset size
* train:  26,330
* valid:   5,263
* test:   10,529
* test2:  10,543 (held out data, not publically available)
