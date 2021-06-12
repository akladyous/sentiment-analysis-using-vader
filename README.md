### Sentiment Analysis using VADER

<img src="https://raw.githubusercontent.com/akladyous/sentiment-analysis-using-vader/main/img/main.png" style="zoom:50%;" />

#### Introduction

Sentiment Analysis is a supervised Machine Learning technique used to determine if a chunk of text is positive, negative or neutral. In text analytics, natural language processing (NLP) and machine learning (ML) techniques are combined to assign sentiment scores to the topics, categories or entities within a phrase.

Sentiment analysis is a powerful tool that businesses can leverage to better understand the overall opinions of their customer, gain insights, and make data-driven decisions.

In this tutorial we will classify text articles using VADER on a dataset from CrowdFlower.

#### Import Libraries

```python
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import sent_tokenize, RegexpTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('punkt')
```

#### Data Collection

for the sake of this tutorial we're going to use a simple dataset from CrowdFlower via data.world (IMDB Sentiment Sampled). for more info visit the  link:

https://data.world/robbertb/imdb-sentiment-sampled

```python
df = pd.read_csv('imdb_sentiment.csv')
df.head()
```

```
	review
0	Protocol is an implausible movie whose only sa...
1	I just watched The Dresser this evening, havin...
2	Besides being boring, the scenes were oppressi...
3	I'm not sure why the producers needed to trade...
4	Honestly - this short film sucks. the dummy us...
```

#### Preprocessing

The NLTK module is a massive tool kit, aimed at helping you with the entire Natural Language Processing (NLP) methodology. NLTK will aid you with everything from splitting sentences from paragraphs, splitting up words, recognizing the part of speech of those words.

In order to apply polarity score we need to prepare your data, we start by the converting our data to lowcase using the built-in function "lower" 

```python
df['review'] = df['review'].apply(lambda txt: txt.lower())
```

Removing stop word using NLTK  by downloading the english stopwords

```python
stop_words=stopwords.words('english')
df['review'] = df['review'].apply(lambda txt: ' '.join([word for word in txt.split() if word not in stop_words]))
```

Sentences Tokenization

```python
df['review'] = df['review'].apply(lambda txt: sent_tokenize(txt))
```

Join the tokenized data into text

```python
df['review'] = df['review'].apply(lambda txt: ' '.join(txt))
```

#### VADER

VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media. 

It is used for sentiment analysis of text which has both the polarities i.e. positive/negative. VADER is used to quantify how much of positive or negative emotion the text has and also the intensity of emotion.

#### Polarity classification

The VADER library returns 4 values such as:

- **pos**: The probability of the sentiment to be **positive**
- **neu**: The probability of the sentiment to be **neutral**
- **neg**: The probability of the sentiment to be **negative**
- **compound**: The normalized compound score which calculates the sum of all lexicon ratings and takes values from -1 to 1

Notice that the `pos`, `neu` and `neg` probabilities add up to **1**. Also, the `compound` score is a very useful metric in case we want a single measure of sentiment. Typical threshold values are the following:

- **positive**: compound **score>=0.05**
- **neutral**: compound score between **-0.05 and 0.05**
- **negative**: compound **score<=-0.05**



Instantiate a new object with NLTK SentimentIntensityAnalyzer

```python
sid = SentimentIntensityAnalyzer()
```

Now we create a new column to the original DataFrame to store the polarity_scores dictionary, the scores extracted will have the keys "neg", "neu", "pos", "compound"  derived from the composite score.

```python
df['score'] = df['review'].apply(lambda txt: sid.polarity_scores(txt))
```

```python
df['score'][0]
```

> {'neg': 0.0, 'neu': 0.653, 'pos': 0.347, 'compound': 0.9565}

<img src="https://raw.githubusercontent.com/akladyous/sentiment-analysis-using-vader/main/img/score.png" style="zoom:50%;" />

```python
df['negative'] = df['score'].apply(lambda txt: txt['neg'])
```

```python
df['neutral'] = df['score'].apply(lambda txt: txt['neu'])
```

```python
df['positive'] = df['score'].apply(lambda txt: txt['pos'])
```

```python
df['compound'] = df['score'].apply(lambda txt: txt['compound'])
```

<img src="https://raw.githubusercontent.com/akladyous/sentiment-analysis-using-vader/main/img/compound.png" style="zoom:50%;" />

We create a function called "polarity_score" to calculate the accuracy test for each review in our dataframe. finally we apply the function by  creating a new column called "sentiment" . the reviews in this column will be classified into positive, negative and neutral.

```python
def polarity_score(compound):
    if compound > 0.05:
        return "positive"
    elif compound < -0.5:
        return "negative"
    elif compound >= -0.05 and compound < 0.05:
        return "neutral"
```

```python
df['sentiment'] = df['compound'].apply(lambda val: polarity_score(val))
df.head()
```

<img src="https://raw.githubusercontent.com/akladyous/sentiment-analysis-using-vader/main/img/sentiment.png" style="zoom:50%;" />

```python
df['sentiment'].value_counts()
```

> ```
> positive    694
> negative    258
> neutral       6
> Name: sentiment, dtype: int64
> ```



#### Conclusion

VADER classifies the sentiments very well. It is easy to use, the ready-made model which can be used across multiple domains, social-media texts.



Thanks for reading and happy web scraping everyone!

You can find my Jupyter Notebook for this on my [github](https://github.com/akladyous/sentiment-analysis-using-vader).