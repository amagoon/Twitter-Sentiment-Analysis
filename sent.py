import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from gensim import corpora, models, similarities
f = open("50KTweets.csv", "r")
tweets = []
for line in f:
    tweets.append(line.decode('utf8').replace('\n','').replace('#',''))
print "%s tweets imported." % len(tweets)
#Load stoplist
stoplist = []

stopFile = open("englishsw.txt", 'r')
for line in stopFile:
    stoplist.append(line.replace('\n',''))
stoplist.append("bieber")

print stoplist
#Remove words in stoplist and transform each tweet to a list of words
ftweets = [[word for word in tweet.lower().split() if word not in stoplist] for tweet in tweets]
print ftweets[0:2]
#Remove words that appear only once
all_tokens = sum(ftweets, [])
tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
ftweets = [[word for word in text if word not in tokens_once] for text in ftweets]
#Create a dictionary of all words
dictionary = corpora.Dictionary(ftweets)
dictionary.save('dictionary.dict')
print(dictionary.token2id)
#Create our bag of words representation (vector)
corpus = [dictionary.doc2bow(text) for text in ftweets]
corpora.MmCorpus.serialize('corpus.mm', corpus)
print(corpus)
#Create LDA model
model = models.ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=100, passes=10)
model.print_topics(1)
from textblob import TextBlob

sentiments = []
polarities = []
subjectivities = []
positive = 0
negative = 0
neutral = 0
for tweet in tweets:
    s = TextBlob(tweet)
    output = "%s %s" % s.sentiment,tweet
    sentiments.append(output)
    polarity = s.sentiment.polarity
    if polarity >= 0.1:
        positive += 1
    elif polarity <= -0.1:
        negative += 1
    else:
        neutral += 1
    polarities.append(polarity)
    subjectivities.append(s.sentiment.subjectivity)
        
print "Positive tweets: %s" % positive
print "Neutral tweets: %s" % neutral
print "Negative tweets: %s" % negative
    
for i in range(0,4):
    print sentiments[i]
    
sentimentFile = open('sentiment_output2.txt', 'w')

for sentiment in sentiments:
    sentimentFile.write(str(sentiment).encode('utf-8'))
    sentimentFile.write('\n')
    
sentimentFile.close()