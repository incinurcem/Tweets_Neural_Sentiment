
from sklearn.preprocessing import LabelEncoder
import nest_asyncio 
nest_asyncio.apply()
import twint
import pandas as pd
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer


#Fetching Tweets

c = twint.Config()
c.Search='today'
c.Lang='en'
c.Pandas= True 
c.Limit= 100 
twint.run.Search(c)
tlist = c.search_tweet_list  
df = twint.storage.panda.Tweets_df



#Editing of the dataframe

df.drop(['conversation_id','created_at','date','time','timezone','user_id','name','place','language','mentions','urls','photos','replies_count','retweets_count','likes_count','hashtags','cashtags','link','retweet','quote_url','video','thumbnail','near','geo','source','user_rt_id','user_rt','retweet_id','reply_to','retweet_date','translate','trans_src','trans_dest'],axis=1,inplace=True)
df = pd.DataFrame(df, columns=('id', 'username', 'tweet', 'Sentiment'))
df.to_csv('/Users/cem/Desktop/tweets6.csv', encoding='utf-8', index=False, header=True)



#Creating a column corresponding to sentiment type

vds= SentimentIntensityAnalyzer()
df['compound'] = [vds.polarity_scores(x)['compound'] for x in df['tweet']]
df['neg'] = [vds.polarity_scores(x)['neg'] for x in df['tweet']]
df['neu'] = [vds.polarity_scores(x)['neu'] for x in df['tweet']]
df['pos'] = [vds.polarity_scores(x)['pos'] for x in df['tweet']]

st=[]
st = df[['neg','neu','pos']].idxmax(axis=1)
df['Sentiment'] = st
df.to_csv('/Users/cem/Desktop/tweets.csv', sep='\t')
print(' numbers of positive sentiment: ', len(df[df['Sentiment'] == 'pos']), '\n', 'numbers of neutral sentiment: ', len(df[df['Sentiment'] == 'neu']), '\n', 'numbers of negative sentiment: ', len(df[df['Sentiment'] == 'neg']))



#Preparing of feature and target dataframes

features = []
features = pd.DataFrame(df, columns=["tweet"]).astype(str)


target = []
target = pd.DataFrame(df, columns=['Sentiment']).astype(str)

le = LabelEncoder()
target = le.fit_transform(target)
target = pd.get_dummies(target)
features = le.fit_transform(features)
features = pd.get_dummies(features)





