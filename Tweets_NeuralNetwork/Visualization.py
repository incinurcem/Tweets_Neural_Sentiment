from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from Tweets_Exploratory_Data_Analysis import df
from Neural_Network_Analysis import history




#Generating Word Cloud

def wordcount_gen(df, category):
    # Combining of all tweets
    combined_tweets = " ".join(
        [tweet for tweet in df[df.Sentiment == category]['tweet']])

    # Preparing of word cloud background
    wc = WordCloud(background_color='white',
                   max_words=50,
                   stopwords=STOPWORDS)

    plt.figure(figsize=(10, 10))
    plt.imshow(wc.generate(combined_tweets))
    plt.title('{} Sentiment Words'.format(category), fontsize=20)
    plt.axis('off')
    plt.show()

wordcount_gen(df, 'neu')




#Creating of history plot

def plot_training_hist(history):
    '''Function to plot history for accuracy and loss'''

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    # first plot
    ax[0].plot(history.history['accuracy'])
    ax[0].plot(history.history['val_accuracy'])
    ax[0].set_title('Model Accuracy')
    ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('accuracy')
    ax[0].legend(['train', 'validation'], loc='best')
    # second plot
    ax[1].plot(history.history['loss'])
    ax[1].plot(history.history['val_loss'])
    ax[1].set_title('Model Loss')
    ax[1].set_xlabel('epoch')
    ax[1].set_ylabel('loss')
    ax[1].legend(['train', 'validation'], loc='best')


plot_training_hist(history)
plt.show()