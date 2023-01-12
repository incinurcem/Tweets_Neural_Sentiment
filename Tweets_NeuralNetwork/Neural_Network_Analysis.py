
from keras.optimizers import SGD
from keras.metrics import Precision, Recall
from keras.layers import Embedding, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from sklearn.model_selection import train_test_split
import keras.backend as K
from Tweets_Exploratory_Data_Analysis import target, features
import tensorflow as tf




#Splitting the datas

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)



#Representation of train, validation and test sets

print('Train Set: ', X_train.shape, y_train.shape)
print('Validation Set: ', X_val.shape, y_val.shape)
print('Test Set : ', X_test.shape, y_test.shape)



#Function of calculating f1 score

def f1_score(precision, recall):
    ''' Function to calculate f1 score '''
    
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val



#Preparing to build model

vocab_size = 5000
embedding_size = 32
epochs = 5
learning_rate = 0.1
decay_rate = learning_rate / epochs
momentum = 0.8

sgd = SGD(lr=learning_rate, momentum=momentum,
          decay=decay_rate, nesterov=False)




#Building the model

model = Sequential()
model.add(Embedding(vocab_size, embedding_size, input_length=1))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=1))
model.add(Bidirectional(LSTM(32)))
model.add(Dropout(0.4))
model.add(Dense(3, activation='softmax'))

tf.keras.utils.plot_model(model, show_shapes=True)
print(model.summary())




#Compiling the model

model.compile(loss='categorical_crossentropy', optimizer=sgd, 
               metrics=['accuracy', Precision(), Recall()])




#Training the model

batch_size = 64
history = model.fit(X_train, y_train,
                      validation_data=(X_val, y_val),
                      batch_size=batch_size, epochs=epochs, verbose=1)
