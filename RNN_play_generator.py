import tensorflow as tf
import keras
from keras.preprocessing import sequence
import os
import numpy as np

path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# from google.colab import files
# path_to_file2 = list(files.upload().keys())[0]

#Read, then decode for py2 compay. 
text = open(path_to_file, 'rb').read().decode(encoding='utf-8') #opening file, 'rb' means read bytes mode. read and decode into 'utf-8' format
#length of text is the number of characters in it
#print('Length of text: {} characters '.format(len(text))) #length of the text is 1115394

#look at the first 250 characters in text
#print(text[:250])

#-----Encoding-----#
#This code is not yet encoded. We need to do this ourselves -- each unique character needs to be encoded as a different integer

vocab = sorted(set(text))
# Creating a mapping from unique charactesr to indices
char2idx = {u: i for i, u in enumerate(vocab)} #must research this line
idx2char = np.array(vocab)

def text_to_int(text):
    return np.array([char2idx[c] for c in text])

text_as_int = text_to_int(text)

#lets look at how part of our text is encoded
print("Text:", text[:13])
print("Encoded:", text_to_int(text[:13]))

#Function to convert numeric values to text -- for convenience/future purposes

def int_to_text(ints):
    try:
        ints = ints.numpy()
    except:
        pass
    return ''.join(idx2char[ints])

#print(int_to_text(text_as_int[:13]))

#-----Create Training Examples-----#
"""
Remember, task is to feed the model a sequence and have it return to us the next character. This means we need to split our data from above into many shorter sequences that we can pass to
the model as training examples. Training examples  that we prepare will use seq_length sequence as input and a seq_length sequence as the output where 
that sequence is the original sequence shift one letter to the right. 

Example: 
input: Hell | output: ello

"""
#first step: create a stream of characters from our text data 
seq_length = 100# length of sequence for a training example

examples_per_epoch = len(text)//(seq_length+1) # for every training example, we need to create a sequence input and output that is 100 characters long. This means we need to have a sequence
# of 101 characters for every training example

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int) #create training examples / targets -- convert entire string dataset into characters/ allows us to have a stream of characters

#Use batch method to turn this stream of characters into batches of desired length
sequences = char_dataset.batch(seq_length+1, drop_remainder=True) # drop remainder means it will drop the remainders if the length of the text is greater than seq_length (100)

#now uses these sequences of 101 length and split them into input and output -- this will create training examples!

def split_input_target(chunk): # example: Hello
        input_text = chunk[:-1] # hell
        target_text = chunk[1:] # ello
        return input_text, target_text # hell, ello

dataset = sequences.map(split_input_target) # we may use map to apply the above function to every entry. Every word in sequences will be split using the function above and loaded into dataset

#We will now create training batches

BATCH_SIZE = 64
VOCAB_SIZE = len(vocab) # vocab is number of unique characters
EMBEDDING_DIMENSION = 256 # how big we want every vector to represent our words are in the embedding layer
RNN_UNITS = 1024 # 

# Buffer Size to shuffle the dataset
# TF data is designed to work with possibly infinite sequences
#  so that it doesnt attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in whcih it shuffles elements.

BUFFER_SIZE = 10000

#create a shuffled dataset with a batched size
data=dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

#----Building the Model-----#
#We will later save this model and add different batches later. For now, we will feed batch sizes of 64, but later we will feed batch sizes of 1. Creating a function will help this process.
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
     model = tf.keras.Sequential([
          tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape =[batch_size, None]), #None means we do not know how long the sequences be will be in each batch. 
          tf.keras.layers.LSTM(rnn_units, #LSTM = Long short term memory
                               return_sequences=True, #Return the intermediate stage at eveyr step. We want to see the model at every intermediate stop, not just the final step.
                               stateful=True, # need further research
                               recurrent_initializer='glorot_uniform'), #Good default, recurrent initializer is how the values will start at. Need more research. 
                               tf.keras.layers.Dense(vocab_size) #contain the amount of vocab size nodes. we want the final layer to have the amount of nodes in it equal to the amount of characters in the vocabulary.
     ])                                                             
     return model

model = build_model(VOCAB_SIZE, EMBEDDING_DIMENSION, RNN_UNITS, BATCH_SIZE)
#model.summary()

#-----Create a loss function-----#
"""This model will output a (64, sequence_length, 65) shaped tensor that represents the probability distribution of each chacater at each timestep for every sequence in the batch.
Before we continue with this, we will look at a simple input and the output of our untrained model. This will help us understand what the model is actually giving us"""



for input_example_batch, target_example_batch in data.take(1):
     example_batch_predictions = model(input_example_batch) # ask our model for a prediction on our first batch of training data
     #print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)") # print out the output shape

# We can see that the prediction is an array of 64 arrays, one for each entry in the batch
# print(len(example_batch_predictions))
# print(example_batch_predictions)

# Lets examine one prediction
pred = example_batch_predictions[0]
# print(len(pred))
# print(pred)

# lets look at a prediction at the first timestep
time_pred = pred[0]
# print(len(time_pred))
# print(time_pred)
# its 65 values representing the probability of each character occuring next

#if we want to determine the predicted character we need to sample the output distribution (pick a value based on probabilities)
sampled_indices = tf.random.categorical(pred, num_samples=1)

#now we can rehspae that array and convert all the integers to numbers to see the actual characters
sampled_indices = np.reshape(sampled_indices, (1, -1))[0] 
predicated_chars = int_to_text(sampled_indices) 
#print(predicated_chars) #This will print out the current characters that the model will output without any training or augmentation

#Loss Function: 
def loss(labels, logits):
     return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

#-----Compiling Model-----#

model.compile(optimizer='adam', loss=loss)


#-----Creating Checkpoints-----#

#We need to setup and configure our model to save checkpoints as it trains. This will allow us to load a model from a checkpoint and continue training it.

# Directory where checkpoints will be saved

checkpoint_dir = './training_checkpoints' # must investigate how to use this further

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
     filepath=checkpoint_prefix,
     save_weights_only = True)

#-----Training the model-----#

history = model.fit(data, epochs=40, callbacks=[checkpoint_callback]) #Do not run this on a slow machine, training takes excessively long

#-----Loading the model-----#

# rebuild the model from a checkpoint using a batch_size of 1 so that we can feed one piece of text to the model and have it make a prediction

model = build_model(VOCAB_SIZE, EMBEDDING_DIMENSION, RNN_UNITS, batch_size=1)

# Once the model is finished training, we can find the latest checkpoint that stores the model's weights using the following line

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

# We can load any checkpointwe want by specifying the exact file to load

checkpoint_num = 10
model.load_weights(tf.train.load_checkpoint("./training_checkpoints/ckpt_" + str(checkpoint_num)))
model.build(tf.TensorShape([1, None])) # 1 means expect the input '1', None means we do not know what the next dimension will be

# Use the function below to to generate text using any starting string we would like

def generate_text(model, start_string):
     # Evaluation step (generating text using the learned model)

     # number of chacaters to generate
     num_generate = 800

     # converting our start string to numbers (vectorizing)
     input_eval = [char2idx[s] for s in start_string]
     input_eval = tf.expand_dims(input_eval, 0)

     text_generated= []

     # Low temperatures resilts in more predictable text.
     # Higher temperatures results in more surprising texts.
     # Experiment to find the best setting.
     temperature = 1.0

     # Here, batch_size == 1
     model.rest_states()
     for i in range(num_generate):
          predictions= model(input_eval)
          # remove the batch dimension
          predictions= tf.squeeze(predictions, 0)

          #using a categorical distribution to predict the character returned by the model
          predictions = predictions / temperature
          predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

          # We pass the predicted character as the next input to the model along with the precious hidden state

          input_eval = tf.expand_dims([predicted_id], 0)

          text_generated.append(idx2char[predicted_id])
     return (start_string + ''.join(text_generated))

inp = input("Type a starting string: ")
print(generate_text(model, inp))