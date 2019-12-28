#!/usr/bin/env python
# coding: utf-8

# # Neural Machine Translation
# 
# Welcome to your first programming assignment for this week! 
# 
# You will build a Neural Machine Translation (NMT) model to translate human readable dates ("25th of June, 2009") into machine readable dates ("2009-06-25"). You will do this using an attention model, one of the most sophisticated sequence to sequence models. 
# 
# This notebook was produced together with NVIDIA's Deep Learning Institute. 
# 
# Let's load all the packages you will need for this assignment.

# In[1]:


from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, CuDNNLSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation,Lambda,Softmax,Reshape

from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
import tensorflow as tf
import numpy as np

from faker import Faker
import random
from tqdm import tqdm
from babel.dates import format_date
from nmt_utils import *
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ##  - Translating human readable dates into machine readable dates
# 
# The model you will build here could be used to translate from one language to another, such as translating from English to Hindi. However, language translation requires massive datasets and usually takes days of training on GPUs. To give you a place to experiment with these models even without using massive datasets, we will instead use a simpler "date translation" task. 
# 
# The network will input a date written in a variety of possible formats (*e.g. "the 29th of August 1958", "03/30/1968", "24 JUNE 1987"*) and translate them into standardized, machine readable dates (*e.g. "1958-08-29", "1968-03-30", "1987-06-24"*). We will have the network learn to output dates in the common machine-readable format YYYY-MM-DD. 
# 
# 
# 
# <!-- 
# Take a look at [nmt_utils.py](./nmt_utils.py) to see all the formatting. Count and figure out how the formats work, you will need this knowledge later. !--> 

# ###  - Dataset
# 
# We will train the model on a dataset of 10000 human readable dates and their equivalent, standardized, machine readable dates. Let's run the following cells to load the dataset and print some examples. 

# In[2]:


m = 10000
dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)


# In[3]:


dataset[:10]


# You've loaded:
# - `dataset`: a list of tuples of (human readable date, machine readable date)
# - `human_vocab`: a python dictionary mapping all characters used in the human readable dates to an integer-valued index 
# - `machine_vocab`: a python dictionary mapping all characters used in machine readable dates to an integer-valued index. These indices are not necessarily consistent with `human_vocab`. 
# - `inv_machine_vocab`: the inverse dictionary of `machine_vocab`, mapping from indices back to characters. 
# 
# Let's preprocess the data and map the raw text data into the index values. We will also use Tx=30 (which we assume is the maximum length of the human readable date; if we get a longer input, we would have to truncate it) and Ty=10 (since "YYYY-MM-DD" is 10 characters long). 

# In[4]:


#human_vocab
#machine_vocab


# In[5]:


Tx = 30
Ty = 10
X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)

print("X.shape:", X.shape)
print("Y.shape:", Y.shape)
print("Xoh.shape:", Xoh.shape)
print("Yoh.shape:", Yoh.shape)


# In[6]:


X[0] #'<pad>': 36


# You now have:
# - `X`: a processed version of the human readable dates in the training set, where each character is replaced by an index mapped to the character via `human_vocab`. Each date is further padded to $T_x$ values with a special character (< pad >). `X.shape = (m, Tx)`
# - `Y`: a processed version of the machine readable dates in the training set, where each character is replaced by the index it is mapped to in `machine_vocab`. You should have `Y.shape = (m, Ty)`. 
# - `Xoh`: one-hot version of `X`, the "1" entry's index is mapped to the character thanks to `human_vocab`. `Xoh.shape = (m, Tx, len(human_vocab))`
# - `Yoh`: one-hot version of `Y`, the "1" entry's index is mapped to the character thanks to `machine_vocab`. `Yoh.shape = (m, Tx, len(machine_vocab))`. Here, `len(machine_vocab) = 11` since there are 11 characters ('-' as well as 0-9). 
# 

# Lets also look at some examples of preprocessed training examples. Feel free to play with `index` in the cell below to navigate the dataset and see how source/target dates are preprocessed. 

# In[7]:


index = 0
print("Source date:", dataset[index][0])
print("Target date:", dataset[index][1])
print()
print("Source after preprocessing (indices):", X[index][0])
print("Target after preprocessing (indices):", Y[index][0])
print()
print("Source after preprocessing (one-hot):", Xoh[index][0])
print("Target after preprocessing (one-hot):", Yoh[index][0])


# ##  Neural machine translation with attention
# 
# If you had to translate a book's paragraph from French to English, you would not read the whole paragraph, then close the book and translate. Even during the translation process, you would read/re-read and focus on the parts of the French paragraph corresponding to the parts of the English you are writing down. 
# 
# The attention mechanism tells a Neural Machine Translation model where it should pay attention to at any step. 
# 
# 
# ###  Attention mechanism
# 
# In this part, you will implement the attention mechanism presented in the lecture videos. Here is a figure to remind you how the model works. The diagram on the left shows the attention model. The diagram on the right shows what one "Attention" step does to calculate the attention variables $\alpha^{\langle t, t' \rangle}$, which are used to compute the context variable $context^{\langle t \rangle}$ for each timestep in the output ($t=1, \ldots, T_y$). 
# 
# <table>
# <td> 
# <img src="images/attn_model.png" style="width:500;height:500px;"> <br>
# </td> 
# <td> 
# <img src="images/attn_mechanism.png" style="width:500;height:500px;"> <br>
# </td> 
# </table>
# <caption><center> **Figure 1**: Neural machine translation with attention</center></caption>
# 

# 
# Here are some properties of the model that you may notice: 
# 
# - There are two separate LSTMs in this model (see diagram on the left). Because the one at the bottom of the picture is a Bi-directional LSTM and comes *before* the attention mechanism, we will call it *pre-attention* Bi-LSTM. The LSTM at the top of the diagram comes *after* the attention mechanism, so we will call it the *post-attention* LSTM. The pre-attention Bi-LSTM goes through $T_x$ time steps; the post-attention LSTM goes through $T_y$ time steps. 
# 
# - The post-attention LSTM passes $s^{\langle t \rangle}, c^{\langle t \rangle}$ from one time step to the next. In the lecture videos, we were using only a basic RNN for the post-activation sequence model, so the state captured by the RNN output activations $s^{\langle t\rangle}$. But since we are using an LSTM here, the LSTM has both the output activation $s^{\langle t\rangle}$ and the hidden cell state $c^{\langle t\rangle}$. However, unlike previous text generation examples (such as Dinosaurus in week 1), in this model the post-activation LSTM at time $t$ does will not take the specific generated $y^{\langle t-1 \rangle}$ as input; it only takes $s^{\langle t\rangle}$ and $c^{\langle t\rangle}$ as input. We have designed the model this way, because (unlike language generation where adjacent characters are highly correlated) there isn't as strong a dependency between the previous character and the next character in a YYYY-MM-DD date. 
# 
# - We use $a^{\langle t \rangle} = [\overrightarrow{a}^{\langle t \rangle}; \overleftarrow{a}^{\langle t \rangle}]$ to represent the concatenation of the activations of both the forward-direction and backward-directions of the pre-attention Bi-LSTM. 
# 
# - The diagram on the right uses a `RepeatVector` node to copy $s^{\langle t-1 \rangle}$'s value $T_x$ times, and then `Concatenation` to concatenate $s^{\langle t-1 \rangle}$ and $a^{\langle t \rangle}$ to compute $e^{\langle t, t'}$, which is then passed through a softmax to compute $\alpha^{\langle t, t' \rangle}$. We'll explain how to use `RepeatVector` and `Concatenation` in Keras below. 
# 
# Lets implement this model. You will start by implementing two functions: `one_step_attention()` and `model()`.
# 
# **1) `one_step_attention()`**: At step $t$, given all the hidden states of the Bi-LSTM ($[a^{<1>},a^{<2>}, ..., a^{<T_x>}]$) and the previous hidden state of the second LSTM ($s^{<t-1>}$), `one_step_attention()` will compute the attention weights ($[\alpha^{<t,1>},\alpha^{<t,2>}, ..., \alpha^{<t,T_x>}]$) and output the context vector (see Figure  1 (right) for details):
# $$context^{<t>} = \sum_{t' = 0}^{T_x} \alpha^{<t,t'>}a^{<t'>}\tag{1}$$ 
# 
# Note that we are denoting the attention in this notebook $context^{\langle t \rangle}$. In the lecture videos, the context was denoted $c^{\langle t \rangle}$, but here we are calling it $context^{\langle t \rangle}$ to avoid confusion with the (post-attention) LSTM's internal memory cell variable, which is sometimes also denoted $c^{\langle t \rangle}$. 
#   
# **2) `model()`**: Implements the entire model. It first runs the input through a Bi-LSTM to get back $[a^{<1>},a^{<2>}, ..., a^{<T_x>}]$. Then, it calls `one_step_attention()` $T_y$ times (`for` loop). At each iteration of this loop, it gives the computed context vector $c^{<t>}$ to the second LSTM, and runs the output of the LSTM through a dense layer with softmax activation to generate a prediction $\hat{y}^{<t>}$. 
# 
# 
# 
# **Exercise**: Implement `one_step_attention()`. The function `model()` will call the layers in `one_step_attention()` $T_y$ using a for-loop, and it is important that all $T_y$ copies have the same weights. I.e., it should not re-initiaiize the weights every time. In other words, all $T_y$ steps should have shared weights. Here's how you can implement layers with shareable weights in Keras:
# 1. Define the layer objects (as global variables for examples).
# 2. Call these objects when propagating the input.
# 
# We have defined the layers you need as global variables. Please run the following cells to create them. Please check the Keras documentation to make sure you understand what these layers are: [RepeatVector()](https://keras.io/layers/core/#repeatvector), [Concatenate()](https://keras.io/layers/merge/#concatenate), [Dense()](https://keras.io/layers/core/#dense), [Activation()](https://keras.io/layers/core/#activation), [Dot()](https://keras.io/layers/merge/#dot).

# Now you can use these layers to implement `one_step_attention()`. In order to propagate a Keras tensor object X through one of these layers, use `layer(X)` (or `layer([X,Y])` if it requires multiple inputs.), e.g. `densor(X)` will propagate X through the `Dense(1)` layer defined above.

# In[8]:


# M1: 使用 全局的 层对象，以在多个 model 中共享他们的权重

# GRADED FUNCTION: one_step_attention
# Defined shared layers as global variables
repeator = RepeatVector(Tx)  
#concatenator = Concatenate(axis=-1)
concatenator = Concatenate(axis=2)
densor = Dense(1, activation = "relu")
#activator = Activation(softmax, name='attention_weights') # We are using a custom softmax(axis = 1) loaded in this notebook
activator=Softmax(axis=1)
dotor = Dot(axes = 1)


def one_step_attention(a, s_prev): #与RNN 类似，是一个 循环结构
    """
    Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
    "alphas" and the hidden states "a" of the Bi-LSTM.
    
    Arguments:
    a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
    s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)
    
    Returns:
    context -- context vector, input of the next (post-attetion) LSTM cell
    """
    
    ### START CODE HERE ###
    # Use repeator to repeat s_prev to be of shape (m, Tx, n_s) so that you can concatenate it with all hidden states "a" (≈ 1 line)
    s_prev = repeator(s_prev) 
    
    # Use concatenator to concatenate a and s_prev on the last axis (≈ 1 line) 
    concat =concatenator([a,s_prev]) #shape: (m, Tx, 2*n_a+n_s)
                                      
    # Use densor to propagate concat through a small fully-connected neural network to compute the "energies" variable e. (≈1 lines)
    e = densor(concat) #  shape: (m, Tx, 1)
                    
    # Use activator and e to compute the attention weights "alphas" (≈ 1 line)
    alphas = activator(e) # shape:  (m, Tx, 1)
    
    # Use dotor together with "alphas" and "a" to compute the context vector to be given to the next (post-attention) LSTM-cell (≈ 1 line)
    context = dotor([alphas,a]) #  shape: (m, 1, 2*n_a)
    
    ### END CODE HERE ###
    
    return context


# In[23]:


#M2: 把层 layer 包装为 model ,并通过重新定义 model 的输入的方式 来共享 layer 的权重 

n_a = 64
n_s = 128 


def one_step_attention_model(Tx, n_a, n_s): 

    """ 
    Arguments:
    a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
    s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)
    
    Returns:
    context -- context vector, input of the next (post-attetion) LSTM cell
    """
    
    repeator = RepeatVector(Tx)  
    concatenator = Concatenate(axis=2)
    densor = Dense(1, activation = "relu")
    activator=Softmax(axis=1)
    dotor = Dot(axes = 1)
    
    
    a0=Input(shape=(Tx, 2*n_a), name='a')
    s_prev0=Input(shape=(n_s,), name='s_prev')
    
    a=a0 # 否则报错 ： ValueError: Graph disconnected: cannot obtain value for tensor Tensor .... The following previous layers were accessed without issue: []
    s_prev=s_prev0
    
    # Use repeator to repeat s_prev to be of shape (m, Tx, n_s) so that you can concatenate it with all hidden states "a" (≈ 1 line)
    s_prev = repeator(s_prev) 
    
    # Use concatenator to concatenate a and s_prev on the last axis (≈ 1 line) 
    concat =concatenator([a,s_prev]) #shape: (m, Tx, 2*n_a+n_s)
                                      
    # Use densor to propagate concat through a small fully-connected neural network to compute the "energies" variable e. (≈1 lines)
    e = densor(concat) #  shape: (m, Tx, 1)
                    
    # Use activator and e to compute the attention weights "alphas" (≈ 1 line)
    alphas = activator(e) # shape:  (m, Tx, 1)
    
    # Use dotor together with "alphas" and "a" to compute the context vector to be given to the next (post-attention) LSTM-cell (≈ 1 line)
    context = dotor([alphas,a]) #  shape: (m, 1, 2*n_a)
    
    model=Model(inputs=[a0, s_prev0] ,outputs=context)
    
    return model

model_one_step_attention = one_step_attention_model(Tx, n_a, n_s)

def one_step_attention_M2(a, s_prev): 
        
    context=model_one_step_attention([a, s_prev])
    
    return context


# You will be able to check the expected output of `one_step_attention()` after you've coded the `model()` function.

# **Exercise**: Implement `model()` as explained in figure 2 and the text above. Again, we have defined global layers that will share weights to be used in `model()`.

# In[9]:


n_a = 64
n_s = 128

# n_s = 64

pre_activation_LSTM_cell=Bidirectional(CuDNNLSTM(n_a, return_sequences=True,return_state = True))

post_activation_LSTM_cell = CuDNNLSTM(n_s, return_state = True)
output_layer = Dense(len(machine_vocab), activation=softmax)


# Now you can use these layers $T_y$ times in a `for` loop to generate the outputs, and their parameters will not be reinitialized. You will have to carry out the following steps: 
# 
# 1. Propagate the input into a [Bidirectional](https://keras.io/layers/wrappers/#bidirectional) [LSTM](https://keras.io/layers/recurrent/#lstm)
# 2. Iterate for $t = 0, \dots, T_y-1$: 
#     1. Call `one_step_attention()` on $[\alpha^{<t,1>},\alpha^{<t,2>}, ..., \alpha^{<t,T_x>}]$ and $s^{<t-1>}$ to get the context vector $context^{<t>}$.
#     2. Give $context^{<t>}$ to the post-attention LSTM cell. Remember pass in the previous hidden-state $s^{\langle t-1\rangle}$ and cell-states $c^{\langle t-1\rangle}$ of this LSTM using `initial_state= [previous hidden state, previous cell state]`. Get back the new hidden state $s^{<t>}$ and the new cell state $c^{<t>}$.
#     3. Apply a softmax layer to $s^{<t>}$, get the output. 
#     4. Save the output by adding it to the list of outputs.
# 
# 3. Create your Keras model instance, it should have three inputs ("inputs", $s^{<0>}$ and $c^{<0>}$) and output the list of "outputs".

# In[10]:


# GRADED FUNCTION: model 
def model(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):
    """
    Arguments:
    Tx -- length of the input sequence
    Ty -- length of the output sequence
    n_a -- hidden state size of the Bi-LSTM
    n_s -- hidden state size of the post-attention LSTM
    human_vocab_size -- size of the python dictionary "human_vocab"
    machine_vocab_size -- size of the python dictionary "machine_vocab"

    Returns:
    model -- Keras model instance
    """
    
    # Define the inputs of your model with a shape (Tx,)
    # Define s0 and c0, initial hidden state for the decoder LSTM of shape (n_s,)
    
    X = Input(shape=(Tx, human_vocab_size)) # shape: (m,Tx,human_vocab_size)
    s0 = Input(shape=(n_s,), name='s0')  # shape of s:  (m, 64)
    c0 = Input(shape=(n_s,), name='c0')  # shape of c:  (m, 64)
    s = s0
    c = c0
    
    # Initialize empty list of outputs
    outputs = []
    
    ### START CODE HERE ###
    
    # Step 1: Define your pre-attention Bi-LSTM. Remember to use return_sequences=True. (≈ 1 line)
    a, forward_h, forward_c, backward_h, backward_c= pre_activation_LSTM_cell(inputs=X) #  shape of a : (m,Tx, 2*n_a)
    s = Concatenate()([forward_h, backward_h]) # shape of s:  (m, 64)
    c = Concatenate()([forward_c, backward_c])


    # Step 2: Iterate for Ty steps
    for t in range(Ty):
    
        # Step 2.A: Perform one step of the attention mechanism to get back the context vector at step t (≈ 1 line)
        # a.shape()
        context = one_step_attention(a, s) # shape of s:  (m, 64)
        
        # Step 2.B: Apply the post-attention LSTM cell to the "context" vector.
        # Don't forget to pass: initial_state = [hidden state, cell state] (≈ 1 line)
        
        s, _, c = post_activation_LSTM_cell(inputs=context,initial_state=[s, c])#输入只有一个时间步

        
        # Step 2.C: Apply Dense layer to the hidden state output of the post-attention LSTM (≈ 1 line)
        out = output_layer(s)
        
        # Step 2.D: Append "out" to the "outputs" list (≈ 1 line)
        outputs.append(out) # shape of out : ( m ,machine_vocab) 
    
    #
    # Step 3: Create model instance taking three inputs and returning the list of outputs. (≈ 1 line)
    model =  Model(inputs=[X, s0, c0], outputs=outputs)
    #shape of outs : ( Ty ,m ,machine_vocab) 
    
    ### END CODE HERE ###
    
    return model


# In[11]:


# testing1 
def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)].T

# y = np.array([1,2,3,4])
# y=np.argmax(y, axis = -1)
# convert_to_one_hot(y,len(machine_vocab))

# testing2
# y=np.array([1,2,3,4])
# y
# y.reshape(2,2)
# y

# testing3   
#array的尊卑关系： numpy -> backend tensor -> layer tensor
def tensor_test():
    
#0.
    x=np.zeros((1,len(machine_vocab)))
    print(x.shape)
    pred0=K.reshape(x, (1,len(machine_vocab)))
    print(pred0)
#     Tensor("Reshape_2:0", shape=(1, 11), dtype=float64)

#1.    
#     pred0=np.zeros((1,len(machine_vocab))) 
#     pred0=Reshape(target_shape=(1,len(machine_vocab)))(pred0)
#   ValueError: Layer reshape_42 was called with an input that isn't a symbolic tensor. Received type: <class 'numpy.ndarray'>. Full input: [array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])]. All inputs to the layer should be tensors.     

#2.   
#     pred0=K.zeros((1,len(machine_vocab))) 
#     pred0=Reshape(target_shape=(1,len(machine_vocab)))(pred0)
#     print(pred0)
#  Tensor("reshape_91/Reshape:0", shape=(1, 1, 11), dtype=float32)
     
    

def reshape_tensor(x, shape):
    return K.reshape(x, shape);

def argmax_tensor(x, axis):
    return K.argmax(x, axis);

def one_hot_tensor(x, num_classes):
    #by # https://fdalvi.github.io/blog/2018-04-07-keras-sequential-onehot/
    return K.one_hot(K.cast(x, 'uint8'), num_classes);

# testing4
def tensor_test1():
    pred=Input(shape=(len(machine_vocab),), name='pred')  
    print (pred)

#   以下两个 reshape 效果相同 都是输出了 layer tensor
#     x = Lambda(reshape_tensor, arguments={'shape': (1, len(machine_vocab))})(pred)
#     print (x)
#     pred=Reshape(target_shape=(1,len(machine_vocab)))(pred)
#     print (pred)

    pred=Lambda(argmax_tensor, arguments={'axis': -1 })(pred)
    print(pred)
    pred=Lambda(one_hot_tensor, arguments={'num_classes': len(machine_vocab) })(pred)  
    print(pred)



tensor_test()


# In[12]:


# by model2

def argmax_tensor(x, axis):
    return K.argmax(x, axis);

def argmin_tensor(x, axis):
    return K.argmin(x, axis);

lambda_argmin=Lambda(argmin_tensor, arguments={'axis': -1 },name='argmin_tensor')


def one_hot_tensor(x, num_classes):
    #by : https://fdalvi.github.io/blog/2018-04-07-keras-sequential-onehot/
    return K.one_hot(K.cast(x, 'uint8'), num_classes);

n_a = 64
n_s = 128 


pre_activation_LSTM_cell=Bidirectional(CuDNNLSTM(n_a, return_sequences=True,return_state = True),name='encoder_lstm')

concatenate_s=Concatenate(name='concatenate_s')
concatenate_c=Concatenate(name='concatenate_c')

concatenate_context=Concatenate()



post_activation_LSTM_cell = CuDNNLSTM(n_s, return_state = True,name='decoder_lstm') 
output_layer = Dense(len(machine_vocab), activation=softmax,name='decoder_output')


lambda_argmax=Lambda(argmax_tensor, arguments={'axis': -1 },name='argmax_tensor')
lambda_one_hot=Lambda(one_hot_tensor, arguments={'num_classes': len(machine_vocab) },name='one_hot_tensor')  

reshape=Reshape(target_shape=(1,len(machine_vocab)))


# In[26]:


#   By XRH in 2019.9.10
#   改进attention：解码时加入上一个时刻的输出单词（eg. 当前词是'-' 下一个词必须为数字）
#  (1)在decoder， 经过softmax 输出后 取最大的 那一个 machine_vocab 的one-hot 
#     向量 与 context 拼接后输入 post_activation_LSTM_cell ，
#  (2)无需更改 lstm 的输出维度 ，仍然保持 n_s=128
# （3）把所有的 layer object 声明为全局的，以便 后面重构 decoder 可以使用训练好的网络结构


def model2(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):
    """
    Arguments:
    Tx -- length of the input sequence
    Ty -- length of the output sequence
    n_a -- hidden state size of the Bi-LSTM
    n_s -- hidden state size of the post-attention LSTM
    human_vocab_size -- size of the python dictionary "human_vocab"
    machine_vocab_size -- size of the python dictionary "machine_vocab"

    Returns:
    model -- Keras model instance
    """
    
    
    X = Input(shape=(Tx, human_vocab_size)) # shape: (m,Tx,human_vocab_size)
    s0 = Input(shape=(n_s,), name='s0')  # shape of s:  (m, 64)
    c0 = Input(shape=(n_s,), name='c0')  # shape of c:  (m, 64)
    
    pred0=Input(shape=(1,len(machine_vocab)), name='pred0')  # shape of pred0 (m ,1, 11)

    
    s = s0
    c = c0
    pred=pred0
    
    print('pred: after Input',pred)

    outputs = []
    
  
    a, forward_h, forward_c, backward_h, backward_c= pre_activation_LSTM_cell(inputs=X) #  shape of a : (m,Tx, 2*n_a)


    s = concatenate_s([forward_h, backward_h]) # shape of s:  (m, 64+64=128)
    c = concatenate_c([forward_c, backward_c])


    for t in range(Ty):
    
      
#         context = one_step_attention(a, s) # shape of context :  (m, 1, 128)
        context=one_step_attention_M2(a, s)
        print('context after one_step_attention: ',context)
        
        


        context=concatenate_context([context,pred])# shape of context: (m,128+11=139) 
       
        print('context after Concatenate:  ',context)
    
        s, _, c = post_activation_LSTM_cell(inputs=context,initial_state=[s, c])#输入只有一个时间步

    
        out = output_layer(s)      
       
 # 1. model 必须全部用 keras 包里的tensor 计算，而不能使用 numpy中的函数，因为都是带上 m个样本 的并行计算（利用GPU加速）      
#         pred= np.argmax(out, axis = -1) 
#         pred=convert_to_one_hot(pred,len(machine_vocab))
        

#2.也不能 直接用keras.backend 中的函数，它的返回仅仅是tensor 
#         pred= K.argmax(out, axis = -1)
#         pred=K.one_hot(pred,len(machine_vocab))
# AttributeError: 'NoneType' object has no attribute '_inbound_nodes'

#3.必须用 keras.layers 中的 它的输入输出 自动会考虑 一个batch 的计算 ；注意对比两个 reshape
# keras.layers.Reshape(target_shape) 输出为 (batch_size,) + target_shape
# keras.backend.reshape(x, shape) 输出为 shape


        pred=lambda_argmax(out)
        pred=lambda_one_hot(pred)
        
        print(pred)

#         pred=RepeatVector(1)(pred)

        pred=reshape(pred)
        print(pred)
          
        outputs.append(out) # shape of out : ( m ,machine_vocab) 
    
    #1.
    #model =  Model(inputs=[X, s0, c0], outputs=outputs) 
    # 未加上 新增的pred0
    #ValueError: Graph disconnected: cannot obtain value for tensor Tensor
    
    #2.
    model =  Model(inputs=[X, s0, c0 ,pred0], outputs=outputs) 
    
    #shape of outputs : ( Ty ,m ,machine_vocab) 
    
    
    return model


# Run the following cell to create your model.

# In[ ]:


model = model(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))


# pred: after Input Tensor("pred0_2:0", shape=(?, 1, 11), dtype=float32)
# context after one_step_attention:  Tensor("model_11/dot_11/MatMul:0", shape=(?, 1, 128), dtype=float32)
# context after Concatenate:   Tensor("concatenate_12_10/concat:0", shape=(?, 1, 139), dtype=float32)
# Tensor("one_hot_tensor_20/one_hot:0", shape=(?, 11), dtype=float32)
# Tensor("reshape_2_10/Reshape:0", shape=(?, 1, 11), dtype=float32)
# context after one_step_attention:  Tensor("model_11_1/dot_11/MatMul:0", shape=(?, 1, 128), dtype=float32)
# context after Concatenate:   Tensor("concatenate_12_11/concat:0", shape=(?, 1, 139), dtype=float32)
# Tensor("one_hot_tensor_21/one_hot:0", shape=(?, 11), dtype=float32)
# Tensor("reshape_2_11/Reshape:0", shape=(?, 1, 11), dtype=float32)
# context after one_step_attention:  Tensor("model_11_2/dot_11/MatMul:0", shape=(?, 1, 128), dtype=float32)
# context after Concatenate:   Tensor("concatenate_12_12/concat:0", shape=(?, 1, 139), dtype=float32)
# Tensor("one_hot_tensor_22/one_hot:0", shape=(?, 11), dtype=float32)
# Tensor("reshape_2_12/Reshape:0", shape=(?, 1, 11), dtype=float32)
# context after one_step_attention:  Tensor("model_11_3/dot_11/MatMul:0", shape=(?, 1, 128), dtype=float32)
# context after Concatenate:   Tensor("concatenate_12_13/concat:0", shape=(?, 1, 139), dtype=float32)
# Tensor("one_hot_tensor_23/one_hot:0", shape=(?, 11), dtype=float32)
# Tensor("reshape_2_13/Reshape:0", shape=(?, 1, 11), dtype=float32)

# pred: after Input Tensor("pred0_3:0", shape=(?, 1, 11), dtype=float32)
# context after one_step_attention:  Tensor("dot_12/MatMul:0", shape=(?, 1, 128), dtype=float32)
# context after Concatenate:   Tensor("concatenate_12_20/concat:0", shape=(?, 1, 139), dtype=float32)
# Tensor("one_hot_tensor_30/one_hot:0", shape=(?, 11), dtype=float32)
# Tensor("reshape_2_20/Reshape:0", shape=(?, 1, 11), dtype=float32)
# context after one_step_attention:  Tensor("dot_12_1/MatMul:0", shape=(?, 1, 128), dtype=float32)
# context after Concatenate:   Tensor("concatenate_12_21/concat:0", shape=(?, 1, 139), dtype=float32)
# Tensor("one_hot_tensor_31/one_hot:0", shape=(?, 11), dtype=float32)
# Tensor("reshape_2_21/Reshape:0", shape=(?, 1, 11), dtype=float32)
# context after one_step_attention:  Tensor("dot_12_2/MatMul:0", shape=(?, 1, 128), dtype=float32)
# context after Concatenate:   Tensor("concatenate_12_22/concat:0", shape=(?, 1, 139), dtype=float32)
# Tensor("one_hot_tensor_32/one_hot:0", shape=(?, 11), dtype=float32)
# Tensor("reshape_2_22/Reshape:0", shape=(?, 1, 11), dtype=float32)

# In[27]:


model2 = model2(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))


# In[ ]:


from keras.utils import plot_model 

plot_model(model, to_file='model.png')


# In[ ]:


from keras.utils import plot_model 

plot_model(model2, to_file='model2.png')


# Let's get a summary of the model to check if it matches the expected output.

# In[ ]:


model.summary()


# In[ ]:


model2.summary()


# **Expected Output**:
# 
# Here is the summary you should see
# <table>
#     <tr>
#         <td>
#             **Total params:**
#         </td>
#         <td>
#          185,484
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **Trainable params:**
#         </td>
#         <td>
#          185,484
#         </td>
#     </tr>
#             <tr>
#         <td>
#             **Non-trainable params:**
#         </td>
#         <td>
#          0
#         </td>
#     </tr>
#                     <tr>
#         <td>
#             **bidirectional_1's output shape **
#         </td>
#         <td>
#          (None, 30, 128)  
#         </td>
#     </tr>
#     <tr>
#         <td>
#             **repeat_vector_1's output shape **
#         </td>
#         <td>
#          (None, 30, 128)  
#         </td>
#     </tr>
#                 <tr>
#         <td>
#             **concatenate_1's output shape **
#         </td>
#         <td>
#          (None, 30, 256) 
#         </td>
#     </tr>
#             <tr>
#         <td>
#             **attention_weights's output shape **
#         </td>
#         <td>
#          (None, 30, 1)  
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **dot_1's output shape **
#         </td>
#         <td>
#          (None, 1, 128) 
#         </td>
#     </tr>
#            <tr>
#         <td>
#             **dense_2's output shape **
#         </td>
#         <td>
#          (None, 11) 
#         </td>
#     </tr>
# </table>
# 

# As usual, after creating your model in Keras, you need to compile it and define what loss, optimizer and metrics your are want to use. Compile your model using `categorical_crossentropy` loss, a custom [Adam](https://keras.io/optimizers/#adam) [optimizer](https://keras.io/optimizers/#usage-of-optimizers) (`learning rate = 0.005`, $\beta_1 = 0.9$, $\beta_2 = 0.999$, `decay = 0.01`)  and `['accuracy']` metrics:

# In[ ]:


### START CODE HERE ### (≈2 lines)
opt = Adam(lr = 0.005, beta_1=0.9, beta_2=0.999, decay = 0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

### END CODE HERE ###


# In[29]:



opt = Adam(lr = 0.005, beta_1=0.9, beta_2=0.999, decay = 0.01)
model2.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


# The last step is to define all your inputs and outputs to fit the model:
# - You already have X of shape $(m = 10000, T_x = 30)$ containing the training examples.
# - You need to create `s0` and `c0` to initialize your `post_activation_LSTM_cell` with 0s.
# - Given the `model()` you coded, you need the "outputs" to be a list of T_y elements of shape (m, 11). So that: `outputs[i][0], ..., outputs[i][Ty]` represent the true labels (characters) corresponding to the $i^{th}$ training example (`X[i]`). More generally, `outputs[i][j]` is the true label of the $j^{th}$ character in the $i^{th}$ training example.

# In[30]:


# tips : 
# n_a = 64
# n_s = 128 
# m = 10000

s0 = np.zeros((m, n_s)) 
c0 = np.zeros((m, n_s))
outputs = list(Yoh.swapaxes(0,1)) #Yoh.swapaxes(0,1) 第0维度 和 第1 维度交换，原来为(m,T_y,11) 变换后 为：(T_y,m,11)

# by model2
pred0=np.zeros((m,1,len(machine_vocab)))


# Let's now fit the model and run it for one epoch.

# In[ ]:


history=model.fit([Xoh, s0, c0], outputs, epochs=40, batch_size=2048,validation_split=0.1)


# In[31]:


#for model2
history2=model2.fit([Xoh, s0, c0, pred0], outputs, epochs=120, batch_size=2048,validation_split=0.1)


# While training you can see the loss as well as the accuracy on each of the 10 positions of the output. The table below gives you an example of what the accuracies could be if the batch had 2 examples: 
# 
# <img src="images/table.png" style="width:700;height:200px;"> <br>
# <caption><center>Thus, `dense_2_acc_8: 0.89` means that you are predicting the 7th character of the output correctly 89% of the time in the current batch of data. </center></caption>
# 
# 
# We have run this model for longer, and saved the weights. Run the next cell to load our weights. (By training a model for several minutes, you should be able to obtain a model of similar accuracy, but loading our model will save you time.) 

# In[ ]:


# 将模型保存到文件 my_model.h5
model2.save('models/xrh_model2.h5')


# In[ ]:


#载入模型 
model2.load_weights('models/xrh_model2.h5')


# In[ ]:


#查看 训练完成的 模型 里面的参数
all_configs=model2.get_config()
all_configs['input_layers']
all_configs['output_layers']
all_configs['layers'][11]
weights = model2.layers[11].get_weights() # Getting params
# model.layers[i].set_weights(weights) # Setting par
weights


# In[ ]:



import tensorflow as tf
import numpy as np
 
 
a=np.array(
[[1,2,3,4,5],
[1,2,3,4,5],
[1,2,3,4,5]]    
)
# a=np.array([[2.29982214e-10,1.05035841e-03,1.04089566e-04,9.98845458e-01,
#   5.10124494e-08,5.03688757e-10,2.52189380e-10,1.18713073e-09,
#   2.30988277e-08,2.21948682e-08,1.48340121e-07]])



input = tf.constant(a)
k = 3
output = tf.nn.top_k(input, k).indices


# one_hot=one_hot_tensor(output,11)
# one_hot=one_hot[0]
# # one_hot[0].shape
# one_hot=K.reshape(one_hot,(1,one_hot.shape[0],one_hot.shape[1]))
# # one_hot.shape
# one_hot_permute=K.permute_dimensions(one_hot,(1,0,2))
# one_hot_permute.shape

with tf.Session() as sess:
    print(sess.run(input))
    print(sess.run(output))
#     print(sess.run(one_hot))
#     print(sess.run(one_hot_permute))


# In[ ]:


#  tf.nn.top_k 输出每一行 的topk 我们希望能输出整个矩阵的 topk

import tensorflow as tf
import numpy as np
 

def all_top_k(input,k):

    flatten=K.flatten(input)
    global_top_k=tf.nn.top_k(flatten, k)
    print('global topk values:',K.eval(global_top_k.values))
    print('glaobal topk indices:',K.eval(global_top_k.indices))
    indices=global_top_k.indices

    indices_row= K.cast(tf.floor(indices/input.shape[-1]),dtype='int32') 
#     K.eval(indices_row)


    indices_col=indices%input.shape[-1] # dtype='int32'
    # indices_col=tf.mod(indices,a.shape[-1]) #  tensorflow 的数学运算 https://blog.csdn.net/zywvvd/article/details/78593618

#     K.eval(indices_col)

    indices=K.concatenate( [K.reshape(indices_row,(1,indices_row.shape[0])) , K.reshape(indices_col,(1,indices_col.shape[0]))] , axis=0)
    indices=K.transpose(indices)
    
    return indices
 
    
a=np.array(
[[1,2,3,4,5],
[1,2,2,2,2],
[1,3,3,3,6]]    
)


k = 3
# result=K.eval(all_top_k(a,k))
# result
# result.shape


decoder_result=[]
# decoder_result=np.zeros((k,Ty))
r0=np.array([3, 1, 2])
r0=np.reshape(r0,(3,1))

# decoder_result[:,0]=b
decoder_result.append(r0)
decoder_result

r=np.array([[0, 1],
             [2, 2],
             [1, 1]])

# r0=decoder_result[0]

r_pre=decoder_result[0]
print('r_pre:',r_pre)

r1=K.cast(K.zeros((k,2)),dtype='int32')

#TODO:  build a empty tensor: r1

#TODO:  少在 tensor 和 numpy 之间的来回转换 可以提升速度？ 全部用tensor 进行计算 

for i in range(k):
    a=K.reshape(r_pre[r[i][0]],(1,r_pre.shape[1])) 
    b=K.reshape(r[i][1],(1,1))
    
    c=K.concatenate( [a,b]  ,axis=1 )
    print(c)
#     r1[i,:].assign( K.concatenate( [a,b]  ,axis=0 ) ) #ValueError: Sliced assignment is only supported for variables
# TODO: 两个 tensor 之间的切片 赋值   
    
    
decoder_result.append(r1)
decoder_result

# r_pre=decoder_result[1]
# print('r_pre:',r_pre)

# r=np.array([[0, 4],
#              [0, 2],
#              [0, 3]])

# r2=np.zeros((k,3))

# for i in range(k):
#     r2[i,:]=np.concatenate( ( r_pre[r[i][0]],[r[i][1]] ),axis=0 )


# decoder_result.append(r2)
# decoder_result


# In[ ]:


##--part1--: 使用 numpy 复现 tf.nn.topk  ## 

# k=3
# arr = np.array([1, 98, 2, 99, 100])
# idx=arr.argsort()[::-1][0:k]
# idx
# arr[idx]# 最大的三个元素 (已排序)

# idx=np.argpartition(arr, k)[0:k]
# arr[idx]#最小的 三个元素

# idx = np.argpartition(arr, -k)[-k:]
# idx
# arr[idx]#最大的 三个元素 (未排序)

# a=np.array(
# [[1,2,3,4,5],
# [1,2,8,2,2],
# [9,3,3,3,6]]    
# )
# np.argpartition(a, -k)
# idx = np.argpartition(a, -k)[ :,-k:]
# idx


def topk_array(matrix, k, axis=1):
    """
    perform topK based on np.argsort
    :param matrix: to be sorted
    :param K: select and sort the top K items
    :param axis: dimension to be sorted.
    :return:
    """
    full_sort = np.argsort(matrix, axis=axis)
    return full_sort[ :,-k:]

def partition_topk_array(matrix, K, axis=1):
    """
    perform topK based on np.argpartition
    :param matrix: to be sorted
    :param K: select and sort the top K items
    :param axis: 0 or 1. dimension to be sorted.
    :return:
    """
    a_part = np.argpartition(matrix, -K, axis=axis)
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        a_sec_argsort_K = np.argsort(matrix[a_part[-K:, :], row_index], axis=axis)
        return a_part[-K:, :][a_sec_argsort_K, row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
#         print('column_index ',column_index)
#         print('matrix[column_index, a_part[:, -K:]] ',matrix[column_index, a_part[:, -K:]]) #选取矩阵中的一组元素
        a_sec_argsort_K = np.argsort(matrix[column_index, a_part[:, -K:]], axis=axis)
#         print('a_sec_argsort_K ',a_sec_argsort_K)
        return a_part[:, -K:][column_index, a_sec_argsort_K] # 乾坤大挪移，变换矩阵中的元素位置

    

# arr = np.array([[1, 98, 2, 99, 100]])

a=np.array(
[[1,2,3,4,5],
[1,2,8,2,2],
[9,3,3,3,6]]    
)
k=3
# partition_topk_array(a, k, axis=1) 

# partition_topk_array(arr, k, axis=1)# 最大的 三个元素 (已排序)


## --ref: 
# https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
# https://stackoverflow.com/questions/41484104/how-numpy-partition-work
# https://blog.csdn.net/SoftPoeter/article/details/86629329
##--part1-- end --##

##--part2--: arg_topK 输出 二维矩阵的 每一行 的topk，我们希望能输出整个矩阵的 topk

def whole_topk_array(input,k):
    """
    输出 input 中所有元素中的 k 个最大的元素的下标，但是这k个元素并不会按照大小排序
    """

    flatten=input.flatten()
#     print(flatten)
    
    global_top_k=np.argpartition(flatten, -k)[-k:]
    
    indices=global_top_k

    indices_row= np.floor(indices/input.shape[-1])
    

    indices_col=indices%input.shape[-1] # dtype='int32'


    indices=np.concatenate( [np.reshape(indices_row,(1,indices_row.shape[0])) , np.reshape(indices_col,(1,indices_col.shape[0]))] , axis=0)
    indices=np.transpose(indices)
    
    return indices.astype(np.int32) # numpy 数据类型转换 ；查看数据类型： arr.dtype


# whole_topk_array(a,k)

##--part2-- end --##

##--part3--: 使用 numpy 复现   tf.one_hot()

# a.reshape(-1,3) # 固定3列 (-1)=多少行不知道，numpy自己算去吧
# a.reshape(-1) #  flatten a 

def one_hot_array(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
#     print(res.shape)
    return res.reshape(list(targets.shape)+[nb_classes])

# a.shape
# b=one_hot_array(a,11)
# b.shape

##--ref: 
# https://stackoverflow.com/questions/38592324/one-hot-encoding-using-numpy
##--part3-- end --##


# In[ ]:


get_ipython().run_cell_magic('time', '', '# By XRH in 2019.9.10\n# beamsearch 的两种实现方式，并比较时间开销  \n\n# by : https://stackoverflow.com/questions/48374905/how-can-i-use-argsort-in-keras\ndef top_k(input, k):\n  # Can also use `.values` to return a sorted tensor\n  return tf.nn.top_k(input, k=k, sorted=True)\n\n\ndef all_top_k(input,k):\n    """\n     tf.nn.top_k 输出每一行 的topk 我们希望能输出整个矩阵的 topk\n    """\n\n    flatten=K.flatten(input)\n    global_top_k=tf.nn.top_k(flatten, k)\n#     print(\'global topk values:\',K.eval(global_top_k.values))\n#     print(\'glaobal topk indices:\',K.eval(global_top_k.indices))\n    indices=global_top_k.indices\n\n    indices_row= K.cast(tf.floor(indices/input.shape[-1]),dtype=\'int32\') \n#     K.eval(indices_row)\n\n\n    indices_col=indices%input.shape[-1] # dtype=\'int32\'\n    # indices_col=tf.mod(indices,a.shape[-1]) #  tensorflow 的数学运算 https://blog.csdn.net/zywvvd/article/details/78593618\n\n#     K.eval(indices_col)\n\n    indices=K.concatenate( [K.reshape(indices_row,(1,indices_row.shape[0])) , K.reshape(indices_col,(1,indices_col.shape[0]))] , axis=0)\n    indices=K.transpose(indices)\n    \n    return indices\n\n\ndef model2_onestep_decode(Tx, Ty,timestep, n_a, n_s, human_vocab_size, machine_vocab_size):\n    """\n    Arguments:\n    Tx -- length of the input sequence\n    timestep -- timestep of decoder \n    Ty -- length of the output sequence\n    n_a -- hidden state size of the Bi-LSTM\n    n_s -- hidden state size of the post-attention LSTM\n    human_vocab_size -- size of the python dictionary "human_vocab"\n    machine_vocab_size -- size of the python dictionary "machine_vocab"\n\n    Returns:\n    model -- Keras model instance\n    """\n    \n    \n    X = Input(shape=(Tx, human_vocab_size)) # shape: (m,Tx,human_vocab_size)\n    s0 = Input(shape=(n_s,), name=\'s\')  # shape of s:  (m, 64)\n    c0 = Input(shape=(n_s,), name=\'c\')  # shape of c:  (m, 64)\n    \n    pred0=Input(shape=(1,len(machine_vocab)), name=\'pred\')  # shape of pred (m ,1, 11)\n    \n    s=s0 # unmutable object: a new tensor is generated \n    c=c0\n    pred=pred0\n    \n    \n#     print(\'pred: after Input\',pred)\n\n    \n    a, forward_h, forward_c, backward_h, backward_c= pre_activation_LSTM_cell(inputs=X) #  shape of a : (m,Tx, 2*n_a) \n    #TODO：这一步的推理是多余的，可以把 encoder 和 decoder 彻底解耦\n\n    \n    if timestep==0: # decoder 的第一个时间步\n\n            s = concatenate_s([forward_h, backward_h]) # shape of s:  (m, 64)\n            c = concatenate_c([forward_c, backward_c])\n            \n            context = one_step_attention(a, s) # shape of context :  (m, 1, 128)\n       \n            context=concatenate_context([context,pred])# shape of context: (m,128+11=139)\n\n            s, _, c = post_activation_LSTM_cell(inputs=context,initial_state=[s, c])\n\n            out = output_layer(s)   \n    else:\n            \n            context = one_step_attention(a, s) # shape of context :  (m, 1, 128)\n       \n            context=concatenate_context([context,pred])# shape of context: (m,128+11=139)\n\n            s, _, c = post_activation_LSTM_cell(inputs=context,initial_state=[s, c])\n\n            out = output_layer(s)    \n    \n    outputs=[s,c,out] # 输出 s c out 作为下一个时间步使用\n        \n          \n    model =  Model(inputs=[X, s0, c0 ,pred0], outputs=outputs) \n    \n    return model\n\n\n\ndef beamsearch(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size,k=3):\n    """\n    @deprecated: too slow\n    cost time: 1min30s\n    """\n    \n    s0 = np.zeros((k, n_s))\n    c0 = np.zeros((k, n_s))\n    pred0=np.zeros((k,1,len(machine_vocab)))\n    \n    s=s0\n    c=c0\n    pred=pred0\n    \n    decoder_result=[]\n    \n    for timestep in range(Ty):\n        \n        onestep_decode = model2_onestep_decode(Tx, Ty,timestep, n_a, n_s, len(human_vocab), len(machine_vocab))\n        s,c,out=onestep_decode.predict([source_oh, s, c,pred])\n        \n        \n        if timestep==0:\n            print(\'timestep :\', timestep)\n\n#             print (\'out:\',out) # shape:(3, 11)\n\n            out_top_K=top_k(out, k).indices  #  shape:(3,3) \n           \n            top_K_indices=K.eval(out_top_K) # cost much time\n            \n            r0=top_K_indices[0]\n            \n            r0=np.reshape(r0,(k,1))\n            decoder_result=r0\n            \n            one_hot=one_hot_tensor(out_top_K,machine_vocab_size )\n#             print (K.eval(one_hot)) # tensor shape:(3,3,11)  \n            \n            one_hot=one_hot[0]\n#             print(K.eval(one_hot)) #  shape:(3, 11) for debug, get the value of tensor\n            one_hot=K.reshape(one_hot,(1,one_hot.shape[0],one_hot.shape[1])) #shape:(1,3, 11)\n            one_hot_permute=K.permute_dimensions(one_hot,(1,0,2)) #shape: (3,1,11)\n            \n            pred=K.eval(one_hot_permute) # tensor -> numpy array  \n#             print(\'pred shape:\',pred.shape)\n        \n        else:\n            print(\'timestep :\', timestep)\n#             print (\'out:\',out)\n            \n            out_top_K=all_top_k(out,k)\n            \n            r=K.eval(out_top_K)\n#             print(\'r:\',r)\n            \n            r_pre=decoder_result\n    \n#             print(\'r_pre:\',r_pre)\n\n            rt=np.zeros((k,timestep+1))\n\n            for i in range(k):\n\n                rt[i,:]=np.concatenate( ( r_pre[r[i][0]],[r[i][1]] ) , axis=0 )\n\n            decoder_result=rt\n            \n            \n            one_hot=one_hot_tensor(r[:,1],machine_vocab_size )\n#             print (K.eval(one_hot)) # shape:(3, 11)\n            \n            one_hot=K.reshape(one_hot,(1,one_hot.shape[0],one_hot.shape[1])) #shape:(1,3, 11)\n#             print(\'one_hot shape:\',one_hot.shape)\n            one_hot_permute=K.permute_dimensions(one_hot,(1,0,2)) #shape: (3,1,11)\n            \n            pred=K.eval(one_hot_permute) # tensor -> numpy array  \n#             print(\'pred shape:\',pred.shape)\n            \n            \n        print(\'decoder_result\',decoder_result) \n    \n    return   decoder_result  \n        \n\ndef beamsearch_v1(source_oh,Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size,k):\n    """\n    np.array -> tensor 很自然 但是 tensor -> np.array 的方法： K.eval(tensor) 非常耗费时间；\n    beamsearch_v1 尝试尽量多用 numpy 库的函数，以减少 tensor 和 np.array 的转换的次数。\n    cost time: 6.13 s\n    """\n    \n    s0 = np.zeros((k, n_s))\n    c0 = np.zeros((k, n_s))\n    pred0=np.zeros((k,1,len(machine_vocab)))\n    \n    s=s0\n    c=c0\n    pred=pred0\n    \n    decoder_result=[]\n    \n    for timestep in range(Ty):\n        \n        onestep_decode = model2_onestep_decode(Tx, Ty,timestep, n_a, n_s, len(human_vocab), len(machine_vocab))\n        s,c,out=onestep_decode.predict([source_oh, s, c,pred]) \n        #source_oh shape：(3, 30, 37) \n        #每次都对 3个相同的样本（k=3）进行 推理，但是每一个 样本对应的 pred 不同 ；\n        #这实现了beamsearch 中，每一个时间步都会根据上一步的 onestep_decoder 输出结果中 选择最好的k个, 输入 onestep_decoder \n        \n        if timestep==0:\n            print(\'timestep :\', timestep)\n\n#             print (\'out:\',out) # shape:(3, 11)\n\n            out_top_K=partition_topk_array(out, k)  #  shape:(3,3) \n            print(out_top_K)\n           \n            top_K_indices=out_top_K \n            \n            r0=top_K_indices[0]\n            \n            r0=np.reshape(r0,(k,1))\n            decoder_result=r0\n            \n            one_hot=one_hot_array(out_top_K,machine_vocab_size )#  shape:(3,3,11)\n            \n            one_hot=one_hot[0]#  shape:(3, 11) \n            one_hot=np.reshape(one_hot,(1,one_hot.shape[0],one_hot.shape[1])) #shape:(1,3, 11)\n            \n            one_hot_permute=one_hot.transpose((1,0,2)) #shape: (3,1,11)\n            \n            pred=one_hot_permute   \n        \n        else:\n            print(\'timestep :\', timestep)\n#             print (\'out:\',out)\n            \n            out_top_K=whole_topk_array(out,k)\n            \n            r=out_top_K\n            \n            r_pre=decoder_result\n    \n\n            rt=np.zeros((k,timestep+1))\n\n            for i in range(k):\n                \n\n                rt[i,:]=np.concatenate( ( r_pre[r[i][0]],[r[i][1]] ) , axis=0 )\n\n            decoder_result=rt\n            \n            \n            one_hot=one_hot_array(r[:,1],machine_vocab_size ) # shape:(3, 11)\n            \n            one_hot=np.reshape(one_hot,(1,one_hot.shape[0],one_hot.shape[1])) #shape:(1,3, 11)\n            one_hot_permute=one_hot.transpose((1,0,2)) #shape: (3,1,11)\n            \n            pred=one_hot_permute\n            \n            \n        print(\'decoder_result\',decoder_result) \n    \n    return   decoder_result  \n        \n    \nexample = "3rd of March 2002"\nsource = np.array(string_to_int(example, Tx, human_vocab))\nsource_oh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source)))\n\nk=3\nsource_oh=source_oh.reshape(1,source_oh.shape[0],source_oh.shape[1])   \n# print(source_oh.shape) \nsource_oh=np.repeat(source_oh, k, axis=0) \nprint(source_oh.shape) #(3, 30, 37) m=3 一个样本 复制为三个输入模型进行推理\n\n  \ndecoder_result=beamsearch_v1(source_oh,Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab),k) \n\nfor prediction in decoder_result:\n    output = int_to_string(prediction, inv_machine_vocab)\n    print("source:", example)\n    print("output:", \'\'.join(output))\n    \n')


# In[ ]:


# %%time
#   By XRH in 2019.9.17
#  对 encoder 和 decoder 进行解耦

def model2_onestep_decoder_v2(Tx, Ty,timestep, n_a, n_s, human_vocab_size, machine_vocab_size):
    """
    Arguments:
    Tx -- length of the input sequence
    timestep -- timestep of decoder 
    Ty -- length of the output sequence
    n_a -- hidden state size of the Bi-LSTM
    n_s -- hidden state size of the post-attention LSTM
    human_vocab_size -- size of the python dictionary "human_vocab"
    machine_vocab_size -- size of the python dictionary "machine_vocab"

    Returns:
    model -- Keras model instance
    """
    
    context0=Input(shape=(1,n_s), name='context')
    
    s0 = Input(shape=(n_s,), name='s')  # shape of s:  (m, 64)
    c0 = Input(shape=(n_s,), name='c')  # shape of c:  (m, 64)
    
    pred0=Input(shape=(1,len(machine_vocab)), name='pred')  # shape of pred (m ,1, 11)
    
    context=context0
    s=s0 # unmutable object: a new tensor is generated 
    c=c0
    pred=pred0
     
#     print('pred: after Input',pred)

    context=concatenate_context([context,pred])# shape of context: (m,128+11=139)

    s, _, c = post_activation_LSTM_cell(inputs=context,initial_state=[s, c])

    out = output_layer(s)    
    
    outputs=[s,c,out] # 输出 s c out 作为下一个时间步使用
        
          
    model =  Model(inputs=[context0 ,s0, c0 ,pred0], outputs=outputs) 
    
    return model


def model2_encoder(Tx, human_vocab_size):
    
    X = Input(shape=(Tx, human_vocab_size)) # shape: (m,Tx,human_vocab_size)
    
    a, forward_h, forward_c, backward_h, backward_c= pre_activation_LSTM_cell(inputs=X) #  shape of a : (m,Tx, 2*n_a) 
 
    s = concatenate_s([forward_h, backward_h]) # shape of s:  (m, 64+64)
    c = concatenate_c([forward_c, backward_c])
    
    context = one_step_attention(a, s) # shape of context :  (m, 1, 128)
    
    outputs=[context,s,c]
    
    model =  Model(inputs=[X], outputs=outputs) 
    
    return model
    

def beamsearch_v2(source_oh,Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size,k=3):
    """
    np.array -> tensor 很自然 但是 tensor -> np.array 的方式： K.eval(tensor) 非常耗费时间；
    beamsearch_v1 尝试尽量多用 numpy 的函数，以减少 tensor 和 np 的转换的次数。
    cost time: 6.13 s
    """
    
    s0 = np.zeros((k, n_s))
    c0 = np.zeros((k, n_s))
    pred0=np.zeros((k,1,len(machine_vocab)))
    
    s=s0
    c=c0
    pred=pred0
    
    decoder_result=[]

#--encoder 和 decoder 的解耦
#bearm serach encoder：
# M1: 
#     encoder_output = Model(inputs=model2.input,  
#         outputs=[ model2.get_layer('decoder_output').get_output_at(1) ]) #TODO
#     a,s,c= encoder_output.predict([source_oh, s0, c0,pred0])

# M2:

    encoder=model2_encoder(Tx, human_vocab_size)
    context,s,c=encoder.predict([source_oh])

#bearm serach decoder：

    for timestep in range(Ty):
        
        onestep_decoder = model2_onestep_decoder_v2(Tx, Ty,timestep, n_a, n_s, len(human_vocab), len(machine_vocab))
        s,c,out=onestep_decoder.predict([context, s, c,pred]) 
        #source_oh shape：(3, 30, 37) 
        #每次都对 3个相同的样本（k=3）进行 推理，但是每一个 样本对应的 pred 不同 ；
        #这实现了beamsearch 中，每一个时间步都会根据上一步的 onestep_decoder 输出结果中 选择最好的k个, 输入 onestep_decoder 
        
        if timestep==0:
            print('timestep :', timestep)

#             print ('out:',out) # shape:(3, 11) softmax 层输出的为 11 unit 的概率；输入的样本数量为3

            out_top_K=partition_topk_array(out, k)  #  shape:(3,3) 从每个样本 的 11unit 中选出最大的k个 
            print(out_top_K)
           
            top_K_indices=out_top_K 
            
            r0=top_K_indices[0] #  shape:(1,3) 因为3个输入样本是一样的，取其中一个即可 
            
            r0=np.reshape(r0,(k,1)) # shape:(3,1)
            decoder_result=r0
            
            one_hot=one_hot_array(out_top_K,machine_vocab_size )#  shape:(3,3,11)
            # 把 out_top_K shape:(3,3) 最后一个维度 变为 one-hot 向量
            
            one_hot=one_hot[0]#  shape:(3, 11) ；只要取一个即可
            one_hot=np.reshape(one_hot,(1,one_hot.shape[0],one_hot.shape[1])) #shape:(1,3, 11)
            
            one_hot_permute=one_hot.transpose((1,0,2)) #shape: (3,1,11) ；
            #交换 第0维 和 第1维，相当于3个不同的 pred 同时输入下一个时间步的 onestep_decoder
            pred=one_hot_permute   
        
        else:
            print('timestep :', timestep)
#             print ('out:',out) # shape:(3, 11)
            
            out_top_K=whole_topk_array(out,k)  #  shape:(3, 2) 找出 3*11 个元素中的k个最大的 元素的标号 
            
            r=out_top_K 
            print('r:',r) 
#             [[1 1]    
#              [0 1]
#              [2 1]] 元素的标号为 [2,1] ，代表 第3个输入的pred 所输出的11个uints中的第1个unit
            
            r_pre=decoder_result # shape:(k,timestep) 上一步 解码的结果 即是 这一步的输入 
            # [[2]
            #  [1]
            #  [3]]
    
            rt=np.zeros((k,timestep+1)) #这一步 会在上一步 已有的解码序列的基础上 增加1个 解码位

            for i in range(k):
                
                rt[i,:]=np.concatenate( ( r_pre[r[i][0]],[r[i][1]] ) , axis=0 )
                # r[2][0]=2 说明是第二个输入的pred，前一步的解码情况为： r_pre[r[2][0]]=[2] ，
                #再连接上这一步的解码位  r[2][1]=1 得到 解码序列：[2,1]
                # 一共k 个解码序列 组成 rt
                
            decoder_result=rt.astype(np.int32) 
#               rt:
#              [[1. 1.] 
#              [2. 1.]
#              [3. 1.]]

#              decoder_result:
#              [[1 1] 
#              [2 1]
#              [3 1]]
            
            
            one_hot=one_hot_array(decoder_result[:,-1],machine_vocab_size ) # shape:(3, 11)
#             print(one_hot.shape)
            
            one_hot=np.reshape(one_hot,(1,one_hot.shape[0],one_hot.shape[1])) #shape:(1,3, 11)
            one_hot_permute=one_hot.transpose((1,0,2)) #shape: (3,1,11)
            
            pred=one_hot_permute
            
        print('decoder_result',decoder_result) 
    
    return   decoder_result  
        
    
example = "3rd of March 2002"
source = np.array(string_to_int(example, Tx, human_vocab))
source_oh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source)))

k=3
source_oh=source_oh.reshape(1,source_oh.shape[0],source_oh.shape[1])   
# print(source_oh.shape) 
source_oh=np.repeat(source_oh, k, axis=0) 
# print(source_oh.shape) #(3, 30, 37) m=3 一个样本 复制为三个输入模型进行推理

  
decoder_result=beamsearch_v2(source_oh,Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab),k) 

for prediction in decoder_result:
    output = int_to_string(prediction, inv_machine_vocab)
    print("source:", example)
    print("output:", ''.join(output))
    


# Draw pictures to show the training process and the accuracy in the verification set and training set.

# In[ ]:


#by model 


#---对 decoder 的10个时间步的 准确率求均值 --#
#--start--#
# Ty = 10
Epoch_num=(history.history['dense_3_acc'])

acc0=np.array(history.history['dense_3_acc'])
acc0=acc0.reshape(40,1)

acc=acc0

for i in range(Ty):
    if i != 0:
        
        acc_t=np.array(history.history['dense_3_acc_'+str(i)])
        acc_t=acc_t.reshape(40,1)
        acc=np.concatenate([acc,acc_t],axis=1)
  

acc=np.mean(acc,axis=1)
print('acc.shape:',acc.shape)   

val_acc0=np.array(history.history['val_dense_3_acc'])
val_acc0=val_acc0.reshape(40,1)

val_acc=val_acc0

for i in range(Ty):
    if i != 0:
        
        val_acc_t=np.array(history.history['val_dense_3_acc_'+str(i)])
        val_acc_t=val_acc_t.reshape(40,1)
        val_acc=np.concatenate([val_acc,val_acc_t],axis=1)  

val_acc=np.mean(val_acc,axis=1)
print(val_acc.shape)  
#--- end --#

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.figure( figsize=(8,4), dpi=100 )

plt.subplot(1,2,1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# In[ ]:


#by model2
#---对 decoder 的10个时间步的 准确率求均值 --#
#--start--#
# Ty = 10
Epoch_num=len(history2.history['decoder_output_acc'])

acc0=np.array(history2.history['decoder_output_acc'])
acc0=acc0.reshape(Epoch_num,1)

acc=acc0

for i in range(Ty):
    if i != 0:
        
        acc_t=np.array(history2.history['decoder_output_acc_'+str(i)])
        acc_t=acc_t.reshape(Epoch_num,1)
        acc=np.concatenate([acc,acc_t],axis=1)
  

acc=np.mean(acc,axis=1)
print('acc.shape:',acc.shape)   

val_acc0=np.array(history2.history['val_decoder_output_acc'])
val_acc0=val_acc0.reshape(Epoch_num,1)

val_acc=val_acc0

for i in range(Ty):
    if i != 0:
        
        val_acc_t=np.array(history2.history['val_decoder_output_acc_'+str(i)])
        val_acc_t=val_acc_t.reshape(Epoch_num,1)
        val_acc=np.concatenate([val_acc,val_acc_t],axis=1)  

val_acc=np.mean(val_acc,axis=1)
print(val_acc.shape)  
#--- end --#

loss = history2.history['loss']
val_loss = history2.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.figure( figsize=(25,15), dpi=200 )

plt.subplot(1,2,1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# You can now see the results on new examples.

# In[ ]:


example = "3rd of March 2002"
source = np.array(string_to_int(example, Tx, human_vocab))


# model input: [Xoh, s0, c0]

print("Xoh.shape:", Xoh.shape)
print("source.shape:", source.shape)

source_oh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source)))
print("source_oh shape:", source_oh.shape)
source_oh=source_oh.reshape(1,source_oh.shape[0],source_oh.shape[1])
print("source_oh shape after reshape:", source_oh.shape)

prediction = model2.predict([source_oh, s0, c0,pred00])
# prediction = model.predict([source_oh, s0, c0])

prediction=np.array(prediction)
print('prediction.shape:',prediction.shape)

prediction=prediction.swapaxes(0,1)


print("Yoh.shape:", Yoh.shape)
print("prediction.shape:", prediction.shape)
prediction = np.argmax(prediction[0], axis = -1)
prediction
output = int_to_string(prediction, inv_machine_vocab)
print("source:", example)
print("output:", ''.join(output))


# In[ ]:


#输出 模型 中间层的计算结果
# M1  
# get_layer12_output_timestep_9 = K.function([model2.layers[0].input],
#                                   [model2.layers[12].get_output_at(9)])
# layer_output = get_layer12_output_timestep_9([source_oh, s0, c0,pred0])[0] #TODO exisits error 

# M2


example = "3rd of March 2002"
source = np.array(string_to_int(example, Tx, human_vocab))

print("Xoh.shape:", Xoh.shape)
print("source.shape:", source.shape)

source_oh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source)))
print("source_oh shape:", source_oh.shape)
source_oh=source_oh.reshape(1,source_oh.shape[0],source_oh.shape[1])
print("source_oh shape after reshape:", source_oh.shape)


layer_timestep_0_1 = Model(inputs=model2.input,
                                     outputs=[model2.get_layer('encoder_lstm').get_output_at(0)])
#TODO: error : Output tensors to a Model must be the output of a Keras `Layer`
#如何通过 get_output_at 拿到lstm layer 的输出


layer_timestep_0_1 = layer_timestep_0_1.predict([source_oh, s0, c0,pred0]) 
# dense3_output_timestep_9.shape
layer_timestep_0_1.shape


model2.get_layer('encoder_lstm')





# ## 5 - BLEU score
# 
# In this last part, you are going to implement the BLEU score to assess the effectiveness 

# In[ ]:


from bleu import compute_bleu
reference = [['this', 'is', 'small', 'test']]
candidate = ['this', 'is', 'a', 'test']

score = compute_bleu(reference, candidate)
print (score)


from nltk.translate.bleu_score import sentence_bleu
print('Cumulative 1-gram: %f' % sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))
print('Cumulative 2-gram: %f' % sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0)))
print('Cumulative 3-gram: %f' % sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0)))
print('Cumulative 4-gram: %f' % sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25)))


# In[ ]:


from bleu import compute_bleu,_get_ngrams


# EXAMPLES = ['3 May 1979', '5 April 09', '21th of August 2016', 'Tue 10 Jul 2007', 'Saturday May 9 2018', 'March 3 2001', 'March 3rd 2001', '1 March 2001']
# print('shape Xoh:',np.shape(Xoh))

One_EXAMPLES = ['3 May 1979']
EXAMPLES = ['3 May 1979', '5 Apr 09', '20th February 2016', 'Wed 10 Jul 2007']
GROUND_TRUTH = ['1979-05-03', '2009-04-05', '2016-02-20', '2007-07-10']


for index,example in enumerate(EXAMPLES):
    
    source = string_to_int(example, Tx, human_vocab)
#     print (source)
    source = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source)))#.swapaxes(0,1)
#     print(np.shape(source))
    
    prediction = model.predict([np.reshape(source,[1,np.shape(Xoh)[1],-1]), s0, c0])
#     print(np.shape(prediction))
    prediction = np.argmax(prediction, axis = -1) #作用于 最后一维的特征
#     print (np.shape(prediction))
#     print (prediction)
    output = [inv_machine_vocab[int(i)] for i in prediction]
    output=''.join(output)
    print("source:", example)
    print("output:", output)
    
    target=GROUND_TRUTH[index]
    print("target:", target)
    
    
    print("BLEU score: ", compute_bleu([target.split('-')], output.split('-'))[0])
    
#     print("BLEU score: ", compute_bleu([[ch for ch in target]],[ch for ch in output] )[0])
    
   


# You can also change these examples to test with your own examples. The next part will give you a better sense on what the attention mechanism is doing--i.e., what part of the input the network is paying attention to when generating a particular output character. 

# ## 3 - Visualizing Attention (Optional / Ungraded)
# 
# Since the problem has a fixed output length of 10, it is also possible to carry out this task using 10 different softmax units to generate the 10 characters of the output. But one advantage of the attention model is that each part of the output (say the month) knows it needs to depend only on a small part of the input (the characters in the input giving the month). We can  visualize what part of the output is looking at what part of the input.
# 
# Consider the task of translating "Saturday 9 May 2018" to "2018-05-09". If we visualize the computed $\alpha^{\langle t, t' \rangle}$ we get this: 
# 
# <img src="images/date_attention.png" style="width:600;height:300px;"> <br>
# <caption><center> **Figure 8**: Full Attention Map</center></caption>
# 
# Notice how the output ignores the "Saturday" portion of the input. None of the output timesteps are paying much attention to that portion of the input. We see also that 9 has been translated as 09 and May has been correctly translated into 05, with the output paying attention to the parts of the input it needs to to make the translation. The year mostly requires it to pay attention to the input's "18" in order to generate "2018." 
# 
# 

# ### 3.1 - Getting the activations from the network
# 
# Lets now visualize the attention values in your network. We'll propagate an example through the network, then visualize the values of $\alpha^{\langle t, t' \rangle}$. 
# 
# To figure out where the attention values are located, let's start by printing a summary of the model .

# In[ ]:


model.summary()


# Navigate through the output of `model.summary()` above. You can see that the layer named `attention_weights` outputs the `alphas` of shape (m, 30, 1) before `dot_2` computes the context vector for every time step $t = 0, \ldots, T_y-1$. Lets get the activations from this layer.
# 
# The function `attention_map()` pulls out the attention values from your model and plots them.

# In[ ]:


attention_map = plot_attention_map(model, human_vocab, inv_machine_vocab, "Tuesday April 08 1993", num = 6, n_s = 128)


# On the generated plot you can observe the values of the attention weights for each character of the predicted output. Examine this plot and check that where the network is paying attention makes sense to you.
# 
# In the date translation application, you will observe that most of the time attention helps predict the year, and hasn't much impact on predicting the day/month.

# ### Congratulations!
# 
# 
# You have come to the end of this assignment 
# 
# <font color='blue'> **Here's what you should remember from this notebook**:
# 
# - Machine translation models can be used to map from one sequence to another. They are useful not just for translating human languages (like French->English) but also for tasks like date format translation. 
# - An attention mechanism allows a network to focus on the most relevant parts of the input when producing a specific part of the output. 
# - A network using an attention mechanism can translate from inputs of length $T_x$ to outputs of length $T_y$, where $T_x$ and $T_y$ can be different. 
# - You can visualize attention weights $\alpha^{\langle t,t' \rangle}$ to see what the network is paying attention to while generating each output.

# Congratulations on finishing this assignment! You are now able to implement an attention model and use it to learn complex mappings from one sequence to another. 
