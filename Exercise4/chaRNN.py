import torch
import numpy as np
import random

TEXT_FILE = 'input.txt'
EPOCHS = 10000
BATCH_SIZE = 2048
TRAIN_LENGTH = 256
HIDDEN_SIZE = 512
HIDDEN_NR = 1

with open(TEXT_FILE, 'r') as f:
    text = f.read()

int2char = dict(enumerate(list(set(text))))
char2int = {}
for i, char in int2char.items():
    char2int[char] = i

text = [char2int[char] for char in text]

use_cuda = torch.cuda.is_available()

def get_batch(batch_size, train_length):
    r = random.randint(0,len(text)-(batch_size*train_length+1))
    source = text[r:r+batch_size*train_length]
    target = text[r+1:r+batch_size*train_length+1]

    source = np.eye(len(int2char))[source]
    target = np.eye(len(int2char))[target]

    source = source.reshape((train_length, batch_size, len(int2char)))
    target = target.reshape((train_length, batch_size, len(int2char)))

    source = torch.from_numpy(source).float()
    target = torch.from_numpy(target).float()

    if use_cuda:
        source = source.cuda()
        target = target.cuda()

    return (source, target)



RNN = torch.nn.LSTM(input_size = len(int2char),
                    hidden_size = HIDDEN_SIZE,
                    num_layers = HIDDEN_NR,
                    dropout = 0.5,
)

DENSE = torch.nn.Linear(HIDDEN_SIZE, len(int2char))
SOFTMAX = torch.nn.Softmax()


optimizer = torch.optim.SGD(list(RNN.parameters())+list(DENSE.parameters()), lr=0.01)

loss_function = torch.nn.MSELoss()

h0 = torch.zeros(HIDDEN_NR, BATCH_SIZE, HIDDEN_SIZE).float()
c0 = torch.zeros(HIDDEN_NR, BATCH_SIZE, HIDDEN_SIZE).float()

if use_cuda:
    RNN.cuda()
    DENSE.cuda()
    h0.cuda()
    c0.cuda()

RNN.train()
DENSE.train()
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    source, target = get_batch(BATCH_SIZE, TRAIN_LENGTH)
    if use_cuda:
        source.cuda()
        target.cuda()
    output, (hn, cn) = RNN(source, (h0.cuda(),c0.cuda()))
    output = DENSE(output)
    output = SOFTMAX(output)
    loss = loss_function(output, target)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(loss)
        #print(output.shape)
        #print(output[:,-1,:].shape)
        print(torch.argmax(output[:,-1,:],-1))
        print(''.join([int2char[x.item()] for x in torch.argmax(output[:,-1,:],-1)]))
        #print(target[:,-1,:],-1)
