import torch
import torch.nn as nn
from torch.autograd import Variable
import unidecode
import string
import random
import time
from tqdm import tqdm
import math
import os
import matplotlib.pyplot as plt

class CharLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, n_layers=1):
        super(CharLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, vocab_size)

    def forward(self, input, hidden):
        batch_size = input.size(0)
        encoded = self.encoder(input)
        output, hidden = self.lstm(encoded.view(1, batch_size, -1), hidden)
        output = self.decoder(output.view(batch_size, -1))
        return output, hidden

    def forward2(self, input, hidden):
        encoded = self.encoder(input.view(1, -1))
        output, hidden = self.lstm(encoded.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self, batch_size):
        return [Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))]


BATCH_SIZE = 100
CHUNK_LEN = 200
LEARNING_RATE = 0.01
FILE_NAME = 'input.txt'
HIDDEN_SIZE = 100
N_LAYERS = 2
EPOCHS = 2000 #2000
LOAD = None

use_cuda = torch.cuda.is_available()

all_characters = string.printable
n_characters = len(all_characters)

def read_file(filename):
    file = unidecode.unidecode(open(filename).read())
    return file, len(file)

file, file_len = read_file(FILE_NAME)

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        try:
            tensor[c] = all_characters.index(string[c])
        except:
            continue
    return tensor

def random_training_set(chunk_len, batch_size):
    inp = torch.LongTensor(batch_size, chunk_len)
    target = torch.LongTensor(batch_size, chunk_len)
    for bi in range(batch_size):
        start_index = random.randint(0, file_len - chunk_len)
        end_index = start_index + chunk_len + 1
        chunk = file[start_index:end_index]
        inp[bi] = char_tensor(chunk[:-1])
        target[bi] = char_tensor(chunk[1:])
    inp = Variable(inp)
    target = Variable(target)
    if use_cuda:
        inp = inp.cuda()
        target = target.cuda()
    return inp, target

def generate(decoder, prime_str='A', predict_len=100, temperature=0.8, cuda=False):
    hidden = decoder.init_hidden(1)
    prime_input = Variable(char_tensor(prime_str).unsqueeze(0))

    if cuda:
        hidden[0] = hidden[0].cuda()
        hidden[1] = hidden[1].cuda()
        prime_input = prime_input.cuda()
    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = decoder(prime_input[:,p], hidden)
        
    inp = prime_input[:,-1]
    
    for p in range(predict_len):
        output, hidden = decoder(inp, hidden)
        
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = all_characters[top_i]
        predicted += predicted_char
        inp = Variable(char_tensor(predicted_char).unsqueeze(0))
        if cuda:
            inp = inp.cuda()

    return predicted

def train(inp, target):
    hidden = decoder.init_hidden(BATCH_SIZE)
    if use_cuda:
        hidden[0] = hidden[0].cuda()
        hidden[1] = hidden[1].cuda()
    decoder.zero_grad()
    loss = 0

    for c in range(CHUNK_LEN):
        output, hidden = decoder(inp[:,c], hidden)
        loss += criterion(output.view(BATCH_SIZE, -1), target[:,c])

    loss.backward()
    decoder_optimizer.step()

    return loss.data / CHUNK_LEN

def save():
    save_filename = os.path.splitext(os.path.basename(FILE_NAME))[0] + '.pt'
    torch.save(decoder, save_filename)
    print('Saved as %s' % save_filename)

# Initialize models and start training

if LOAD is not None:
    decoder.load(LOAD)
else:
    decoder = CharLSTM(
        n_characters,
        HIDDEN_SIZE,
        n_layers=N_LAYERS
    )

decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

if use_cuda:
    decoder.cuda()

start = time.time()
all_losses = []
loss_avg = 0
perplexity = []

try:
    print("Training for %d epochs..." % EPOCHS)
    for epoch in tqdm(range(1, EPOCHS + 1)):
        loss = train(*random_training_set(CHUNK_LEN, BATCH_SIZE))
        loss_avg += loss
        perplexity.append(2**loss)

        if epoch % 100 == 0:
            print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / EPOCHS * 100, loss))
            print(generate(decoder, 'Wh', 100, cuda=use_cuda), '\n')

    print("Saving...")
    save()

except KeyboardInterrupt:
    print("Saving before quit...")
    save()



def randomString(stringLength=10):
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for i in range(stringLength))

print(generate(decoder, randomString(6), 100, cuda=use_cuda), '\n')
print(generate(decoder, randomString(6), 100, cuda=use_cuda), '\n')
print(generate(decoder, randomString(6), 100, cuda=use_cuda), '\n')
print(generate(decoder, 'The', 100, cuda=use_cuda), '\n')
print(generate(decoder, 'What is', 100, cuda=use_cuda), '\n')
print(generate(decoder, 'Shall I give', 100, cuda=use_cuda), '\n')
print(generate(decoder, 'X087hNYB BHN BYFVuhsdbs', 100, cuda=use_cuda), '\n')

plt.plot(perplexity)
plt.show()
plt.savefig('image.png')