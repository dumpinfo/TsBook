import numpy as np
import tensorflow as tf

class Rnn_Engine(object):
    def __init__(self, params):
        self.data = open('datasets/input.txt', 'r', encoding='utf8').read()
        self.chars = list(set(self.data))
        self.data_size, self.vocab_size = len(self.data), len(self.chars)
        print('d={0}; v={1}'.format(self.data_size, self.vocab_size))
        self.char_to_ix = { ch:i for i,ch in enumerate(self.chars) }
        self.ix_to_char = { i:ch for i, ch in enumerate(self.chars) }
        self.hidden_size = 128
        self.seq_length = 25
        self.learning_rate = 1e-1
        
    def train(self):
        self.build_model()
        n, p = 0, 0
        while True:
            if p+self.seq_length+1 >= len(self.data) or n == 0:
                hprev = np.zeros((self.hidden_size, 1))
                p = 0
            inputs = [self.char_to_ix[ch] for ch in self.data[p:p+self.seq_length]]
            targets = [self.char_to_ix[ch] for ch in self.data[p+1:p+self.seq_length+1]]
            if n % 100 == 0:
                sample_ix = self.sample(hprev, inputs[0], 200)
                txt = ''.join(self.ix_to_char[ix] for ix in sample_ix)
                print('-----\n {0} \n-----'.format(txt))
            loss, dWxh, dWhh, dWhy, dbh, dby, hprev = self.loss_func(inputs, targets, \
                                                                hprev)
            self.smooth_loss = self.smooth_loss * 0.999 + loss *0.001
            if n % 100 == 0:
                print('iter {0}, loss: {1}'.format(n, self.smooth_loss))
            for param, dparam, mem in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by], \
                                            [dWxh, dWhh, dWhy, dbh, dby], \
                                            [self.mWxh, self.mWhh, self.mWhy, self.mbh, self.mby]):
                mem += dparam * dparam
                param += -self.learning_rate * dparam / np.sqrt(mem + 1e-8)
            p += self.seq_length
            n += 1
                
        
        
        
    def build_model(self):
        self.Wxh = np.random.randn(self.hidden_size, self.vocab_size)*0.01
        self.Whh = np.random.randn(self.hidden_size, self.hidden_size)*0.01
        self.Why = np.random.randn(self.vocab_size, self.hidden_size)*0.01
        self.bh = np.zeros((self.hidden_size, 1))
        self.by = np.zeros((self.vocab_size, 1))
        self.mWxh, self.mWhh, self.mWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), \
                    np.zeros_like(self.Why)
        self.mbh, self.mby = np.zeros_like(self.bh), np.zeros_like(self.by)
        self.smooth_loss = -np.log(1.0/self.vocab_size)*self.seq_length
        
    def loss_func(self, inputs, targets, hprev):
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(hprev)
        loss = 0
        for t in range(len(inputs)):
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][inputs[t]] = 1
            hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh)
            ys[t] = np.dot(self.Why, hs[t]) + self.by
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
            loss += -np.log(ps[t][targets[t], 0])
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), \
                            np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dhnext = np.zeros_like(hs[0])
        for t in reversed(range(len(inputs))):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1
            dWhy += np.dot(dy, hs[t].T)
            dby += dy
            dh = np.dot(self.Why.T, dy) + dhnext
            dhraw = (1 -hs[t] * hs[t])*dh
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t-1].T)
            dhnext = np.dot(self.Whh.T, dhraw)
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)
        return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]
        
    def sample(self, h, seed_ix, n):
        x = np.zeros((self.vocab_size, 1))
        x[seed_ix] = 1
        ixes = []
        for t in range(n):
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            y = np.dot(self.Why, h) + self.by
            p = np.exp(y) / np.sum(np.exp(y))
            ix = np.random.choice(range(self.vocab_size), p=p.ravel())
            x = np.zeros((self.vocab_size, 1))
            x[ix] = 1
            ixes.append(ix)
        return ixes
        