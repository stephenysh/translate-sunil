import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np





class CharCNN(nn.Module):
    def __init__(self, params, DEVICE):
        super(CharCNN, self).__init__()
        print('build character CNN')



        self.char_embed = nn.Embedding(params.N_char, params.dim_char)



        Cout = int(params.n_char_filters / 3)



        self.conv12 = nn.Conv2d(1, Cout, (2, params.dim_char),groups=1)
        self.conv13 = nn.Conv2d(1, Cout, (3, params.dim_char),groups=1)
        self.conv14 = nn.Conv2d(1, Cout, (4, params.dim_char),groups=1)


        self.pool = nn.AdaptiveMaxPool2d((1, 1))

        self.char_embed.weight.data.copy_(torch.from_numpy(self.random_embedding(params.N_char, params.dim_char)).to(DEVICE))
        self.char_embed.weight.requires_grad = True

        self.char_dropout = nn.Dropout(params.nn_dropout)



    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb



    def forward(self, X):  # (N,Lw,Lc)
        x0 = self.char_embed(X)  # (N,Lw,Lc,D)
        x0 = self.char_dropout(x0)  #(N,Lw,Lc,D)

        batch_size = x0.shape[0]
        Lw = x0.shape[1]
        Lc = x0.shape[2]
        D = x0.shape[3]

        #print(self.conv12.parameters())
        #print(list(self.conv12.parameters())[0].shape)


        x0 = x0.reshape(batch_size*Lw, 1, Lc,D)  #(N,Cin,Hin,Win)


        x1 = F.relu(self.conv12(x0))  #(N*Lw, Filters, Hout,1)
        x1 = self.pool(x1) #(N*Lw, Filters,1,1)
        x1 = x1.reshape(batch_size,Lw,-1)  # (N,Lw,25)  #TODO: is alignement after reshape?????

        x2 = F.relu(self.conv13(x0))
        x2 = self.pool(x2)
        x2 = x2.reshape(batch_size, Lw, -1)

        x3 = F.relu(self.conv14(x0))
        x3 = self.pool(x3)
        x3 = x3.reshape(batch_size, Lw, -1)

        Y = torch.cat((x1, x2, x3), 2)
        #Y [N, maxLw, 100]


        return Y




