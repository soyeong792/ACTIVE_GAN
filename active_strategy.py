import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, Dataset

class Strategy:
    def __init__(self, data_loader, idxs_lb, data_type, net, optim, n_epoch):
        #self.loader_tr = data_loader

        if data_type == 'Synthetic':
            self.n_epoch = 1
        self.num_workers  = data_loader.num_workers
        self.batch_size = data_loader.batch_size
        self.optimizer = optim
        self.X = data_loader.dataset.tensors[0]
        self.Y = data_loader.dataset.tensors[1]
        self.idxs_lb = idxs_lb
        self.net = net
        self.n_epoch = n_epoch
        #self.handler = handler
        #self.args = args
        self.n_pool = len(self.Y)
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

    def query(self, n):
        pass

    def update(self, idxs_lb,loader_tr):
        self.loader_tr = loader_tr
        self.idxs_lb = idxs_lb

    def _train(self, epoch, loader_tr, optimizer):
        self.clf.train()
        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            x = x.to(self.device)
            y = y.to(self.device)
            optimizer.zero_grad()
            out = self.clf(x.cuda().float())  # e1
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()

    def train(self):
        #n_epoch = self.args['n_epoch']d
        self.clf = self.net.to(self.device)
        optimizer = self.optimizer #optim.SGD(self.clf.parameters(), **self.args['optimizer_args'])

        idxs_train = np.arange(self.n_pool)[self.idxs_lb]

        #print(self.X[idxs_train])

        loader_tr = DataLoader(TensorDataset(self.X[idxs_train], self.Y[idxs_train],torch.Tensor(idxs_train)), batch_size=self.batch_size,
                            shuffle=True, num_workers=self.num_workers)

        #torch.utils.data.DataLoader(tr_dataset, batch_size = batch_size, shuffle=True, num_workers = n_workers)    


        for epoch in range(1, self.n_epoch+1):
            self._train(epoch, loader_tr, optimizer)

    def predict(self, X, Y):
        loader_te = DataLoader(TensorDataset(X, Y, torch.Tensor(np.arange(len(Y)))), batch_size=self.batch_size,
                            shuffle=False, num_workers=self.num_workers)

        self.clf.eval()
        P = torch.zeros(len(Y), dtype=Y.dtype)
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out = self.clf(x.cuda().float()) #, e1

                pred = out.max(1)[1]
                #print(pred,idxs)
                P[idxs.type(torch.LongTensor)] = pred.cpu()

        return P

    
    def predict_prob(self, X, Y, num_classes):
        loader_te = DataLoader(TensorDataset(X, Y, torch.Tensor(np.arange(len(Y)))), batch_size=self.batch_size,
                            shuffle=False, num_workers=self.num_workers)

        self.clf.eval()
        probs = torch.zeros([len(Y), num_classes]) 
        # [NUM_QUERY,2] #len(np.unique(Y)) #circleě labelě´ [0,1]
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out = self.clf(x.cuda().float())
                prob = F.softmax(out, dim=1)
                probs[idxs.type(torch.LongTensor)] = prob.cpu()
        
        return probs

    def predict_prob_dropout(self, X, Y, n_drop):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform']),
                            shuffle=False, **self.args['loader_te_args'])

        self.clf.train()
        probs = torch.zeros([len(Y), len(np.unique(Y))])
        for i in range(n_drop):
            #print('n_drop {}/{}'.format(i+1, n_drop))
            with torch.no_grad():
                for x, y, idxs in loader_te:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs[idxs] += prob.cpu()
        probs /= n_drop
        
        return probs

    def predict_prob_dropout_split(self, X, Y, n_drop,num_classes):
        loader_te = DataLoader(TensorDataset(X, Y, torch.Tensor(np.arange(len(Y)))), batch_size=self.batch_size,
                            shuffle=False, num_workers=self.num_workers)

        self.clf.train()
        probs = torch.zeros([n_drop, len(Y), num_classes])
        for i in range(n_drop):
            #print('n_drop {}/{}'.format(i+1, n_drop))
            with torch.no_grad():
                for x, y, idxs in loader_te:
                    x, y = x.to(self.device), y.to(self.device)
                    out = self.clf(x.cuda().float())
                    probs[i][idxs.type(torch.LongTensor)] += F.softmax(out, dim=1).cpu()
        
        return probs

    def get_embedding(self, X, Y):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform']),
                            shuffle=False, **self.args['loader_te_args'])

        self.clf.eval()
        embedding = torch.zeros([len(Y), self.clf.get_embedding_dim()])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                embedding[idxs] = e1.cpu()
        
        return embedding

