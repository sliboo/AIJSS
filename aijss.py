from __future__ import print_function, division
import argparse
import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.nn import Linear
from utils import load_data, load_graph
from GNN import GNNLayer
from evaluation import eva
from torch.autograd import Variable
import time
from datetime import datetime

tic = time.time()
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(10, args.hidden1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(args.hidden1, args.hidden2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(args.hidden2, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        validity = self.model(z)
        return validity


class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)

        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):
        r_e1 = F.relu(self.enc_1(x))
        r_e2 = F.relu(self.enc_2(r_e1))
        r_e3 = F.relu(self.enc_3(r_e2))
        r = self.z_layer(r_e3)

        r_d1 = F.relu(self.dec_1(r))
        r_d2 = F.relu(self.dec_2(r_d1))
        r_d3 = F.relu(self.dec_3(r_d2))
        x_bar = self.x_bar_layer(r_d3)

        return x_bar, r_e1, r_e2, r_e3, r


class AIJSS(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z, n_clusters, v=1):
        super(AIJSS, self).__init__()

        # autoencoder for intra information
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)
        self.ae.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))

        # GCN for inter information
        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)
        self.gnn_3 = GNNLayer(n_enc_1, n_enc_3)
        self.gnn_4 = GNNLayer(n_enc_3, n_z)
        self.gnn_5 = GNNLayer(n_z, n_z)

        self.gnn_6 = GNNLayer(n_z, n_dec_1)
        self.gnn_7 = GNNLayer(n_dec_1, n_dec_2)
        self.gnn_8 = GNNLayer(n_dec_2, n_dec_3)
        self.gnn_9 = GNNLayer(n_dec_3, n_input)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        self.s = nn.Sigmoid()

        # degree
        self.v = v

    def forward(self, x, adj):
        # content information learning module
        x_bar, r_e1, r_e2, r_e3, r = self.ae(x)

        # dual information interaction module
        z1 = self.gnn_1(x, adj)
        h2 = F.relu(self.ae.enc_2(z1 + r_e1))
        z2 = self.gnn_2(z1, adj)
        h3 = F.relu(self.ae.enc_3(h2))
        z3 = self.gnn_3(z2 + h2, adj)
        z = self.gnn_4(z3, adj)

        a_r = self.gnn_5(z + r, adj)
        z_i = z + r
        z_l = torch.spmm(adj, z_i)

        # decoder
        dec_z1 = self.gnn_6(z, adj, active=True)
        dec_z2 = self.gnn_7(dec_z1, adj, active=True)
        dec_z3 = self.gnn_8(dec_z2, adj, active=True)
        z_hat = self.gnn_9(dec_z3, adj, active=True)

        # adj_hat = self.s(torch.mm(z, z.t()))
        adj_hat = self.s(torch.mm(z_hat, z_hat.t()))

        # Joint-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(z_l.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        q1 = 1.0 / (1.0 + torch.sum(torch.pow(r.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q1 = q1.pow((self.v + 1.0) / 2.0)
        q1 = (q1.t() / torch.sum(q1, 1)).t()

        return x_bar, z_hat, adj_hat, q, q1, a_r, z, r, z_l


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def num_net_parameter(net):
    all_num = sum(i.numel() for i in net.parameters())
    print ('[The network parameters]', all_num)

def train_AIJSS(dataset):
    model = AIJSS(500, 500, 2000, 2000, 500, 500,
                  n_input=args.n_input,
                  n_z=args.n_z,
                  n_clusters=args.n_clusters,
                  v=1.0).to(device)

    print(model)
    print("alpha_1: ", args.alpha1, " alpha_2: ", args.alpha2, " alpha_3: ", args.alpha3,
          " alpha_4: ", args.alpha4, " alpha_5: ", args.alpha5, " alpha_6: ", args.alpha6)

    print(num_net_parameter(model))

    acc_reuslt_q1 = []
    nmi_result_q1 = []
    ari_result_q1 = []
    f1_result_q1 = []

    optimizer = Adam(model.parameters(), lr=args.lr)

    # KNN Graph
    adj = load_graph(args.name, args.k)
    adj = adj.cuda()

    adversarial_loss = torch.nn.BCELoss()

    # cluster parameter initiate
    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y

    with torch.no_grad():
        _, _, _, _, z = model.ae(data)

    discriminator = Discriminator()
    discriminator = discriminator.cuda()
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.adlr, betas=(0.5, 0.999))

    Tensor = torch.cuda.FloatTensor

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    eva(y, y_pred, 'pae')

    for epoch in range(args.epo):

        valid = Variable(Tensor(adj.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(adj.shape[0], 1).fill_(0.0), requires_grad=False)

        if epoch % 1 == 0:
            # update_interval
            _, _, _, q, q1, _, _, _, _ = model(data, adj)
            tmp_q = q.data
            p = target_distribution(tmp_q)
            tmp_q1 = q1.data
            p1 = target_distribution(tmp_q1)

            res = tmp_q1.cpu().numpy().argmax(1)  # Q1


            acc, nmi, ari, f1 = eva(y, res, str(epoch) + 'Q1')
            acc_reuslt_q1.append(acc)
            nmi_result_q1.append(nmi)
            ari_result_q1.append(ari)
            f1_result_q1.append(f1)


        x_bar, z_hat, adj_hat, q, q1, a_r, z, r, z_l = model(data, adj)

        ad_loss = 0.5 * adversarial_loss(discriminator(a_r), valid)
        ae_loss = F.mse_loss(x_bar, data)
        w_loss = F.mse_loss(z_hat, torch.spmm(adj, data))
        a_loss = F.mse_loss(adj_hat, adj.to_dense())
        qp_loss = F.kl_div(q.log(), p, reduction='batchmean')
        q1q_loss = F.kl_div(q1.log(), q, reduction='batchmean')
        q1p1_loss = F.kl_div(q1.log(), p1, reduction='batchmean')

        loss = ae_loss + args.alpha1 * w_loss + args.alpha2 * a_loss \
            + args.alpha3 * qp_loss + args.alpha4 * q1p1_loss + args.alpha5 * q1q_loss \
            + args.alpha6 * ad_loss


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Sample noise as discriminator ground truthR
        z = Variable(Tensor(np.random.normal(0, 1, (adj.shape[0], 10))))

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(z), valid)
        fake_loss = adversarial_loss(discriminator(a_r.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)

        d_loss.backward()
        optimizer_D.step()

    print("#################" + args.name + "####################")
    acc_max = np.max(acc_reuslt_q1)
    nmi_max = nmi_result_q1[acc_reuslt_q1.index(np.max(acc_reuslt_q1))]
    ari_max = ari_result_q1[acc_reuslt_q1.index(np.max(acc_reuslt_q1))]
    f1_max = f1_result_q1[acc_reuslt_q1.index(np.max(acc_reuslt_q1))]
    epoch_max =acc_reuslt_q1.index(np.max(acc_reuslt_q1))
    print('the result of this iter:\nacc:{:.4f},\nnmi:{:.4f},\nari:{:.4f},\nf1:{:.4f},\nmax_epoch:{}\n'.format(round(acc_max, 5),
                                                                        round(nmi_max, 5),
                                                                        round(ari_max, 5),
                                                                        round(f1_max, 5),
                                                                        epoch_max))
    return acc_max, nmi_max, ari_max, f1_max

if __name__ == "__main__":

    iters = 5  #

    acc = []
    nmi = []
    ari = []
    f1 = []

    for iter_num in range(iters):
        print('the {}-th iter'.format(iter_num + 1))
        parser = argparse.ArgumentParser(
            description='train',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--name', type=str, default='hhar')
        parser.add_argument('--k', type=int, default=3)
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--n_clusters', default=3, type=int)
        parser.add_argument('--n_z', default=10, type=int)
        parser.add_argument('--pretrain_path', type=str, default='pkl')
        parser.add_argument('--adlr', type=float, default=1e-4)
        parser.add_argument('--hidden1', type=int, default=64)
        parser.add_argument('--hidden2', type=int, default=16)
        parser.add_argument('--alpha1', type=float, default=1)  # The parameters of w_loss
        parser.add_argument('--alpha2', type=float, default=1)  # The parameters of ae_loss
        parser.add_argument('--alpha3', type=float, default=0.1)  # The parameters of qp_loss
        parser.add_argument('--alpha4', type=float, default=0.1)  # The parameters of q1p1_loss
        parser.add_argument('--alpha5', type=float, default=1)  # The parameters of q1q_loss
        parser.add_argument('--alpha6', type=float, default=0.1)  # The parameters of ad_loss
        parser.add_argument('--epo', type=int, default=400)
        parser.add_argument('--save_dir', type=str, default='tsne')
        args = parser.parse_args()
        args.cuda = torch.cuda.is_available()
        print("use cuda: {}".format(args.cuda))
        device = torch.device("cuda" if args.cuda else "cpu")

        args.pretrain_path = 'data/{}.pkl'.format(args.name)
        dataset = load_data(args.name)

        if args.name == 'usps':
            args.n_clusters = 10
            args.n_input = 256
            args.alpha1 = 10
            args.alpha2 = 0.1
            args.alpha3 = 10
            args.alpha4 = 0.01
            args.alpha5 = 0.01
            args.alpha6 = 0.01

        if args.name == 'hhar':
            args.k = 5
            args.n_clusters = 6
            args.n_input = 561
            args.alpha1 = 10
            args.alpha2 = 0.01
            args.alpha4 = 1

        if args.name == 'reut':
            args.lr = 1e-4
            args.n_clusters = 4
            args.n_input = 2000
            args.k = 10
            args.alpha2 = 0.1
            args.alpha3 = 1
            args.alpha4 = 10
            args.alpha6 = 1
            args.epo = 100

        if args.name == 'acm':
            args.k = None
            args.lr = 5e-5
            args.n_clusters = 3
            args.n_input = 1870
            args.alpha5 = 0.1

        if args.name == 'dblp':
            args.k = None
            args.lr = 2e-3
            args.n_clusters = 4
            args.n_input = 334
            args.alpha1 = 10
            args.alpha6 = 0.01
            args.epo = 200

        if args.name == 'cite':
            args.lr = 1e-4
            args.k = None
            args.n_clusters = 6
            args.n_input = 3703
            args.alpha3 = 1
            args.epo = 150

        print(args)
        acc_max, nmi_max, ari_max, f1_max = train_AIJSS(dataset)
        acc.append(acc_max)
        nmi.append(nmi_max)
        ari.append(ari_max)
        f1.append(f1_max)
    print('MEAN:acc, nmi, ari, f1: \n{:.4f}\n{:.4f}\n{:.4f}\n{:.4f}\n'.format(np.mean(acc), np.mean(nmi), np.mean(ari),
                                                                           np.mean(f1)))
    print('MAX:acc, nmi, ari, f1: \n{:.4f}\n{:.4f}\n{:.4f}\n{:.4f}\n'.format(np.max(acc), np.max(nmi), np.max(ari),
                                                                           np.max(f1)))

    toc = time.time()
    print("Time:", (toc - tic))
