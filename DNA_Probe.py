import argparse
import logging
import time
import numpy
import scipy
import torch
import pandas
from utils import *
from sklearn.model_selection import train_test_split
from model import ProbeDataset
from model import AttentionLSTM as ProbeEfficiencyModel
from torch.utils.data import DataLoader
from scipy import stats
from tqdm import tqdm
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(description='DNA Probe Model',
                                     usage='DNA_Probe.py subcommand ['
                                           'options]')
    subparsers = parser.add_subparsers(dest='subcommand', required=True,
                                       # description='train, predict',
                                       title='available subcommands', help='')
    # train parser
    train_parser = subparsers.add_parser('train', help='train a new model')
    train_parser.add_argument('-input', type=str, required=True,
                              help='input data')
    train_parser.add_argument('-output', type=str, required=True,
                              help='model output path')
    train_parser.add_argument('-gpu', type=str, default='',
                              help='gpu id to use, leave it empty to use CPU')
    train_parser.add_argument('-kmer', type=int, default=0, help='K-mer size')
    train_parser.add_argument('-onehot',
                              help='use onehot encoding for DNA base',
                              action='store_true', default=True)
    train_parser.add_argument('-use_struct', action='store_true', default=False,
                              help='use structure information in the model')
    train_parser.add_argument('-embed_dim', help='embedding size', type=int,
                              default=4)
    train_parser.add_argument('-epochs', help='epochs for training', type=int,
                              default=60)
    train_parser.add_argument('-batch_size', help='training batch size',
                              type=int, default=64)
    train_parser.add_argument('-lr', help='learning rate', type=float,
                              default=1e-4)

    # predict parser
    predict_parser = subparsers.add_parser('predict', help='predict based on '
                                                           'existing model')
    predict_parser.add_argument('-input', type=str, required=True,
                                help='input data')
    predict_parser.add_argument('-model', type=str, required=True,
                                help='DNA Probe model')
    predict_parser.add_argument('-output', type=str, required=True,
                                help='prediction output path')
    predict_parser.add_argument('-gpu', type=str, default='',
                                help='gpu id to use, leave it empty to use CPU')
    predict_parser.add_argument('-batch_size', help='prediction batch size',
                                type=int, default=256)

    _args = parser.parse_args()

    if _args.gpu:
        if torch.cuda.is_available():
            _args.device = torch.device('cuda:' + _args.gpu)
        else:
            logging.warning('GPU {} is NOT available, '
                            'use GPU instead!'.format(_args.gpu))
            _args.device = torch.device('cpu')
    else:
        _args.device = torch.device('cpu')

    if _args.subcommand == 'train':
        assert _args.kmer <= 10
        if _args.use_struct:
            _args.onehot = True
        if _args.onehot:
            _args.kmer = 1
        _args.kmer_dict = build_kmers_dict(_args.kmer)

        # model parameters
        _args.fc_in_dim = 32
        _args.dropout = 0.2
        _args.lstm_hidden_size = 64
        _args.lstm_layers = 2
    else:
        print('pred')
    return _args


def load_data(args):
    dataset = pandas.read_csv(args.input, sep="\t", header=None)
    # if args.use_struct:
    # if dataset.shape[1] != 3:
    #     print("Please provide the data as required!")
    #     exit(1)
    # print("Use Structure information to train")
    # print("Use onehot encoding")

    if args.subcommand == 'train':
        if args.use_struct:
            assert dataset.shape[1] >= 3
        else:
            assert dataset.shape[1] >= 2
        print("Dataset size: {}, use 10% of data as "
              "validation set".format(dataset.shape))
        # split data
        train_idx, val_idx = train_test_split(range(len(dataset)),
                                              test_size=0.1, random_state=1)
        train_set = ProbeDataset(args, dataset.iloc[train_idx,])
        val_set = ProbeDataset(args, dataset.iloc[val_idx,])
        _train_loader = DataLoader(train_set, batch_size=args.batch_size,
                                   shuffle=True, num_workers=4, pin_memory=True,
                                   drop_last=True)
        _val_loader = DataLoader(val_set, batch_size=args.batch_size,
                                 shuffle=True, num_workers=4, pin_memory=True,
                                 drop_last=True)
        return [_train_loader, _val_loader]
    else:
        if args.use_struct:
            assert dataset.shape[1] >= 2
            dataset = dataset.iloc[:, 0:2]
        else:
            assert dataset.shape[1] >= 1
            dataset = dataset.iloc[:, 0:1]
        # append a placeholder column
        dataset['target'] = 0
        return DataLoader(ProbeDataset(args, dataset),
                          batch_size=args.batch_size, num_workers=4,
                          pin_memory=True)


def train(args):
    def train_one_epoch(model, dataloader, device):
        model.train()
        pbar = tqdm(total=len(dataloader.dataset), ncols=80, unit='seqs',
                    desc='training...', ascii=' >=', leave=False)
        total_loss = []
        for idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            pred = model(data)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            total_loss.append(loss.detach().item())
            pbar.set_description("training loss: {:8.4f}".format(loss.item()))
            pbar.update(dataloader.batch_size)
        pbar.close()
        with torch.no_grad():
            print('| epoch {:3d} |  train loss {:8.4f}'.format(epoch + 1,
                                                               numpy.mean(
                                                                   total_loss)))

    train_loader, val_loader = load_data(args)
    args.seq_len = train_loader.dataset[0][0].shape[0]
    model = ProbeEfficiencyModel(args).to(args.device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    min_loss = 9999
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        train_one_epoch(model, train_loader, args.device)
        val_loss, y_pred, y_true = evaluate(model, val_loader, args.device)
        val_corr = scipy.stats.pearsonr(y_pred, y_true)[0]

        print('| epoch {:3d} |  val   loss {:8.4f} | val   corr {:8.4f}'.format(
            epoch + 1, val_loss, val_corr))
        # save model
        if val_loss < min_loss:
            min_loss = val_loss
            torch.save({'model_state_dict': model.state_dict(),
                        'train_args': args.__dict__}, args.output)
            print("New model saved to {}".format(args.output))
        print('| end of epoch {:3d} | time: {:6.3f}s '.format(epoch + 1,
                                                              time.time() - epoch_start_time))
        print('-' * 80)


def evaluate(model, dataloader, device):
    model.eval()
    pbar = tqdm(total=len(dataloader.dataset), ncols=80, desc="validating...",
                unit='seqs', ascii=' >=', leave=False)
    y_true = []
    y_pred = []
    total_loss = []
    with torch.no_grad():
        for idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            pred = model(data)
            loss = F.mse_loss(pred, target).item()
            total_loss.append(loss)
            y_pred.extend(pred)
            y_true.extend(target.data)
            pbar.update(dataloader.batch_size)
    pbar.close()
    return numpy.mean(total_loss), torch.stack(
        y_pred).cpu().numpy(), torch.stack(y_true).cpu().numpy()


def predict(args):
    checkpoint = torch.load(args.model, map_location=torch.device('cpu'))
    train_args = checkpoint['train_args']
    for k, v in train_args.items():
        if k not in args:
            setattr(args, k, v)

    # check if model and data are compatible
    dataloader = load_data(args)
    model = ProbeEfficiencyModel(args).to(args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    loss, y_pred, y_true = evaluate(model, dataloader, args.device)
    pandas.DataFrame(y_pred).to_csv(args.output, header=False, index=False)
    print("Prediction results saved to {}".format(args.output))


if __name__ == '__main__':
    args = parse_args()
    if args.subcommand == 'train':
        print('training config:', args.__dict__)
        train(args)
    elif args.subcommand == 'predict':
        predict(args)
    else:
        print('please specify the subcommand, train or predict')
