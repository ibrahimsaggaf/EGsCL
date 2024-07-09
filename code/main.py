import argparse
import torch
from pathlib import Path

from utils import check_path
from data import AgeingData
from model import EGsCL


def main(args):
    torch.manual_seed(args.seed)
    check_path(args.res_path)

    file_X = Path(args.X)
    file_y = Path(args.y)
    if args.X2 is not None:
        file_X2 = Path(args.X2)
        data = AgeingData(args.train_size, args.cv, args.seed, file_X, file_y, args.delimiter, bool(args.header), 
                        file_X2=file_X2, delimiter2=args.delimiter2, header2=bool(args.header2))
        
    else:
        data = AgeingData(args.train_size, args.cv, args.seed, file_X, file_y, args.delimiter, bool(args.header))
    
    data.split()
    device = torch.device(f'cuda:{args.gpu}')

    if data.dim == 128:
        enc_kwargs = {
            'enc_in_dim': data.dim,
            'enc_dim': 69,
            'enc_out_dim': 64,
            'proj_dim': 48,
            'proj_out_dim': 32
        }
    else:
        enc_kwargs = {
            'enc_in_dim': data.dim,
            'enc_dim': 1024,
            'enc_out_dim': 512,
            'proj_dim': 256,
            'proj_out_dim': 128
        }

    model = EGsCL(
            data, args.val_metric, args.loss, args.epochs, args.batch_size, args.step, args.temperature, args.lr, 
            args.wd, device, args.res_path, enc_kwargs, args.beta
        )
    model.fit_cv()
    torch.save(model.cv_res, 
            Path(args.res_path, f'EGsCL--{args.loss}--{args.val_metric}--{args.beta}--cv_res.pt')
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--X', help='full path of X file')
    parser.add_argument('--y', help='full path of y file')
    parser.add_argument('--delimiter', help='delimiter')
    parser.add_argument('--header', type=int, help='if the file has a header and an index column')
    parser.add_argument('--X2', required=False, help='full path of X2 file')
    parser.add_argument('--delimiter2', required=False, help='delimiter2')
    parser.add_argument('--header2', required=False, type=int, help='if the file has a header and an index column')
    parser.add_argument('--beta', type=float, help='shift the Gaussian noise mean, possible values are 0.0, a positive float number, or fixed')
    parser.add_argument('--res_path', help='the path where the results are saved')
    parser.add_argument('--val_metric', default='mcc', help='the metric for model selection')
    parser.add_argument('--loss', default='SupCon', help='the contrastive loss, either SupCon or SimCLR')
    parser.add_argument('--train_size', type=float, default=0.8, help='the training set size when splitting the data')
    parser.add_argument('--cv', type=int, default=10, help='the number of cross validation folds')
    parser.add_argument('--batch_size', type=int, default=-1, help='the batch size. To use all training samples pass the value -1')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs for contrastive learning')
    parser.add_argument('--step', type=int, default=1, help='number of steps to measure validation performance for contrastive learning')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for contrastive learning')
    parser.add_argument('--wd', type=float, default=1e-6, help='weight decay')
    parser.add_argument('--temperature', type=float, default=0.1, help='temperature hyperparameter')
    parser.add_argument('--seed', type=int, default=1111, help='seed')
    parser.add_argument('--gpu', type=int, default=0, help='the GPU id')

    args = parser.parse_args()
    main(args)
