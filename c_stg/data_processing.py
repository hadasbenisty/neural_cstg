import torch
import torch.utils.data as data_utils
from c_stg.utils import norm_minmax


class DataContainer:
    def __init__(self, params, data, fold):

        # train set
        self.xtr = data.explan_feat[data.traininds[fold], :]
        self.ytr = data.output_label[data.traininds[fold]]
        self.rtr = data.context_feat[data.traininds[fold]]
        self.rtr = 2 * (norm_minmax(self.rtr.reshape(-1, 1)) - 0.5)
        # test set
        self.xte = data.explan_feat[data.testinds[fold], :]
        self.yte = data.output_label[data.testinds[fold]]
        self.rte = data.context_feat[data.testinds[fold]]
        self.rte = 2 * (norm_minmax(self.rte.reshape(-1, 1)) - 0.5)
        # develop set
        if not params.post_process_mode:  # finding hyperparameters
            self.xdev = data.explan_feat[data.devinds[fold], :]
            self.ydev = data.output_label[data.devinds[fold]]
            self.rdev = data.context_feat[data.devinds[fold]]
            self.rdev = 2 * (norm_minmax(self.rdev.reshape(-1, 1)) - 0.5)

        # train
        xtr = self.xtr
        ytr = self.ytr[:, None] if len(self.ytr.shape) == 1 else self.ytr  # one hot
        rtr = self.rtr[:, None] if len(self.rtr.shape) == 1 else self.rtr
        # test
        xtest = self.xte
        ytest = self.yte[:, None] if len(self.yte.shape) == 1 else self.yte
        rtest = self.rte[:, None] if len(self.rte.shape) == 1 else self.rte
        # develop
        if not params.post_process_mode:  # finding hyperparameters
            xdev = self.xdev
            ydev = self.ydev[:, None] if len(self.ydev.shape) == 1 else self.ydev
            rdev = self.rdev[:, None] if len(self.rdev.shape) == 1 else self.rdev

            ytest = torch.empty_like(torch.tensor(rtest))

        # Datasets
        train_set = data_utils.TensorDataset(torch.tensor(xtr), torch.tensor(ytr), torch.tensor(rtr))
        if not params.post_process_mode:  # finding hyperparameters
            dev_set = data_utils.TensorDataset(torch.tensor(xdev), torch.tensor(ydev), torch.tensor(rdev))
            test_set = data_utils.TensorDataset(torch.tensor(xtest), ytest, torch.tensor(rtest))
        else:
            test_set = data_utils.TensorDataset(torch.tensor(xtest), torch.tensor(ytest), torch.tensor(rtest))

        # Dataloaders
        self.train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=params.batch_size, shuffle=True)
        self.test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=params.batch_size, shuffle=False)
        if not params.post_process_mode:  # finding hyperparameters
            self.dev_dataloader = torch.utils.data.DataLoader(dev_set, batch_size=params.batch_size, shuffle=True)

    def get_Dataloaders(self, params):
        if not params.post_process_mode:  # finding hyperparameters
            return self.train_dataloader, self.dev_dataloader, self.test_dataloader
        else:
            return self.train_dataloader, self.test_dataloader