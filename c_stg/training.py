import numpy as np
import torch
import scipy.io
import torch.nn as nn
from flavors.utils import acc_score


# Training
def train(params, model, train_dataloader, dev_dataloader, criterion, optimizer, stg_regularizer, final_test=False):

    acc_train_array = [0]
    loss_train_array = [0]
    acc_dev_array = [0]
    loss_dev_array = [0]

    for epoch in range(params.num_epoch):
        print(epoch)
        model.train()
        for batch, (input, target, B) in enumerate(train_dataloader):

            input = input.to(params.device).float()
            target = target.to(params.device).float()
            B = B.to(params.device).float()

            output = model(input, B)
            output = torch.squeeze(output)
            loss = criterion(output, torch.squeeze(target))

            if params.stg:
                stg_loss = torch.mean(model.reg(model.gates.mu/model.sigma))
                loss += stg_regularizer * stg_loss

            optimizer.zero_grad()
            with torch.autograd.set_detect_anomaly(True):
                loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
            optimizer.step()

        model.eval()

        acc_train, loss_train = test_process(params, model, train_dataloader, criterion, stg_regularizer)
        acc_train_array.append(acc_train * 100)
        loss_train_array.append(loss_train)
        print(f"acc train :{acc_train_array[-1]}")

        acc_dev, loss_dev = test_process(params, model, dev_dataloader, criterion, stg_regularizer)
        acc_dev_array.append(acc_dev * 100)
        loss_dev_array.append(loss_dev)
        print(f"acc dev :{acc_dev_array[-1]}")

        if not params.post_process_mode:
            model_path = "model.model"
            loss_path = "loss.mat"
            if acc_dev == np.max(acc_dev_array):  # We also save the model when the dev accuracy increases
                torch.save(model.state_dict(), model_path)
                print('Model saved! Validation accuracy improved from {:3f} to {:3f}'.format(np.max(acc_dev_array[:-1]),
                                                                                             np.max(acc_dev_array)))
                scipy.io.savemat(loss_path, {'train_loss_array': loss_train_array, 'dev_loss_array': loss_dev_array,
                                             'train_acc': acc_train_array, 'dev_acc': acc_dev_array})

    return acc_train_array, loss_train_array, acc_dev_array, loss_dev_array


def test_process(params, model, test_dataloader, criterion, stg_regularizer):

    y_pred = None
    all_targets = None
    train_loss = 0
    train_count = 0

    for batch, (input, target, B) in enumerate(test_dataloader):
        input = input.to(params.device).float()
        B = B.to(params.device).float()

        with torch.no_grad():
            output = model(input, B)
        output = torch.squeeze(output)
        y_pred_cur = output.float().detach().cpu().numpy().reshape(-1, 1)
        if y_pred is None:
            y_pred = y_pred_cur
            all_targets = target
        else:
            y_pred = np.vstack((y_pred, y_pred_cur))
            all_targets = np.vstack((all_targets, target))

        target = target.to(params.device).float()
        loss = criterion(output, torch.squeeze(target))
        loss += stg_regularizer * torch.mean(model.reg(model.gates.mu/model.sigma))
        train_loss += loss.item() * len(input)
        train_count += len(input)

    acc = acc_score(all_targets, y_pred)
    loss = train_loss / train_count
    return acc, loss


def get_prob_alpha(params, model, r):
    if len(r.shape) == 1:
        r = r[:, None]
    B = torch.tensor(r)
    B = B.to(params.device).float()

    model.eval()
    mu = model.gates.get_feature_importance(B)
    mu = mu.detach().cpu().numpy()

    return mu