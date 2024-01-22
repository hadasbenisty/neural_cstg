import numpy as np
import torch
import scipy.io
import torch.nn as nn


# Training
def train(params, model, train_dataloader, dev_dataloader, criterion, optimizer, stg_regularizer, acc_score):

    acc_train_array = [0]
    loss_train_array = [0]
    acc_dev_array = [0]
    loss_dev_array = [0]

    # to ensure effective learning
    same_acc_count = 0
    uneffective_learn = False

    for epoch in range(params.num_epoch):
        print(epoch)
        model.train()
        for batch, (input, target, B) in enumerate(train_dataloader):

            input = input.to(params.device).float()
            if params.output_dim > 2 and params.classification_flag:  # for cross entropy pytorch format
                target = target.to(params.device).long()
            else:
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

        acc_train, loss_train, _, _ = test_process(params, model, train_dataloader, criterion, stg_regularizer, acc_score)
        if acc_train == acc_train_array[-1]/100:
            same_acc_count += 1
            if same_acc_count == 40:
                uneffective_learn = True
                return acc_train_array, loss_train_array, acc_dev_array, loss_dev_array, uneffective_learn
        else:
            same_acc_count = 0
        acc_train_array.append(acc_train * 100)
        loss_train_array.append(loss_train)
        print(f"acc train :{acc_train_array[-1]}")

        acc_dev, loss_dev, _, _ = test_process(params, model, dev_dataloader, criterion, stg_regularizer, acc_score)
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

    return acc_train_array, loss_train_array, acc_dev_array, loss_dev_array, uneffective_learn


def test_process(params, model, test_dataloader, criterion, stg_regularizer, acc_score):

    y_pred = None
    all_targets = None
    train_loss = 0
    train_count = 0

    for batch, (input, target, B) in enumerate(test_dataloader):
        input = input.to(params.device).float()
        B = B.to(params.device).float()

        with torch.no_grad():
            output = model(input, B)

        if params.output_dim > 2 and params.classification_flag:  # for cross entropy pytorch format
            target = target.to(params.device).long()
            y_pred_cur = output.float()
            _, labels_pred_cur = torch.max(y_pred_cur, 1)

            if y_pred is None:
                y_pred = y_pred_cur
                all_targets = target
                labels_pred = labels_pred_cur
            else:
                y_pred = torch.concatenate((y_pred, y_pred_cur))
                all_targets = torch.concatenate((all_targets, target))
                labels_pred = torch.concatenate((labels_pred, labels_pred_cur))

        else:
            output = torch.squeeze(output)
            y_pred_cur = output.float().detach().cpu().numpy().reshape(-1, 1)
            labels_pred_cur = np.argmax(y_pred_cur, 1)[:, None]

            if y_pred is None:
                y_pred = y_pred_cur
                all_targets = target
                labels_pred = labels_pred_cur
            else:
                y_pred = np.vstack((y_pred, y_pred_cur))
                all_targets = np.vstack((all_targets, target))
                labels_pred = np.vstack((labels_pred, labels_pred_cur))

            target = target.to(params.device).float()

        loss = criterion(output, torch.squeeze(target))
        loss += stg_regularizer * torch.mean(model.reg(model.gates.mu/model.sigma))
        train_loss += loss.item() * len(input)
        train_count += len(input)

    if params.output_dim == 1:
        labels_pred = torch.tensor(labels_pred.flatten())

    acc = acc_score(all_targets, y_pred, params)
    loss = train_loss / train_count

    return acc, loss, all_targets, labels_pred


def get_prob_alpha(params, model, r):
    if len(r.shape) == 1:
        r = r[:, None]
    B = torch.tensor(r)
    B = B.to(params.device).float()

    model.eval()
    # mu = model.gates.get_feature_importance(B)
    # mu = mu.detach().cpu().numpy()
    if params.ML_model_name == "fc_stg_layered_param_modular_model_sigmoid":
        mu = model.gates.get_feature_importance(B)
        mu = mu.detach().cpu().numpy()
        return mu
    elif params.ML_model_name == "fc_stg_layered_param_modular_model_sigmoid_extension":
        mu, w = model.gates.get_feature_importance(B)
        mu = mu.detach().cpu().numpy()
        w = w.detach().cpu().numpy()
        return mu, w
    else:
        RuntimeError("No suitable option")