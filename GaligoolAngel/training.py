import torch.optim as optim
import torch.nn as nn
import torch
from sklearn.model_selection import KFold
from variables import l1_lambda, l2_lambda

# Assuming the model is already defined as 'model'
# Define Loss Function
# For regression tasks, you can use nn.MSELoss()
# For classification tasks, you can use nn.CrossEntropyLoss() (if your output is class scores)
# Adjust according to your specific task
def training(model, features, y):
    loss_function = nn.MSELoss()

    # Choose an optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    epochs = 20  # Number of epochs
    for epoch in range(epochs):
        tot_loss = 0
        for sample in range(len(y)):
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(features[sample])

            # Compute Loss
            loss = loss_function(outputs, y[sample])
            tot_loss = tot_loss + loss.item()
            # Backward pass and optimize
            loss.backward()
            optimizer.step()


        print(f'Epoch {epoch + 1}, Loss: {tot_loss}')
    return model


def training_k_fold(model, features, y, n_splits=3):
    loss_function = nn.MSELoss()

    # K-Fold Cross-validation
    kf = KFold(n_splits=n_splits)

    for fold, (train_index, val_index) in enumerate(kf.split(features)):
        print(f"Fold {fold + 1}")

        # Split data into training and validation
        train_features, val_features = features[train_index], features[val_index]
        train_y, val_y = y[train_index], y[val_index]

        # Choose an optimizer with L2 regularization (weight decay)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # optimizer.zero_grad()
        # outputs = model(features)
        # loss = loss_function(outputs, y)
        # loss.backward()
        # print(model.hypernetwork.weights.grad)
        # print(model.prediction_network.fc1.weight.grad)
        # print(model.prediction_network.fc2.weight.grad)

        # Number of epochs
        epochs = 10

        for epoch in range(epochs):
            tot_loss = 0

            # Training phase
            model.train()
            for i in range(len(train_y)):
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(train_features[i].unsqueeze(0))

                # Compute Loss
                loss = loss_function(outputs, train_y[i].unsqueeze(0))

                # L1 Regularization for the Hypernetwork
                l1_penalty = torch.tensor(0.).to(features.device)
                for param in model.hypernetwork.parameters():
                    l1_penalty += param.abs().sum()

                # L2 Regularization for the Prediction Network
                l2_penalty = torch.tensor(0.).to(features.device)
                for param in model.prediction_network.parameters():
                    l2_penalty += torch.norm(param, 2) ** 2

                # Combine loss with regularization terms
                loss += l1_lambda * l1_penalty + l2_lambda * l2_penalty
                tot_loss += loss.item()

                # Backward pass and optimize
                loss.backward()
                optimizer.step()
            tot_loss = tot_loss / len(train_y)
            # Validation phase
            model.eval()
            with torch.no_grad():
                val_loss = 0
                for i in range(len(val_y)):
                    outputs = model(val_features[i].unsqueeze(0))
                    val_loss += loss_function(outputs, val_y[i].unsqueeze(0)).item()
            val_loss = val_loss / len(val_y)
            print(f'Epoch {epoch + 1}, Training Loss: {tot_loss}, Validation Loss: {val_loss}')

    return model
