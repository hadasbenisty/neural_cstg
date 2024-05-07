# # import pandas as pd
# # import torch
# # from imblearn.over_sampling import SMOTE
# # from sklearn.model_selection import train_test_split
# # from sklearn.preprocessing import StandardScaler
# # import torch.nn as nn
# # import torch.optim as optim
# # import torch.nn.functional as F
# # from torch.utils.data import DataLoader, Dataset
# # from torchsummary import summary
# # import itertools
# # import numpy as np
# # from sklearn.utils.class_weight import compute_class_weight
# # from sklearn.model_selection import KFold
# #
# #
# # def get_best_model_parameters(results_df):
# #     best_parameters = {}
# #
# #     # Group results by target variable (y)
# #     grouped_results = results_df.groupby('target_variable')
# #
# #     for target_variable, group in grouped_results:
# #         # Sort the group DataFrame by validation accuracy in descending order
# #         sorted_group = group.sort_values(by='val_accuracy', ascending=False)
# #
# #         # Get the parameters of the best model (first row after sorting)
# #         best_row = sorted_group.iloc[0]
# #
# #         # Extract and store the relevant parameters and accuracies
# #         best_params = {
# #             'hidden_sizes': best_row['hidden_sizes'],
# #             'lr': best_row['lr'],
# #             'activation': best_row['activation'],
# #             'optimizer': best_row['optimizer'],
# #             'val_accuracy': best_row['val_accuracy'],
# #             'test_accuracy': best_row['test_accuracy']
# #         }
# #
# #         best_parameters[target_variable] = best_params
# #
# #     return best_parameters
# #
# #
# # def train_model(model, criterion, optimizer, train_loader, val_loader, epochs=10):
# #     best_val_accuracy = 0.0  # Initialize best validation accuracy
# #     for epoch in range(epochs):
# #         model.train()
# #         for inputs, labels in train_loader:
# #             optimizer.zero_grad()
# #             outputs = model(inputs)
# #             loss = criterion(outputs, labels)
# #             loss.backward()
# #             optimizer.step()
# #
# #         model.eval()
# #         val_loss = 0.0
# #         correct = 0
# #         total = 0
# #         with torch.no_grad():
# #             for inputs, labels in val_loader:
# #                 outputs = model(inputs)
# #                 val_loss += criterion(outputs, labels).item()
# #                 _, predicted = torch.max(outputs, 1)
# #                 total += labels.size(0)
# #                 correct += (predicted == labels).sum().item()
# #
# #         val_loss /= len(val_loader)
# #         val_accuracy = correct / total
# #
# #         print(
# #             f'Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
# #
# #         # Check if the current model has achieved the best validation accuracy
# #         if val_accuracy > best_val_accuracy:
# #             best_val_accuracy = val_accuracy
# #             best_model_state_dict = model.state_dict()
# #
# #     return best_val_accuracy, best_model_state_dict
# #
# #
# # def test_model(model, test_loader):
# #     model.eval()
# #     correct = 0
# #     total = 0
# #     with torch.no_grad():
# #         for inputs, labels in test_loader:
# #             outputs = model(inputs)
# #             _, predicted = torch.max(outputs, 1)
# #             total += labels.size(0)
# #             correct += (predicted == labels).sum().item()
# #     test_accuracy = correct / total
# #     print(f'Test Accuracy: {test_accuracy:.4f}')
# #     return test_accuracy
# #
# #
# # def preprocess_train_evaluate(X, y, use_smote=True):
# #     if use_smote:
# #         smote = SMOTE(random_state=42)
# #         X_resampled, y_resampled = smote.fit_resample(X, y)
# #     else:
# #         X_resampled, y_resampled = X, y
# #
# #     X_train_val, X_test, y_train_val, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2)
# #     X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25)
# #
# #     scaler = StandardScaler()
# #     X_train_scaled = scaler.fit_transform(X_train)
# #     X_val_scaled = scaler.transform(X_val)
# #     X_test_scaled = scaler.transform(X_test)
# #
# #     X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
# #     X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
# #     X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
# #     y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
# #     y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)
# #     y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)
# #
# #     class CustomDataset(Dataset):
# #         def __init__(self, X, y):
# #             self.X = X
# #             self.y = y
# #
# #         def __len__(self):
# #             return len(self.X)
# #
# #         def __getitem__(self, idx):
# #             return self.X[idx], self.y[idx]
# #
# #     train_dataset = CustomDataset(X_train_tensor, y_train_tensor)
# #     val_dataset = CustomDataset(X_val_tensor, y_val_tensor)
# #     test_dataset = CustomDataset(X_test_tensor, y_test_tensor)
# #
# #     train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# #     val_loader = DataLoader(val_dataset, batch_size=64)
# #     test_loader = DataLoader(test_dataset, batch_size=64)
# #
# #     class FlexibleNN(nn.Module):
# #         def __init__(self, input_size, hidden_sizes, output_size, activation):
# #             super(FlexibleNN, self).__init__()
# #             self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_sizes[0])])
# #             for i in range(len(hidden_sizes) - 1):
# #                 self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
# #             self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
# #             self.activation = activation
# #
# #         def forward(self, x):
# #             for hidden_layer in self.hidden_layers:
# #                 x = self.activation(hidden_layer(x))
# #             x = F.softmax(self.output_layer(x), dim=1)  # Softmax activation
# #             return x
# #
# #     hidden_layers_configurations = [
# #         [64],
# #         [64, 32],
# #         [128, 64, 32]
# #     ]
# #
# #     learning_rates = [0.001, 0.01, 0.0001]
# #
# #     activations = [F.relu, F.leaky_relu, F.sigmoid]  # Add more activation functions as needed
# #
# #     optimizers = [optim.Adam, optim.SGD, optim.RMSprop]
# #
# #     combinations = list(itertools.product(hidden_layers_configurations, learning_rates, activations, optimizers))
# #
# #     results = []
# #
# #     for hidden_sizes, lr, activation_func, optimizer_class in combinations:
# #         model = FlexibleNN(input_size=X_train_tensor.shape[1], hidden_sizes=hidden_sizes, output_size=2,
# #                            activation=activation_func)
# #
# #         optimizer = optimizer_class(model.parameters(), lr=lr)
# #
# #         class_weights = compute_class_weight('balanced', classes=np.unique(y_train_tensor.numpy()),
# #                                              y=y_train_tensor.numpy())
# #         class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
# #
# #         criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
# #
# #         summary(model, input_size=(X_train_tensor.shape[1],))
# #
# #         best_val_accuracy, best_model_state_dict = train_model(model, criterion, optimizer, train_loader, val_loader)
# #
# #         test_accuracy = test_model(model, test_loader)
# #
# #         result = {
# #             'target_variable': y.name,
# #             'hidden_sizes': hidden_sizes,
# #             'lr': lr,
# #             'activation': activation_func.__name__,
# #             'optimizer': optimizer_class.__name__,
# #             'val_accuracy': best_val_accuracy,
# #             'test_accuracy': test_accuracy
# #         }
# #         results.append(result)
# #
# #         # Save the state dictionary of the best model for future use
# #         torch.save(best_model_state_dict, f"best_model_{y.name}.pt")
# #
# #     return results
# #
# #
# # if __name__ == '__main__':
# #     import pandas as pd
# #     import numpy as np
# #     from sklearn.model_selection import KFold
# #
# #     # Read the CSV file into a DataFrame
# #     df = pd.read_csv(r"C:\Users\WesalAwida\PycharmProjects\fl\data.csv")
# #
# #     # Assuming suicid_data is the name of your DataFrame
# #     columns_to_convert = ['Parent_Reported_Suicidality', 'Parent_Reported_SI', 'Parent_Reported_SB', 'Self_Reported_Sl']
# #
# #     # Convert string values to integers
# #     for column in columns_to_convert:
# #         df[column] = df[column].astype(int)
# #
# #     # Now, convert values to 0 or 1 based on the condition
# #     for column in columns_to_convert:
# #         df[column] = (df[column] > 0).astype(int)
# #     X = df.drop(columns=columns_to_convert)
# #
# #     all_results = []
# #
# #     kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold cross-validation
# #
# #     combined_results = pd.DataFrame(columns=['SMOTE', 'target_variable', 'val_accuracy', 'test_accuracy'])
# #
# #     # Iterate over each target variable and call the preprocess_train_evaluate function
# #     for column in columns_to_convert:
# #         y = df[column]
# #         print(f"\nProcessing for target variable: {column}")
# #
# #         results_for_y_no_smote = []
# #         results_for_y_with_smote = []
# #
# #         for train_index, test_index in kf.split(X):
# #             X_train, X_test = X.iloc[train_index], X.iloc[test_index]
# #             y_train, y_test = y.iloc[train_index], y.iloc[test_index]
# #
# #             print("\nExperiment without SMOTE:")
# #             result_no_smote = preprocess_train_evaluate(X_train, y_train, use_smote=False)
# #             results_for_y_no_smote.extend(result_no_smote)
# #
# #             print("\nExperiment with SMOTE:")
# #             result_with_smote = preprocess_train_evaluate(X_train, y_train, use_smote=True)
# #             results_for_y_with_smote.extend(result_with_smote)
# #
# #             # Compute average results
# #             avg_results_no_smote = {}
# #             for key in results_for_y_no_smote[0].keys():
# #                 if key in ['target_variable', 'hidden_sizes', 'lr', 'activation', 'optimizer']:
# #                     continue
# #                 avg_results_no_smote[key] = np.mean([result[key] for result in results_for_y_no_smote])
# #
# #             avg_results_with_smote = {}
# #             for key in results_for_y_with_smote[0].keys():
# #                 if key in ['target_variable', 'hidden_sizes', 'lr', 'activation', 'optimizer']:
# #                     continue
# #                 avg_results_with_smote[key] = np.mean([result[key] for result in results_for_y_with_smote])
# #
# #             # Append results to all_results or do further processing as needed
# #
# #             # Extract relevant information for the DataFrame
# #             avg_results_no_smote_df = {'target_variable': column, 'SMOTE': 'without_smote', **avg_results_no_smote}
# #             avg_results_with_smote_df = {'target_variable': column, 'SMOTE': 'with_smote', **avg_results_with_smote}
# #
# #             # Combine the two DataFrames
# #             combined_results = pd.concat([combined_results, pd.DataFrame(avg_results_no_smote_df, index=[0]),
# #                                           pd.DataFrame(avg_results_with_smote_df, index=[0])])
# #
# #         # Rearrange columns
# #         combined_results = combined_results[['SMOTE', 'target_variable', 'val_accuracy', 'test_accuracy']]
# #
# #         # Save to CSV
# #         combined_results.to_csv("combined_results.csv", index=False)
# #
# #         print("Combined results saved to combined_results.csv")
# #
# #
# import pandas as pd
#
# # Load the CSV file into a DataFrame
# df = pd.read_csv(r"C:\Users\WesalAwida\PycharmProjects\neural_cstg\wesal_df_data\combined_results.csv")
#
# # Pivot the DataFrame to get the desired table format
# table_val = df.pivot_table(index='SMOTE', columns='target_variable', values='val_accuracy', aggfunc='max')
# table_test = df.pivot_table(index='SMOTE', columns='target_variable', values='test_accuracy', aggfunc='max')
#
# # Reorder columns to match the desired order
# table_val = table_val[['Parent_Reported_Suicidality', 'Parent_Reported_SI', 'Parent_Reported_SB', 'Self_Reported_Sl']]
# table_test = table_test[['Parent_Reported_Suicidality', 'Parent_Reported_SI', 'Parent_Reported_SB', 'Self_Reported_Sl']]
#
# # Combine both tables into one
# table_combined = pd.concat([table_val, table_test], keys=['Validation Accuracy', 'Test Accuracy'])
# table_combined.to_csv("fl_results_table.csv")
#
# print(table_combined)


import pandas as pd
import itertools

# Assuming df is your DataFrame with columns for Age, Parent_Reported_Suicidality, Parent_Reported_SI, Parent_Reported_SB, Self_Reported_Sl
# Replace 'df' with the name of your DataFrame
df = pd.read_csv(r"C:\Users\WesalAwida\PycharmProjects\neural_cstg\wesal_df_data\suicide_data\suicide_data.csv")

# Retrieve column names
columns = df.columns
print(columns)
x = ['Age', 'Sex', 'Financial_problems', 'Two_Parents_Household',
       'Adoptionor_FosterCare', 'Childrens_Aid_Service',
       'Family_Relationship_Difficulties', 'Between_Caregivers_Violence',
       'Caregiver_To_Child_Violence', 'Head_Injury', 'Stimulant_Meds',
       'Full_Scale_IQ', 'WISC_Vocabulary', 'WISC_BlockDesign',
       'Social_Withdrawal', 'Social_Conflicts', 'Academic_Difficulty',
       'School_Truancy', 'Inattention', 'Hyperactivity_Impulsivity',
       'Irritability', 'Defiance', 'Aggresive_Conduct_Problems',
       'NonAggresive_Conduct_Problems', 'Depression', 'Anxiety',
       'Sleep_Prolems', 'Somatization']

y= ['Parent_Reported_Suicidality', 'Parent_Reported_SI', 'Parent_Reported_SB', 'Self_Reported_Sl']

