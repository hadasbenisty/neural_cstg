% A script to create a toy example for the stg model
number_outputs = 10;
number_features = number_outputs + 1000;
num_meas = 1000;
SNR = 10;

% Create X and real Y
X = randn([number_outputs, num_meas]);
[Y,A] = createRandomLinearTransformation(X);

% add noise
N = 1/SNR * randn(size(Y));
Y_tot = Y + N;
% Pad X
X_fake = rand([number_features - number_outputs, num_meas]);
X_all = [X ; X_fake];

features = X_all;
y = Y_tot;

% Save Data
load("data\paths\paths.mat");
save(fullfile(inputs_path, 'dataset'), 'y', 'features', 'Y_tot', 'Y', 'N');