clear; close all;clc;
%--------------------------------------------------------------------------
%% Part 0: Generate Data
%--------------------------------------------------------------------------
n = 2; % Dimensionality of data
mix_pdf.r_plus = 4;
mix_pdf.r_minus = 2;
mix_pdf.prior = 0.5;
mix_pdf.mu = zeros(1, n);
mix_pdf.Sigma = eye(n);

% Number of samples
N_train = 1000;
N_test = 10000;

% Generate training and test sets
[X_train, y_train] = generate_multiring_dataset(N_train, n, mix_pdf);
[X_test, y_test] = generate_multiring_dataset(N_test, n, mix_pdf);

%--------------------------------------------------------------------------
%% Part 1: Model Order Selection
%--------------------------------------------------------------------------

% Parameters
K = 10;
P_list = [2, 4, 8, 16, 24, 32, 48, 64, 128];
P_best_list = [];

figure;
hold on;
xlabel('Number of Perceptrons');
ylabel('Probability of Error');
title('No. Perceptrons vs Cross-Validation Pr(error)');

data = X_train;
labels = y_train;

% Cross-validation to find optimal perceptrons for current training set
[P_best, P_CV_err] = k_fold_cv_perceptrons(K, P_list, data, labels);
P_best_list = [P_best_list; P_best];

% Plot error vs perceptrons
plot(P_list, P_CV_err, 'DisplayName', sprintf('N = %d', N_train));
hold off;

%--------------------------------------------------------------------------
%% Part 2: Model Training
%--------------------------------------------------------------------------

trained_mlps = cell(length(X_train), 1);
num_restarts = 10;

% Loop through each training dataset
fprintf("Training model for N = %d\n", size(X_train, 1));
X_i = X_train;
y_i = categorical(y_train);  % Convert labels to categorical for classification

% Store models and losses for each restart
restart_mlps = cell(num_restarts, 1);
restart_losses = zeros(num_restarts, 1);

% Train with multiple restarts to avoid suboptimal local minima
for r = 1:num_restarts
    layers = createTwoLayerMLP(size(X_i, 2), P_best_list, 2);
    options = trainingOptions('sgdm', ...
        'InitialLearnRate', 0.01, ...
        'Momentum', 0.9, ...
        'MaxEpochs', 100, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', true); % false
    [trained_net, info] = trainNetwork(X_i, y_i, layers, options);
    restart_mlps{r} = trained_net;
    tmp = info.TrainingLoss
    restart_losses(r) =min(tmp);
    clear tmp
end

% Select the model with the minimum loss from multiple restarts
[~, best_idx] = min(restart_losses);
trained_mlps = restart_mlps{best_idx};

%--------------------------------------------------------------------------
%% Part 3: Performance Assessment
%--------------------------------------------------------------------------

X_test_tensor = X_test;
y_test_categorical = categorical(y_test);  % Convert true labels to categorical

fprintf("Probability of error results summarized below per trained MLP: \n");
fprintf("\t # of Training Samples \t Pr(error)\n");

predicted_labels = classify(trained_mlps, X_test_tensor);
prob_error = sum(predicted_labels ~= y_test_categorical) / length(y_test_categorical);
pr_error_list = prob_error;
fprintf("\t\t %d \t\t   %.3f\n", N_train, prob_error); % 0.173

% Plot decision boundaries
figure;
hold on;
scatter(X_test(y_test == -1, 1), X_test(y_test == -1, 2), 'bo');
scatter(X_test(y_test == 1, 1), X_test(y_test == 1, 2), 'k+');

% Define grid for decision boundary
[x1Grid, x2Grid] = meshgrid(linspace(min(X_test(:, 1)), max(X_test(:, 1)), 200), ...
    linspace(min(X_test(:, 2)), max(X_test(:, 2)), 200));
XGrid = [x1Grid(:), x2Grid(:)];
Z = classify(trained_mlps, XGrid); % Predict class for grid points
Z = reshape(Z, size(x1Grid));

% Use surf for decision boundaries
h = surf(x1Grid, x2Grid, double(Z), 'EdgeColor', 'none'); 
view(2); 
colormap(parula);
alpha(h, 0.2); 

y_test_double = double(y_test_categorical); 
predicted_labels_double=double(predicted_labels);

% Identify correct and incorrect classifications
tn = (predicted_labels_double == 1) & (y_test_double == 1); % True Negatives
fp = (predicted_labels_double == 2) & (y_test_double == 1);  % False Positives
fn = (predicted_labels_double == 1) & (y_test_double == 2);  % False Negatives
tp = (predicted_labels_double == 2) & (y_test_double == 2);  % True Positives

% Overlay correct and incorrect classifications
plot(X_test(tn, 1), X_test(tn, 2), 'og', 'MarkerSize', 6, 'DisplayName', 'Correct Class -1'); % Green circle
plot(X_test(fp, 1), X_test(fp, 2), 'or', 'MarkerSize', 6, 'DisplayName', 'Incorrect Class -1'); % Red circle
plot(X_test(fn, 1), X_test(fn, 2), '+r', 'MarkerSize', 6, 'DisplayName', 'Incorrect Class 1'); % Red plus
plot(X_test(tp, 1), X_test(tp, 2), '+g', 'MarkerSize', 6, 'DisplayName', 'Correct Class 1'); % Green plus

title('MLP Decision Boundary');
xlabel('x_1');
ylabel('x_2');
hold off;

% Display confusion matrix
figure;
predicted_labels_double(predicted_labels_double == 1) = -1; % If 1 is present, change it to -1
predicted_labels_double(predicted_labels_double == 2) = 1;  % If 2 is present, change it to 1
confusionchart(y_test, predicted_labels_double);
title('Confusion Matrix');

function layers = createTwoLayerMLP(n, P, C)
% Define the layers of a two-layer MLP
layers = [
    featureInputLayer(n, 'Name', 'input')
    fullyConnectedLayer(P, 'Name', 'fc1')
    eluLayer('Name', 'elu')
    fullyConnectedLayer(C, 'Name', 'fc2')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
    ];
end

function trainedModel = trainTwoLayerMLP(X_train, y_train, layers, numEpochs)
y_train = categorical(y_train);
options = trainingOptions('sgdm', ...
    'MaxEpochs', numEpochs, ...
    'InitialLearnRate', 0.01, ...
    'Momentum', 0.9, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false);
trainedModel = trainNetwork(X_train, y_train, layers, options);
end

function predicted_classes = model_predict(trainedModel, X_test)

    predicted_logits = predict(trainedModel, X_test); % Perform forward pass
    predicted_probs = 1 ./ (1 + exp(-predicted_logits));
    [~, predicted_classes] = max(predicted_probs, [], 2);
    predicted_classes = 2 * (predicted_classes - 1) - 1;  % This maps 1 -> -1 and 2 -> 1
    predicted_classes = reshape(predicted_classes, [], 1); % Ensure the output is a c
end


function [optimal_P, error_valid_m] = k_fold_cv_perceptrons(K, P_list, data, labels)
% Partition data into K folds
cv = cvpartition(labels, 'KFold', K);
error_valid_mk = zeros(length(P_list), K);
for m = 1:length(P_list)
    P = P_list(m);

    for k = 1:K
        % Training and validation indices for k-th fold
        trainIndices = training(cv, k);
        validIndices = test(cv, k);
        X_train_k = data(trainIndices, :);
        y_train_k = labels(trainIndices);
        X_valid_k = data(validIndices, :);
        y_valid_k = labels(validIndices);

        layers = createTwoLayerMLP(size(X_train_k, 2), P, length(unique(labels))); % max(labels)
        trainedModel = trainTwoLayerMLP(X_train_k, y_train_k, layers, 100);

        predicted_classes = model_predict(trainedModel, X_valid_k);
        error_valid_mk(m, k) = mean(predicted_classes ~= y_valid_k); 
    end
end

error_valid_m = mean(error_valid_mk, 2);
[~, min_idx] = min(error_valid_m);
optimal_P = P_list(min_idx);
end

function [X, labels] = generate_multiring_dataset(N, n, pdf_params)
% Generate multiring dataset
X = zeros(N, n);
labels = ones(N, 1);
indices = rand(N, 1) < pdf_params.prior;
labels(indices) = -1;
num_neg = sum(indices);

theta = rand(N, 1) * 2 * pi - pi;
uniform_component = [cos(theta), sin(theta)];

% Generate positive class samples
X(~indices, :) = pdf_params.r_plus * uniform_component(~indices, :) + ...
    mvnrnd(pdf_params.mu, pdf_params.Sigma, N - num_neg);
% Generate negative class samples
X(indices, :) = pdf_params.r_minus * uniform_component(indices, :) + ...
    mvnrnd(pdf_params.mu, pdf_params.Sigma, num_neg);
end


