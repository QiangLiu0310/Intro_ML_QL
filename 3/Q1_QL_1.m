clear; close all;clc;
%--------------------------------------------------------------------------
%% Part 1: Data Distribution
%--------------------------------------------------------------------------

N = 10000;
C = 4;
gmm_pdf = struct();
gmm_pdf.priors = ones(1, C) / C;  % uniform prior
gmm_pdf.mu = [0, 0, 0; 2, 0, 0; 4, 0, 0; 6, 0, 0];  % Gaussian distribution means
gmm_pdf.Sigma(:, :, 1) = [1, 0.3, 1.4; 0.3, 1, 0.3; 1.4, 0.3, 7];
gmm_pdf.Sigma(:, :, 2) = [1, -0.4, -0.7; -0.4, 1, -0.4; -0.7, -0.4, 3];
gmm_pdf.Sigma(:, :, 3) = [1, 0.4, 0.7; 0.4, 1, 0.4; 0.7, 0.4, 3];
gmm_pdf.Sigma(:, :, 4) = [1, -0.3, -1.4; -0.3, 1, -0.3; -1.4, -0.3, 7];

[X, labels] = generate_data_from_gmm(N, gmm_pdf);

% Plot the original data and their true labels
figure;
hold on;
scatter3(X(labels == 1, 1), X(labels == 1, 2), X(labels == 1, 3), 'r', 'DisplayName', 'Class 1');
scatter3(X(labels == 2, 1), X(labels == 2, 2), X(labels == 2, 3), 'b', 'DisplayName', 'Class 2');
scatter3(X(labels == 3, 1), X(labels == 3, 2), X(labels == 3, 3), 'g', 'DisplayName', 'Class 3');
scatter3(X(labels == 4, 1), X(labels == 4, 2), X(labels == 4, 3), 'k', 'DisplayName', 'Class 4');
xlabel('x_1');
ylabel('x_2');
zlabel('x_3');
axis equal; grid on
view(3);
title('Data and True Class Labels');
legend;
hold off;

%--------------------------------------------------------------------------
%% Part 3: Generate Data
%--------------------------------------------------------------------------

N_train = [100, 500, 1000, 5000, 10000];
N_test = 100000;
X_train = cell(1, numel(N_train));
y_train = cell(1, numel(N_train));

for i = 1:numel(N_train)
    N_i = N_train(i);
    fprintf("Generating the training data set; Ntrain = %d\n", N_i);
    [X_i, y_i] = generate_data_from_gmm(N_i, gmm_pdf);
    X_train{i} = X_i;
    y_train{i} = y_i;
end
fprintf("Generating the test set; Ntest = %d\n", N_test);
[X_test, y_test] = generate_data_from_gmm(N_test, gmm_pdf);

%--------------------------------------------------------------------------
%% Part 4: Theoretically Optimal Classifier
%--------------------------------------------------------------------------
class_cond_likelihoods = zeros(C, N_test);
for i = 1:C
    class_cond_likelihoods(i, :) = mvnpdf(X_test, gmm_pdf.mu(i, :), gmm_pdf.Sigma(:, :, i))';
end
[~, decisions] = max(class_cond_likelihoods, [], 1);
misclass_samples = sum(decisions' ~= y_test);
min_prob_error = misclass_samples / N_test;
fprintf("Probability of Error on Test Set using the True Data PDF: %.4f\n", min_prob_error);

%--------------------------------------------------------------------------
%% Part 5: Model Order Selection
%--------------------------------------------------------------------------

% Parameters
K = 10;
P_list = [2, 4, 8, 16, 24, 32, 48, 64, 128, 256, 512];
P_best_list = [];

figure;
hold on;
xlabel('Number of Perceptrons');
ylabel('Probability of Error');
title('No. Perceptrons vs Cross-Validation Pr(error)');

for i = 1:length(N_train)
    data = X_train{i};
    labels = y_train{i};

    % Cross-validation to find optimal perceptrons for current training set
    [P_best, P_CV_err] = k_fold_cv_perceptrons(K, P_list, data, labels);
    P_best_list = [P_best_list; P_best];

    % Plot error vs perceptrons
    plot(P_list, P_CV_err, 'DisplayName', sprintf('N = %d', N_train(i)));
end

% Add legend and plot minimum error line
yline(min_prob_error, '--k', 'Min. Pr(error)', 'LabelHorizontalAlignment', 'left');
legend('show');
hold off;


%--------------------------------------------------------------------------
%% Part 6: Model Training
%--------------------------------------------------------------------------

% List of trained MLPs for later testing
trained_mlps = cell(length(X_train), 1);
num_restarts = 10;

% Loop through each training dataset
for i = 1:length(X_train)
    fprintf("Training model for N = %d\n", size(X_train{i}, 1));
    X_i = X_train{i};
    y_i = categorical(y_train{i});  % Convert labels to categorical for classification

    % Store models and losses for each restart
    restart_mlps = cell(num_restarts, 1);
    restart_losses = zeros(num_restarts, 1);

    % Train with multiple restarts to avoid suboptimal local minima
    for r = 1:num_restarts
        layers = createTwoLayerMLP(size(X_i, 2), P_best_list(i), C);
        options = trainingOptions('sgdm', ...
            'InitialLearnRate', 0.01, ...
            'Momentum', 0.9, ...
            'MaxEpochs', 100, ...
            'Shuffle', 'every-epoch', ...
            'Verbose', true); % false
        [trained_net, info] = trainNetwork(X_i, y_i, layers, options);
        restart_mlps{r} = trained_net;
        % restart_losses(r) = trained_net.Loss;
        tmp = info.TrainingLoss
        restart_losses(r) =min(tmp);
        clear tmp
    end

    % Select the model with the minimum loss from multiple restarts
    [~, best_idx] = min(restart_losses);
    trained_mlps{i} = restart_mlps{best_idx};
end

%--------------------------------------------------------------------------
%% Part 7: Performance Assessment
%--------------------------------------------------------------------------

X_test_tensor = X_test;
y_test_categorical = categorical(y_test);  % Convert true labels to categorical
pr_error_list = zeros(length(X_train), 1);

fprintf("Probability of error results summarized below per trained MLP: \n");
fprintf("\t # of Training Samples \t Pr(error)\n");

for i = 1:length(X_train)
    predicted_labels = classify(trained_mlps{i}, X_test_tensor);
    prob_error = sum(predicted_labels ~= y_test_categorical) / length(y_test_categorical);
    pr_error_list(i) = prob_error;
    fprintf("\t\t %d \t\t   %.3f\n", N_train(i), prob_error);
end

figure;
hold on;
plot(log10(N_train), pr_error_list, '-o', 'LineWidth', 1.5);
yline(min_prob_error, '--', 'Min. Pr(error)', 'LineWidth', 1.5);

title('No. of Training Samples vs Test Set Pr(error)');
xlabel('Log(Number of Training Samples)');
ylabel('Probability of Error');
legend('Pr(error) per model', 'Location', 'northeast');
grid on;
hold off;

function [X, labels] = generate_data_from_gmm(N, pdf_params)
n = size(pdf_params.mu, 2);
X = zeros(N, n);
labels = zeros(N, 1);
u = rand(N, 1);
thresholds = [0, cumsum(pdf_params.priors)];

for l = 1:length(pdf_params.priors)
    indices = find(thresholds(l) <= u & u < thresholds(l+1));
    N_labels = numel(indices);
    labels(indices) = l;
    X(indices, :) = mvnrnd(pdf_params.mu(l, :), pdf_params.Sigma(:, :, l), N_labels);
end
end

function layers = createTwoLayerMLP(n, P, C)
% Define the layers of a two-layer MLP
layers = [
    featureInputLayer(n, 'Name', 'input')
    fullyConnectedLayer(P, 'Name', 'fc1')
    % reluLayer('Name', 'relu')
    eluLayer('Name','elu')
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


function predictions = model_predict(trainedModel, X_test)
pred_probs = predict(trainedModel, X_test);
[~, predictions] = max(pred_probs, [], 2);
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

        layers = createTwoLayerMLP(size(X_train_k, 2), P, max(labels));
        trainedModel = trainTwoLayerMLP(X_train_k, y_train_k, layers, 100);

        predictions = model_predict(trainedModel, X_valid_k);
        error_valid_mk(m, k) = mean(predictions ~= y_valid_k);
    end
end

error_valid_m = mean(error_valid_mk, 2);
[~, min_idx] = min(error_valid_m);
optimal_P = P_list(min_idx);
end


