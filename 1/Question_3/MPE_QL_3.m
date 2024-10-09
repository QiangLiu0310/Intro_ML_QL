clear; close all; clc;
%--------------------------------------------------------------------------
%% Human Activity Recognition Using Smartphones
%--------------------------------------------------------------------------
data_path_train = '/Users/ln915/Documents/Lvx/2024_Fall/Intro2ML/Assignments/1/data/human+activity+recognition+using+smartphones/UCI_HAR_Dataset/train/';
data_path_test = '/Users/ln915/Documents/Lvx/2024_Fall/Intro2ML/Assignments/1/data/human+activity+recognition+using+smartphones/UCI_HAR_Dataset/test/';

% Load the training data
X_train = readmatrix([data_path_train 'X_train.txt']);
y_train = readmatrix([data_path_train 'y_train.txt']);

% Load the testing data
X_test = readmatrix([data_path_test 'X_test.txt']);
y_test = readmatrix([data_path_test 'y_test.txt']);

%--------------------------------------------------------------------------
%% Train the MAP Classifier on the Training Dataset
%--------------------------------------------------------------------------
classes = unique(y_train); % Identify the unique class labels in the training set
C = length(classes);       % Number of classes
[N_train, d] = size(X_train); % N_train = number of training samples, d = number of features

mu = zeros(C, d);            % Mean vectors for each class
cov_matrices = cell(C, 1);    % Covariance matrices for each class
priors = zeros(C, 1);         % Priors for each class

lambda = 1e-2; % Regularization parameter (lambda)

for i = 1:C
    class_samples = X_train(y_train == classes(i), :); % Get samples of class i
    mu(i, :) = mean(class_samples);                    % Mean of class i
    cov_sample = cov(class_samples);                   % Covariance of class i
    if abs(det(cov_sample)) < 1e-6                     % Regularize covariance matrix if needed
        cov_matrices{i} = cov_sample + lambda * eye(d);
    else
        cov_matrices{i} = cov_sample;
    end
    priors(i) = size(class_samples, 1) / N_train;      % Prior probability of class i
end

%--------------------------------------------------------------------------
%% Test the Classifier on the Test Dataset
%--------------------------------------------------------------------------
N_test = size(X_test, 1);  % Number of test samples
p_x_given_c_test = zeros(N_test, C); % Store the likelihood P(x|C=c)

for i = 1:C
    p_x_given_c_test(:, i) = mvnpdf(X_test, mu(i, :), cov_matrices{i});
end

p_x_test = p_x_given_c_test * priors; % P(x) = sum_c P(x|C=c) * P(C=c)
classPosteriors_test = (p_x_given_c_test .* priors') ./ p_x_test; % Posterior P(C=c|x)

% Predict the labels for test data
loss = ones(C, C) - eye(C);
expectedRisks_test = loss * classPosteriors_test';

[~, predicted_label_test] = min(expectedRisks_test, [], 1);
predicted_labels_test = classes(predicted_label_test);

%--------------------------------------------------------------------------
%% Evaluate the Classifier
%--------------------------------------------------------------------------
% Calculate confusion matrix for the test data
conf_mat_test = confusionmat(y_test, predicted_labels_test);
disp('Confusion Matrix (Test Data):');
disp(conf_mat_test);

% Calculate classification error rate for test data
error_rate_test = sum(predicted_labels_test ~= y_test) / N_test;
fprintf('Classification Error Rate (Test Data): %.2f%%\n', error_rate_test * 100);

% Plot the confusion matrix as a heatmap
figure;
heatmap(classes, classes, conf_mat_test, 'Title', 'Confusion Matrix (Test Data)', ...
    'XLabel', 'Predicted Class', 'YLabel', 'True Class');