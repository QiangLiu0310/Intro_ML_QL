% Minimum probability of error classifiers
% Qiang Liu 20241002
clear; close all; clc;
%--------------------------------------------------------------------------
%% human activity recognition using smartphones
%--------------------------------------------------------------------------

data_path_train = '/Users/ln915/Documents/Lvx/2024_Fall/Intro2ML/Assignments/1/data/human+activity+recognition+using+smartphones/UCI_HAR_Dataset/train/';
data_path_test = '/Users/ln915/Documents/Lvx/2024_Fall/Intro2ML/Assignments/1/data/human+activity+recognition+using+smartphones/UCI_HAR_Dataset/test/';

% Load the training data
X_train = readmatrix([data_path_train 'X_train.txt']);
y_train = readmatrix([data_path_train 'y_train.txt']);

% Load the testing data
X_test = readmatrix([data_path_test 'X_test.txt']);
y_test = readmatrix([data_path_test 'y_test.txt']);

% combine train and test

x = [X_train; X_test];
y=[y_train; y_test];

classes = unique(y); % although labels should be 0-10, but only 3 4 5 6 7 8
C = length(classes);  % Number of classes
[N, d] = size(x);     % N = number of samples, d = number of features

mu = zeros(C, d);
cov_matrices = cell(C, 1);
priors = zeros(C, 1);

lambda = 1e-2;% Regularization parameter (lambda)
% alpha=5;
for i = 1:C % Estimate mean vectors, covariance matrices, and class priors
    class_samples = x(y == classes(i), :);
    mu(i, :) = mean(class_samples);
    cov_sample = cov(class_samples);
    if abs(det(cov_sample)) < 1e-6     % Regularize covariance matrix if needed
        % lambda=alpha*trace(cov_sample)/rank(cov_sample);
        cov_matrices{i} = cov_sample + lambda * eye(d);
    end
    priors(i) = size(class_samples, 1) / N;
end

%--------------------------------------------------------------------------
%% Classify the samples using minimum-P(error) classification rule
%--------------------------------------------------------------------------

for i=1:C
    p_x_given_c(:, i)=mvnpdf(x, mu(i,:), cov_matrices{i});
end

p_x=p_x_given_c*priors;
classPosteriors = (p_x_given_c .* priors') ./ p_x; % P(C=c|x)

loss = ones(C, C) - eye(C);
expectedRisks = loss * classPosteriors';

[~, predicted_label] = min(expectedRisks, [], 1);
predicted_labels = classes(predicted_label);


% [~, predicted_label] = max(classPosteriors, [],2);
% predicted_labels = classes(predicted_label);

% Calculate confusion matrix
conf_mat = confusionmat(y, predicted_labels);
disp('Confusion Matrix:');
disp(conf_mat);

% Calculate classification error
error_rate = sum(predicted_labels ~= y) / N;
fprintf('Classification Error Rate: %.2f%%\n', error_rate * 100);

figure;
heatmap(classes, classes, conf_mat, 'Title', 'Confusion Matrix', ...
    'XLabel', 'Predicted Class', 'YLabel', 'True Class');

%--------------------------------------------------------------------------
%% Visulization: use PCA to reduce the dimensions
%--------------------------------------------------------------------------

% Perform PCA
[coeff, score, ~, ~, explained] = pca(x);

figure;
gscatter(score(:,1), score(:,2), y);
xlabel('1st Principal Component');
ylabel('2nd Principal Component');
title('human activity recognition: PCA Visualization (First 2 Components)');
legend('show');

figure;
scatter3(score(:,1), score(:,2), score(:,3), 10, y, 'filled');
xlabel('1st Principal Component');
ylabel('2nd Principal Component');
zlabel('3rd Principal Component');
title('human activity recognition: PCA Visualization (First 3 Components)');
grid on;
