% Minimum probability of error classifiers
% Qiang Liu 20241002
clear; close all; clc;
%--------------------------------------------------------------------------
%% Wine Quality dataset
%--------------------------------------------------------------------------
data_path = '/Users/ln915/Documents/Lvx/2024_Fall/Intro2ML/Assignments/1/data/wine+quality/';
red_wine = readmatrix([data_path 'winequality-red.csv']);
white_wine = readmatrix([data_path 'winequality-white.csv']);

% Separate features and labels
x = red_wine(:,1:end-1);
label =red_wine(:,end);

classes = unique(label); % although labels should be 0-10, but only 3 4 5 6 7 8
C = length(classes);  % Number of classes
[N, d] = size(x);     % N = number of samples, d = number of features

mu = zeros(C, d);
cov_matrices = cell(C, 1);
priors = zeros(C, 1);

lambda = 1e-2;% Regularization parameter (lambda)

for i = 1:C % Estimate mean vectors, covariance matrices, and class priors
    class_samples = x(label == classes(i), :);
    mu(i, :) = mean(class_samples);
    cov_sample = cov(class_samples);
    if abs(det(cov_sample)) < 1e-6     % Regularize covariance matrix if needed
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

predicted_labels = zeros(N, 1);
loss = ones(C, C) - eye(C);
expectedRisks = loss * classPosteriors';

[~, predicted_label] = min(expectedRisks, [], 1);
predicted_labels = classes(predicted_label);


% [~, predicted_label] = max(classPosteriors, [],2);
% predicted_labels = classes(predicted_label);

% Calculate confusion matrix
conf_mat = confusionmat(label, predicted_labels);
disp('Confusion Matrix:');
disp(conf_mat);

% Calculate classification error
error_rate = sum(predicted_labels ~= label) / N;
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
gscatter(score(:,1), score(:,2), label);
xlabel('1st Principal Component');
ylabel('2nd Principal Component');
title('Wine Quality Dataset: PCA Visualization (First 2 Components)');
legend('show');

figure;
scatter3(score(:,1), score(:,2), score(:,3), 10, label, 'filled');
xlabel('1st Principal Component');
ylabel('2nd Principal Component');
zlabel('3rd Principal Component');
title('Wine Quality Dataset: PCA Visualization (First 3 Components)');
grid on;
