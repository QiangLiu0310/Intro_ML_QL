% Minimum probability of error classification/ MAP classifier
% Qiang Liu 20241001
clear; close all; clc;
%--------------------------------------------------------------------------
%% Part A 1
%--------------------------------------------------------------------------
C = 3; % classes
N = 10000; % sample numbers
p = [0.3 0.3 0.4]; % priors
n = p.*N; % samples per class

mu_1 = [0 0 0]; % let the distances to be 2-3 for each two
mu_2 = [2 1 2];
mu_3_1 = [-1.5 -0.5 1.5];
mu_3_2 = [2.5 1.5 2.5];

sigma1 = diag([2, 1, 0.5]);
sigma2 = diag([1.5, 2, 1]);
sigma3 = diag([0.7, 1.8, 1.2]);

Xclass1 = mvnrnd(mu_1, sigma1, n(1));
Xclass2 = mvnrnd(mu_2, sigma2, n(2));
Xclass3_1 = mvnrnd(mu_3_1, sigma3, n(3)/2);
Xclass3_2 = mvnrnd(mu_3_2, sigma3, n(3)/2);
Xclass3 = [Xclass3_1; Xclass3_2];

X=[Xclass1; Xclass2; Xclass3];
label = [ones(n(1),1); 2*ones(n(2),1); 3*ones(n(3),1)];

%--------------------------------------------------------------------------
%% Part A 2
%--------------------------------------------------------------------------

p_x_given_L1 = mvnpdf(X, mu_1, sigma1);
p_x_given_L2 = mvnpdf(X, mu_2, sigma2);
p_x_given_L3_1 = mvnpdf(X, mu_3_1, sigma3);
p_x_given_L3_2 = mvnpdf(X, mu_3_2, sigma3);
p_x_given_L3 = 0.5*(p_x_given_L3_1+p_x_given_L3_2);

p_x_given_L=[p_x_given_L1, p_x_given_L2 p_x_given_L3]';
p_x=p*p_x_given_L;
classPosteriors = (p_x_given_L .* p') ./ p_x; % P(L=l|x)

loss_matrices = {
    ones(C, C) - eye(C), ... % 0-1 loss
    [0 10 10; 1 0 10; 1 1 0], ...% Define loss matrices for the different cases
    [0 100 100; 1 0 100; 1 1 0]
    }; % 

% Loop over different loss matrices
for i = 1:length(loss_matrices)
    loss = loss_matrices{i};
    expectedRisks = loss * classPosteriors;
    [~, predicted_label] = min(expectedRisks, [], 1);

    % Confusion matrix
    conf_mat = confusionmat(label, predicted_label');
    % Normalize the confusion matrix
    conf_mat_norm = conf_mat ./ sum(conf_mat, 2);
    disp(['Confusion Matrix for Loss Function ', num2str(i)]);
    disp(conf_mat_norm);

    figure(i+3);
    heatmap([1 2 3], [1 2 3], conf_mat_norm, 'Title', 'Confusion Matrix', ...
    'XLabel', 'Predicted Class', 'YLabel', 'True Class');


    figure(i); % Create a new figure for each loss function
    hold on;
    markers = {'o', 'd', '^'};
    correct_class = find(predicted_label' == label);
    incorrect_class = find(predicted_label' ~= label);

    for class_idx = 1:3
        class_samples = find(label == class_idx);
        correct_class_samples = intersect(correct_class, class_samples);
        incorrect_class_samples = intersect(incorrect_class, class_samples);
        % Correct classifications: unfilled green markers
        scatter3(X(correct_class_samples, 1), X(correct_class_samples, 2), X(correct_class_samples, 3), ...
            50, 'MarkerEdgeColor', [0 0.7 0], 'Marker', markers{class_idx}, 'LineWidth', 1.5);
        % Incorrect classifications: unfilled red markers
        scatter3(X(incorrect_class_samples, 1), X(incorrect_class_samples, 2), X(incorrect_class_samples, 3), ...
            50, 'MarkerEdgeColor', [0.8 0 0], 'Marker', markers{class_idx}, 'LineWidth', 1.5);
    end

    view(3); % 3D view
    grid on;
    xlabel('X1'); ylabel('X2'); zlabel('X3');
    title(['3D Classification Results with Loss Function ', num2str(i)]);
    legend('Class 1 - Correct', 'Class 1 - Incorrect', ...
        'Class 2 - Correct', 'Class 2 - Incorrect', ...
        'Class 3 - Correct', 'Class 3 - Incorrect');
    hold off;
end

figure(7); % show the distribution of X
scatter3(X(label==1,1), X(label==1,2), X(label==1,3), 'ro'); % Class 1 with stars
hold on;
scatter3(X(label==2,1), X(label==2,2), X(label==2,3), 'gd'); % Class 2 with diamonds
scatter3(X(label==3,1), X(label==3,2), X(label==3,3), 'b^'); % Class 3 with triangles
xlabel('X1'); ylabel('X2'); zlabel('X3');
title('generated samples');
legend('Class 1', 'Class 2', 'Class 3');
grid on;
hold off;

