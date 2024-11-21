clear; close all; clc;
%--------------------------------------------------------------------------
%% Preparation
%--------------------------------------------------------------------------
% Set parameters for data generation
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

% Plot training and test datasets
figure;
subplot(2, 1, 1);
hold on;
title('Training Set');
scatter(X_train(y_train == -1, 1), X_train(y_train == -1, 2), 'bo');
scatter(X_train(y_train == 1, 1), X_train(y_train == 1, 2), 'k+');
xlabel('x_1');
ylabel('x_2');
legend('Class -1', 'Class 1');
hold off;

subplot(2, 1, 2);
hold on;
title('Test Set');
scatter(X_test(y_test == -1, 1), X_test(y_test == -1, 2), 'bo');
scatter(X_test(y_test == 1, 1), X_test(y_test == 1, 2), 'k+');
xlabel('x_1');
ylabel('x_2');
legend('Class -1', 'Class 1');
hold off;

%--------------------------------------------------------------------------
%% Part 1. SVM
%--------------------------------------------------------------------------

% SVM training with grid search and cross-validation
K = 10; % Number of folds
C_range = logspace(-3, 3, 7);
gamma_range = logspace(-3, 3, 7);

% Store cross-validation results
cv_error = zeros(length(C_range), length(gamma_range));

best_error = Inf;
best_model = [];
best_C = NaN;
best_gamma = NaN;

for i = 1:length(C_range)
    for j = 1:length(gamma_range)
        SVMModel = fitcsvm(X_train, y_train, 'KernelFunction', 'rbf', ...
            'BoxConstraint', C_range(i), 'KernelScale', 1/sqrt(2*gamma_range(j)), ...
            'CrossVal', 'on', 'KFold', K);
        error = kfoldLoss(SVMModel);
        cv_error(i, j) = kfoldLoss(SVMModel);

        if error < best_error
            best_error = error;
            best_model = SVMModel.Trained{1};
            best_C = C_range(i);
            best_gamma = gamma_range(j);
        end
    end
end

fprintf('Best Regularization Strength (C): %.3f\n', best_C);
fprintf('Best Kernel Width (gamma): %.3f\n', best_gamma);
fprintf('SVM Cross-Validation Pr(error): %.3f\n', best_error);

%% Plot Pr(error) vs C for each gamma
figure;
hold on;
for j = 1:length(gamma_range)
    % Extract data for this gamma
    errors_for_gamma = cv_error(:, j);
    plot(C_range, errors_for_gamma, '-o', ...
         'DisplayName', sprintf('\\gamma = %.3f', gamma_range(j)));
end

set(gca, 'XScale', 'log'); % Log scale for C
xlabel('C (Regularization Parameter)');
ylabel('Pr(error)');
title('Pr(error) vs Regularization Parameter for Different \gamma');
legend('show');
grid on;
hold off;

%%  Test the best model on the test set
predictions = predict(best_model, X_test);
prob_error_test = mean(predictions ~= y_test);
fprintf('SVM Pr(error) on the test data set: %.4f\n', prob_error_test);

% Plot decision boundaries
figure;
hold on;
scatter(X_test(y_test == -1, 1), X_test(y_test == -1, 2), 'bo');
scatter(X_test(y_test == 1, 1), X_test(y_test == 1, 2), 'k+');

% Define grid for decision boundary
[x1Grid, x2Grid] = meshgrid(linspace(min(X_test(:, 1)), max(X_test(:, 1)), 200), ...
    linspace(min(X_test(:, 2)), max(X_test(:, 2)), 200));
XGrid = [x1Grid(:), x2Grid(:)];
Z = predict(best_model, XGrid); % Predict class for grid points
Z = reshape(Z, size(x1Grid));

% Use surf for decision boundaries
h = surf(x1Grid, x2Grid, double(Z), 'EdgeColor', 'none'); 
view(2); 
colormap(parula);
alpha(h, 0.2); 

% Identify correct and incorrect classifications
tn = (predictions == -1) & (y_test == -1); % True Negatives
fp = (predictions == 1) & (y_test == -1);  % False Positives
fn = (predictions == -1) & (y_test == 1);  % False Negatives
tp = (predictions == 1) & (y_test == 1);  % True Positives

% Overlay correct and incorrect classifications
plot(X_test(tn, 1), X_test(tn, 2), 'og', 'MarkerSize', 6, 'DisplayName', 'Correct Class -1'); % Green circle
plot(X_test(fp, 1), X_test(fp, 2), 'or', 'MarkerSize', 6, 'DisplayName', 'Incorrect Class -1'); % Red circle
plot(X_test(fn, 1), X_test(fn, 2), '+r', 'MarkerSize', 6, 'DisplayName', 'Incorrect Class 1'); % Red plus
plot(X_test(tp, 1), X_test(tp, 2), '+g', 'MarkerSize', 6, 'DisplayName', 'Correct Class 1'); % Green plus


title('SVM Decision Boundary with Transparency');
xlabel('x_1');
ylabel('x_2');
hold off;

% Display confusion matrix
figure;
confusionchart(y_test, predictions);
title('Confusion Matrix');

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


