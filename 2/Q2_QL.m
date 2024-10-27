clear; close all; clc;

%--------------------------------------------------------------------------
%% Part 0: Prepare the data
%--------------------------------------------------------------------------
clear; close all; clc;

% Part 0: Prepare the data
Ntrain = 100;  % Number of training points
Nvalidate = 1000; 
[xTrain, yTrain, xValidate, yValidate] = hw2q2(Ntrain, Nvalidate);

% Construct the design matrix with cross-terms for cubic polynomial
n = size(xTrain, 2);  % Number of training samples

% Augment xTrain with all cubic terms (including cross-terms)
x_aug = [ones(1, n);                % Bias term
         xTrain(1, :);              % x1
         xTrain(2, :);              % x2
         xTrain(1, :).^2;           % x1^2
         xTrain(2, :).^2;           % x2^2
         xTrain(1, :) .* xTrain(2, :);  % x1 * x2
         xTrain(1, :).^3;           % x1^3
         xTrain(2, :).^3;           % x2^3
         (xTrain(1, :).^2) .* xTrain(2, :);  % x1^2 * x2
         xTrain(1, :) .* (xTrain(2, :).^2)]; % x1 * x2^2

%--------------------------------------------------------------------------
%% Part 1: Maximum Likelihood Estimation (MLE)
%--------------------------------------------------------------------------

sigma = 1^2;  % Given noise variance
costfunc_ml = @(w) sum((yTrain - (w' * x_aug)).^2) / (2 * sigma);  % Squared error cost function
w_ml = fminsearch(costfunc_ml, zeros(10, 1));  % Minimize squared error
w_ml=(x_aug * x_aug')\ (x_aug *yTrain' );

yTrain_pred = w_ml' * x_aug;
n_validate = size(xValidate, 2);  % Number of validation points

x_aug_validate = [ones(1, n_validate);                % Bias term
                  xValidate(1, :);                    % x1
                  xValidate(2, :);                    % x2
                  xValidate(1, :).^2;                 % x1^2
                  xValidate(2, :).^2;                 % x2^2
                  xValidate(1, :) .* xValidate(2, :); % x1 * x2
                  xValidate(1, :).^3;                 % x1^3
                  xValidate(2, :).^3;                 % x2^3
                  (xValidate(1, :).^2) .* xValidate(2, :);  % x1^2 * x2
                  xValidate(1, :) .* (xValidate(2, :).^2)]; % x1 * x2^2

yValidate_pred = w_ml' * x_aug_validate;

mse_validate = mean((yValidate - yValidate_pred).^2);
disp(['Mean Squared Error on Validation Set: ', num2str(mse_validate)]);

%--------------------------------------------------------------------------
%% Part 2: MAP
%--------------------------------------------------------------------------

expandFeatures = @(x) [x(1,:).^3; x(1,:).^2 .* x(2,:); x(1,:) .* x(2,:).^2; ...
                       x(2,:).^3; x(1,:).^2; x(1,:) .* x(2,:); x(2,:).^2; ...
                       x(1,:); x(2,:); ones(1, size(x, 2))];

% Expand the training and validation features
PhiTrain = expandFeatures(xTrain); 
PhiValidate = expandFeatures(xValidate);  

gamma_vals = logspace(-7, 7, 1000);

w_map = zeros(size(PhiTrain, 1), length(gamma_vals));
errors_validate = zeros(1, length(gamma_vals));
for i = 1:length(gamma_vals)
    gamma = gamma_vals(i);
        
    w_init = zeros(size(PhiTrain, 1), 1);
    costfunc_map = @(w) 1/2*sum((yTrain - (w'*PhiTrain)).^2) + 1/(2*gamma) * (w'*w);
    % w_map(:, i) = fminsearch(costfunc_map, w_init);
w_map(:, i) =(PhiTrain * PhiTrain' + (1 / gamma) * eye(size(PhiTrain, 1)))\ (PhiTrain *yTrain' );

    yPredValidate = w_map(:, i)'*PhiValidate;
    errors_validate(i) = mean((yValidate - yPredValidate).^2);
end



[~, best_gamma_idx] = min(errors_validate);
best_w_map = w_map(:, best_gamma_idx);

%--------------------------------------------------------------------------
%% Display Results
%--------------------------------------------------------------------------
figure;
semilogx(gamma_vals(1:1:end), errors_validate(1:1:end),'b-', 'LineWidth', 2);
xlabel('\gamma');
ylabel('Validation Mean Squared Error');
title('MAP Model Performance for Different \gamma');
hold on;
yline(mse_validate, 'r--', 'LineWidth', 2);  % ML MSE

xlabel('\gamma');
ylabel('Mean Squared Error');
title('MSE for MAP as a function of \gamma');
legend('MAP MSE', 'ML MSE');


