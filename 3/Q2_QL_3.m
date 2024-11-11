clear; close all; clc;

% Parameters
dataset_sizes = [10, 100, 1000]; % Different dataset sizes to test
num_folds = 10; % Number of folds for cross-validation
num_repeats = 100; % Number of times to repeat the experiment
max_components = 10; % Max number of components to evaluate

% True GMM Parameters
alpha_true = [0.2, 0.25, 0.25, 0.3];
mu_true = [-10, -5, 0, 10; 0, 0, 0, 0];
Sigma_true(:,:,1) = [3 1; 1 20];
Sigma_true(:,:,2) = [5 1; 1 5];
Sigma_true(:,:,3) = [5 1; 1 5];
Sigma_true(:,:,4) = [4 -1; 1 8];

% Store selected model orders for each dataset size
selected_model_orders = zeros(length(dataset_sizes), num_repeats);

for d = 1: 1% length(dataset_sizes)
    N = dataset_sizes(d); % Number of samples in the dataset

    for repeat = 1:num_repeats
        % Generate dataset
        x = randGMM(N, alpha_true, mu_true, Sigma_true);

        % 10-fold cross-validation
        indices = crossvalind('Kfold', N, num_folds);
        log_likelihoods = zeros(max_components, 1);

        for M = 1: max_components

            % ensure M does not exceed N
            if M>=N
                warning ('number of components M exceeds number of data points N. Sikpping M=%d.', M, N);
                break;
            end

            fold_log_likelihoods = zeros(num_folds, 1);

            for fold = 1:num_folds
                % Split data into training and validation sets
                validation_idx = (indices == fold);
                training_idx = ~validation_idx;
                x_train = x(:, training_idx);
                x_val = x(:, validation_idx);

                % Fit GMM to the training set with M components
                [alpha, mu, Sigma] = fitGMM(x_train, M);

                % Evaluate log-likelihood on validation set
                fold_log_likelihoods(fold) = sum(log(evalGMM(x_val, alpha, mu, Sigma)));
            end
            % Average log-likelihood across folds
            log_likelihoods(M) = mean(fold_log_likelihoods);
        end

        % Select the model order with the highest average log-likelihood
        if d~=1
            [~, best_model_order] = max(log_likelihoods);
            selected_model_orders(d, repeat) = best_model_order;
        else
            [~, best_model_order] = max(log_likelihoods(1:9));
            selected_model_orders(d, repeat) = best_model_order;
        end
        disp(num2str(repeat))
    end
end

% Display results
for d = 1:length(dataset_sizes)
    fprintf('Dataset size: %d\n', dataset_sizes(d));
    counts = histcounts(selected_model_orders(d, :), 1:max_components+1);
    for M = 1:max_components
        fprintf('Model order %d selected %d times\n', M, counts(M));
    end
    fprintf('\n');
end

% Plot histograms for each dataset size
figure;
for d = 1:length(dataset_sizes)
    subplot(1, length(dataset_sizes), d);
    histogram(selected_model_orders(d, :), 'BinEdges', 1:max_components+1, 'FaceColor', 'b');
    title(sprintf('Dataset size: %d', dataset_sizes(d)));
    xlabel('Model Order (Number of Components)');
    ylabel('Frequency of Selection');
    xlim([1, max_components+1]);
    xticks(1:max_components);
end

% Optimized Functions
function [alpha, mu, Sigma] = fitGMM(x, M)
% Initialize GMM parameters
[d, N] = size(x);
regWeight = 1e-10;
delta = 1e-2;

% Initialize parameters
alpha = ones(1, M) / M;
shuffledIndices = randperm(N);
mu = x(:, shuffledIndices(1:M));
Sigma = repmat(eye(d), [1, 1, M]); % Initialize as identity matrices

% E-M loop
Converged = false;
while ~Converged
    % E-step: compute responsibilities
    temp = zeros(M, N);
    for l = 1:M
        temp(l, :) = alpha(l) * evalGaussian(x, mu(:, l), Sigma(:, :, l));
    end
    plgivenx = temp ./ sum(temp, 1);

    % M-step: update parameters
    alphaNew = mean(plgivenx, 2)';
    muNew = (x * plgivenx') ./ sum(plgivenx, 2)';

    for l = 1:M
        v = x - muNew(:, l);
        weighted_v = bsxfun(@times, v, sqrt(plgivenx(l, :)));
        SigmaNew(:, :, l) = weighted_v * weighted_v' / sum(plgivenx(l, :)) + regWeight * eye(d);
    end

    % Check for convergence
    Dalpha = sum(abs(alphaNew - alpha));
    Dmu = sum(sum(abs(muNew - mu)));
    DSigma = sum(sum(abs(SigmaNew - Sigma), 1), 3);
    Converged = ((Dalpha + Dmu + DSigma) < delta);

    % Update parameters for the next iteration
    alpha = alphaNew;
    mu = muNew;
    Sigma = SigmaNew;
end
end

function x = randGMM(N, alpha, mu, Sigma)
d = size(mu, 1);
cum_alpha = [0, cumsum(alpha)];
u = rand(1, N);
x = zeros(d, N);

for m = 1:length(alpha)
    ind = find(cum_alpha(m) < u & u <= cum_alpha(m + 1));
    x(:, ind) = randGaussian(length(ind), mu(:, m), Sigma(:, :, m));
end
end

function x = randGaussian(N, mu, Sigma)
n = length(mu);
z = randn(n, N);
A = chol(Sigma, 'lower');
x = A * z + repmat(mu, 1, N);
end

function gmm = evalGMM(x, alpha, mu, Sigma)
gmm = zeros(1, size(x, 2));
for m = 1:length(alpha)
    gmm = gmm + alpha(m) * evalGaussian(x, mu(:, m), Sigma(:, :, m));
end
end

function g = evalGaussian(x, mu, Sigma)
[n, N] = size(x);
invSigma = inv(Sigma);
C = (2 * pi)^(-n / 2) * det(invSigma)^(1 / 2);
E = -0.5 * sum((x - repmat(mu, 1, N)) .* (invSigma * (x - repmat(mu, 1, N))), 1);
g = C * exp(E);
end