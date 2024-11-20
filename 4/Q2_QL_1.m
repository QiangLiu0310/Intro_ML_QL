clear; close all; clc;

image = imread("24004.jpg");

% Generate feature vector for the image
[img_np, feature_vector] = generate_feature_vector(image);

% Perform GMM Segmentation
num_components = 4;
options = statset('MaxIter', 400, 'TolFun', 1e-3);
gmm = fitgmdist(feature_vector', num_components, 'Options', options);

% Predict cluster labels and reshape into an image
gmm_predictions = cluster(gmm, feature_vector');
labels_img = reshape(gmm_predictions, size(img_np, 1), size(img_np, 2));
figure;
imshow(labels_img, []);
title('GMM Image Segmentation Result with K = 4');


% Parameters for cross-validation
K_folds = 10;
n_components_list = [2, 4, 6, 8, 10, 15, 20];
best_three_components = k_fold_gmm_components(K_folds, n_components_list, feature_vector);

% Visualize segmentation results
figure;
subplot(2, 2, 1);
imshow(image);
title('Original Image');

for j = 1:length(best_three_components)
    comp = best_three_components(j);
    gmm = fitgmdist(feature_vector', comp, 'Options', statset('MaxIter', 400, 'TolFun', 1e-3), 'RegularizationValue', 1e-6);
    gmm_predictions = cluster(gmm, feature_vector');
    labels_img = reshape(gmm_predictions, size(img_np, 1), size(img_np, 2));
    
    subplot(2, 2, j+1);
    imshow(labels_img, []);
    title(sprintf('Top %d with K = %d', j, comp));
end

%%
function [image_np, normalized_data] = generate_feature_vector(image)
    [rows, cols, channels] = size(image);
    [row_indices, col_indices] = ndgrid(1:rows, 1:cols);
    
    if channels == 1 % Grayscale image
        % Flatten row indices, column indices, and pixel values
        features = [row_indices(:)'; col_indices(:)'; double(image(:)')];
    elseif channels == 3 % RGB image
        % Flatten row indices, column indices, and RGB pixel values
        features = [row_indices(:)'; col_indices(:)'; ...
                    double(reshape(image(:,:,1), [], 1))'; ...
                    double(reshape(image(:,:,2), [], 1))'; ...
                    double(reshape(image(:,:,3), [], 1))'];
    else
        error('Unsupported image dimensions.');
    end
    
    % Normalize features
    min_f = min(features, [], 2);
    max_f = max(features, [], 2);
    ranges = max_f - min_f;
    normalized_data = (features - min_f) ./ ranges;
    image_np = image; % Keep original image unchanged
end

%%  K-fold cross-validation function
function best_components = k_fold_gmm_components(K, n_components_list, feature_vector)
    cv = cvpartition(size(feature_vector, 2), 'KFold', K);
    log_lld_valid_mk = zeros(length(n_components_list), K);

    for m = 1:length(n_components_list)
        num_components = n_components_list(m);
        for k = 1:K
            train_indices = training(cv, k);
            valid_indices = test(cv, k);

            % Fit GMM on training set with regularization
            gmm = fitgmdist(feature_vector(:, train_indices)', num_components, ...
                            'Options', statset('MaxIter', 400, 'TolFun', 1e-3), ...
                            'RegularizationValue', 1e-6);
            % Log-likelihood on validation set
            log_lld_valid_mk(m, k) = sum(log(pdf(gmm, feature_vector(:, valid_indices)')));
        end
    end
    
    log_lld_valid_m = mean(log_lld_valid_mk, 2);
    [~, sorted_indices] = sort(log_lld_valid_m, 'descend');
    best_components = n_components_list(sorted_indices(1:3));
    
    % Plot results
    figure;
    plot(n_components_list, log_lld_valid_m, '-o');
    title('No. Components vs Cross-Validation Log-Likelihood');
    xlabel('K');
    ylabel('Log-likelihood');
    grid on;
end
    