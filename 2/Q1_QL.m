clear; close all; clc;

%--------------------------------------------------------------------------
%% Preparation
%--------------------------------------------------------------------------
if (0)
    N = [20 200 2000 10000]; % Number of samples
    p = [0.6 0.4]; % Class priors
    m01 = [-0.9; -1.1]; m02 = [0.8; 0.75];
    m11 = [-1.1; 0.9]; m12 = [0.9; -0.75];
    C = [0.75 0; 0 1.25]; % Covariance matrix
    labels = cell(1, length(N));
    data = cell(1, length(N));
    Ncounts = cell(1, length(N)); % Cell array to store Ncount for each iteration

    for i = 1:length(N)
        label = rand(1, N(i)) >= p(1);
        labels{i} = label;
        Ncount = [sum(label == 0), sum(label == 1)];
        Ncounts{i} = Ncount; % Store Ncount in the cell array
        x = zeros(2, N(i));
        x(:, label == 0) = [mvnrnd(m01, C, ceil(Ncount(1)/2)); mvnrnd(m02, C, ceil(Ncount(1)/2))]';
        x(:, label == 1) = [mvnrnd(m11, C, ceil(Ncount(2)/2)); mvnrnd(m12, C, ceil(Ncount(2)/2))]';
        data{i} = x;
    end

else
    load('/Users/ln915/Documents/Lvx/2024_Fall/Intro2ML/Assignments/EECE_5644_QL/2/data/data_for_q1.mat')
end
%--------------------------------------------------------------------------
%% Part 1: Theoretically Optimal Classifier (Log-likelihood Ratio)
%--------------------------------------------------------------------------
discriminator = log(0.5 * evalGaussian(data{4}, m11, C) + 0.5 * evalGaussian(data{4}, m12, C)) ...
    - log(0.5 * evalGaussian(data{4}, m01, C) + 0.5 * evalGaussian(data{4}, m02, C));

% Sort the log-likelihood ratio for thresholding
[sortedScores, ind] = sort(discriminator, 'ascend');
thresholdList = [min(sortedScores) - eps, (sortedScores(1:end-1) + sortedScores(2:end)) / 2, max(sortedScores) + eps];

n_gamma = length(thresholdList); % Number of thresholds

TPR = zeros(1, n_gamma); % True Positive Rate, y-axis of the ROC curve
FPR = zeros(1, n_gamma); % False Positive Rate, x-axis of the ROC curve
P_error = zeros(1, n_gamma); % Classification error

for i = 1: n_gamma
    decision = (discriminator >= thresholdList(i));
    TP = sum(decision == 1 & labels{4} == 1); % True Positives
    FP = sum(decision == 1 & labels{4} == 0); % False Positives
    FN = sum(decision == 0 & labels{4} == 1); % False Negatives
    TN = sum(decision == 0 & labels{4} == 0); % True Negatives
    TPR(i) = TP / (TP + FN);
    FPR(i) = FP / (FP + TN);
    P_error(i) = (FP + FN) / N(4);
end

% Plot ROC curve
figure(1);
plot(FPR, TPR, '-o');
xlabel('False Positive Rate (FPR)');
ylabel('True Positive Rate (TPR)');
title('ROC Curve');
grid on; hold on;

% Find the threshold that minimizes P_error
[min_Perror, best_idx] = min(P_error);
gamma1_best = thresholdList(best_idx);

% Mark the best point on the ROC curve (min-P(error))
plot(FPR(best_idx), TPR(best_idx), 'kp', 'MarkerSize', 15, 'MarkerFaceColor', 'g');
hold on;

disp(['Minimum P(error): ', num2str(100*min_Perror), '%']);
disp(['Best gamma: ', num2str(gamma1_best)]);
disp(['theoretically optimal gamma: ', num2str(log(p(1)/p(2)))]);

x=data{4};
label=labels{4};
decision = (discriminator >= gamma1_best);
ind00 = find(decision==0 & label==0);
ind10 = find(decision==1 & label==0);
ind01 = find(decision==0 & label==1);
ind11 = find(decision==1 & label==1);

figure(2), % class 0 circle, class 1 +, correct green, incorrect red
plot(x(1,ind00),x(2,ind00),'o', 'MarkerEdgeColor',  [0 1 0]); hold on,
plot(x(1,ind10),x(2,ind10),'o', 'MarkerEdgeColor',  [1 0 0]); hold on,
plot(x(1,ind01),x(2,ind01),'+', 'MarkerEdgeColor',  [1 0 0]); hold on,
plot(x(1,ind11),x(2,ind11),'+', 'MarkerEdgeColor',  [0 1 0]); hold on,
axis equal;

% Draw the decision boundary
horizontalGrid = linspace(floor(min(data{4}(1,:))),ceil(max(data{4}(1,:))),101);
verticalGrid = linspace(floor(min(data{4}(2,:))),ceil(max(data{4}(2,:))),91);
[h,v] = meshgrid(horizontalGrid,verticalGrid);
lambda = [0 1;1 0];
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * p(1)/p(2);
discriminantScoreGridValues = log(0.5 * evalGaussian([h(:)';v(:)'], m11, C) + 0.5 * evalGaussian([h(:)';v(:)'], m12, C)) ...
    - log(0.5 * evalGaussian([h(:)';v(:)'], m01, C) + 0.5 * evalGaussian([h(:)';v(:)'], m02, C))- log(gamma);
minDSGV = min(discriminantScoreGridValues);
maxDSGV = max(discriminantScoreGridValues);
discriminantScoreGrid = reshape(discriminantScoreGridValues,91,101);
contour(horizontalGrid,verticalGrid,discriminantScoreGrid,[minDSGV*[0.9,0.6,0.3],0,[0.3,0.6,0.9]*maxDSGV]);
legend('Correct decisions for data from Class 0','Wrong decisions for data from Class 0','Wrong decisions for data from Class 1','Correct decisions for data from Class 1','Equilevel contours of the discriminant function' ),
title('Data and their classifier decisions versus true labels');
xlabel('x_1')
ylabel('x_2')

%--------------------------------------------------------------------------
%% Part 2 Maximum likelihood parameter estimation
%--------------------------------------------------------------------------

% logstic-linear-fuction-based-approximation
hfunc = @(x,w) 1./(1 + exp(-w' * x));
cf = @(n, l, x, theta) (-1/n) * sum(l.*log(hfunc(x, theta)) + (1-l).*log(1-hfunc(x, theta))); % Cross-entropy-based loss
z = cell(1, length(N));
for i = 1:length(N)
    z{i} = [ones(N(i),1), data{i}']'; % Adding bias term
end
initial_theta = zeros(size(p,2)+1, 1);
theta = cell(1, 3);
cost = cell(1, 3);
decision = cell(1, 3);

for i = 1:3
    [theta{i}, cost{i}] = fminsearch(@(t)(cf(N(i), labels{i}, z{i}, t)), initial_theta);
    decision{i} = hfunc(z{i}, theta{i}) >= 0.5; % Using hfunc for logistic output and comparing with 0.5 threshold
end

figure(3)
for i = 1:3
    subplot(3,1,i),
    e = plot_train(labels{i}, decision{i}, Ncounts{i}, p, data{i});
    title("Train dataset with " + N(i) + " samples with probability error of " + e + "%"), hold off
end

zxv1 = [ones(N(4),1), data{4}']'; % Validation dataset
validation_decision = cell(1, 3);

for i = 1:3
    validation_decision{i} = hfunc(zxv1, theta{i}) >= 0.5; % Classification on validation set
    figure(i+3)
    e = plot_train(labels{4}, validation_decision{i}, Ncounts{4}, p, data{4});
    title("Validation dataset with " + N(i) + " Samples as training dataset with probability error of " + e + "%"), hold off
end

%% logstic-quadratic-fuction-based-approximation
z_quad = cell(1, length(N));
for i = 1:length(N)
    tmp=data{i};
    z_quad{i} = [ones(N(i),1), tmp', tmp(1,:)'.*tmp(1,:)', tmp(1,:)'.*tmp(2,:)', tmp(2,:)'.*tmp(2,:)' ]';
end

initial_theta2 = zeros(6, 1);
theta2 = cell(1, 3); 
cost2 = cell(1, 3);  

for i = 1:3
    [theta2{i}, cost2{i}] = fminsearch(@(t)(cf(N(i), labels{i}, z_quad{i}, t)), initial_theta2);
end

decision2 = cell(1, 3);
for i = 1:3
    decision2{i} = hfunc(z_quad{i}, theta2{i}) >= 0.5;
end

figure(7)
for i = 1:3
    subplot(3,1,i),
    e = plot_train(labels{i}, decision2{i}, Ncounts{i}, p, data{i});
    title("Train dataset with " + N(i) + " samples with probability error of " + e + "%"), hold off
end
tmp=data{4};
zxv2 = [ones(N(4),1), tmp', tmp(1,:)'.*tmp(1,:)', tmp(1,:)'.*tmp(2,:)', tmp(2,:)'.*tmp(2,:)' ]'; % Validation dataset
validation_decision2 = cell(1, 3);

for i = 1:3
    validation_decision2{i} = hfunc(zxv2, theta2{i}) >= 0.5; % Classification on validation set
    figure(i+7)
    e = plot_train(labels{4}, validation_decision2{i}, Ncounts{4}, p, data{4});
    title("Validation dataset with " + N(i) + " Samples as training dataset with probability error of " + e + "%"), hold off
end



function g = evalGaussian(x, mu, Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each column of X
[n, N] = size(x);
C = ((2 * pi)^n * det(Sigma))^(-1/2);
E = -0.5 * sum((x - repmat(mu, 1, N)) .* (inv(Sigma) * (x - repmat(mu, 1, N))), 1);
g = C * exp(E);
end

function P_error = plot_train(label, decision, nc, p, x)
i00 = find(decision==0 & label==0);
i10 = find(decision==1 & label==0);
i01 = find(decision==0 & label==1);
i11 = find(decision==1 & label==1);
P_error = 100*(sum(decision==1 & label==0)+  sum(decision==0 & label==1))/sum(nc);

plot(x(1, i00),x(2, i00), 'o', 'MarkerEdgeColor',  [0 1 0]);hold on
plot(x(1, i10),x(2, i10), 'o', 'MarkerEdgeColor', [1 0 0]);hold on
plot(x(1, i01),x(2, i01), '+','MarkerEdgeColor', [1 0 0]);hold on
plot(x(1, i11),x(2, i11), '+','MarkerEdgeColor', [0 1 0]);hold on
legend('Class 0 Correct Decisions','Class 0 Wrong Decisions','Class 1 Wrong Decisions','Class 1 Correct Decisions');
end


