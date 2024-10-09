% Espected Risk Minimization
% Qiang Liu 20240929
clear; close all; clc;
%--------------------------------------------------------------------------
%% Preparation
%--------------------------------------------------------------------------
if (0)
    n = 4; % number of feature dimensions
    N = 10000; % number of iid samples
    m(:,1) = [-1;-1;-1;-1];
    m(:,2) = [1;1;1;1];
    C(:,:,1) = [2 -0.5 0.3 0; -0.5 1 -0.5 0; 0.3 -0.5 1 0; 0 0 0 2];
    C(:,:,2) = [1 0.3 -0.2 0; 0.3 2 0.3 0; -0.2 0.3 1 0; 0 0 0 3];
    p = [0.35,0.65]; % class priors for labels 0 and 1 respectively
    label = rand(1,N) >= p(1); % True label
    Ncount = [length(find(label==0)), length(find(label==1))]; % number of samples from each class
    x = zeros(n,N);
    x(:,label==0) = mvnrnd(m(:,1),C(:,:,1),Ncount(1))';
    x(:,label==1) = mvnrnd(m(:,2),C(:,:,2),Ncount(2))';
    save('erm_test_data.mat') % save the work space for the following questions
end

%--------------------------------------------------------------------------
%% Part A 1
%--------------------------------------------------------------------------

load('/Users/ln915/Documents/Lvx/2024_Fall/Intro2ML/Assignments/1/data/erm_test_data.mat')

lambda=[0 1; 1 0];% 0-1 loss, theoretical
gamma0 = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2))*p(1)/p(2);

discriminator=evalGaussian(x,m(:,2),C(:,:,2))./ evalGaussian(x,m(:,1),C(:,:,1));
[sortedScores,ind] = sort(discriminator,'ascend');
thresholdList = [min(sortedScores)-eps,(sortedScores(1:end-1)+sortedScores(2:end))/2, max(sortedScores)+eps];

gamma1=thresholdList;
n_gamma = length(gamma1);

%--------------------------------------------------------------------------
%% Part A 2
%--------------------------------------------------------------------------

TPR = zeros(1, n_gamma); % True Positive Rate, y-axis of the ROC curve
FPR = zeros(1, n_gamma); % False Positive Rate, x-axis of the ROC curve
P_error=zeros(1, n_gamma); 

for i = 1: n_gamma
    % gamma1(i)=gamma0;
    decision = (discriminator >= gamma1(i));
    TP = sum(decision == 1 & label == 1); % True Positives
    FP = sum(decision == 1 & label == 0); % False Positives
    FN = sum(decision == 0 & label == 1); % False Negatives
    TN = sum(decision == 0 & label == 0); % True Negatives
    TPR(i) = TP / (TP + FN); % record the current value
    FPR(i) = FP / (FP + TN);
    % P_error(i) = [FP FN]*Ncount'/N;
        P_error(i)=(FP+FN)/N;
end

figure(1);% Plot ROC Curve
plot(FPR, TPR, '-o');
xlabel('False Positive Rate (FPR)');
ylabel('True Positive Rate (TPR)');
title('ROC Curve');
grid on;
hold on;

%--------------------------------------------------------------------------
%% Part A 3
%--------------------------------------------------------------------------

best_gamma=find(P_error==min(P_error));
gamma1_best = gamma1(best_gamma);

for i=1:size(best_gamma)
plot( FPR(best_gamma(i)),  TPR(best_gamma(i)), 'kp', 'MarkerSize', 15, 'MarkerFaceColor', 'g'); 
hold on;
end

min_Perror=min(P_error);
%--------------------------------------------------------------------------
%% Part B
%--------------------------------------------------------------------------
clearvars -except gamma1_best FPR(best_gamma)  TPR(best_gamma)  min_Perror

load('/Users/ln915/Documents/Lvx/2024_Fall/Intro2ML/Assignments/1/data/erm_test_data.mat')

C_incorrect=C.*diag(ones(1, n));
discriminator=evalGaussian(x,m(:,2),C_incorrect(:,:,2))./ evalGaussian(x,m(:,1),C_incorrect(:,:,1));

[sortedScores,ind] = sort(discriminator,'ascend');
thresholdList = [min(sortedScores)-eps,(sortedScores(1:end-1)+sortedScores(2:end))/2, max(sortedScores)+eps];
gamma1=thresholdList;

n_gamma = length(gamma1);

TPR_new = zeros(1, n_gamma); % True Positive Rate, y-axis of the ROC curve
FPR_new = zeros(1, n_gamma); % False Positive Rate, x-axis of the ROC curve
P_error=zeros(1, n_gamma); 

for i = 1: n_gamma
    decision = (discriminator >= gamma1(i));
    TP = sum(decision == 1 & label == 1); % True Positives
    FP = sum(decision == 1 & label == 0); % False Positives
    FN = sum(decision == 0 & label == 1); % False Negatives
    TN = sum(decision == 0 & label == 0); % True Negatives
    TPR_new(i) = TP / (TP + FN); % record the current value
    FPR_new(i) = FP / (FP + TN);
    % P_error(i) = [FP FN]*Ncount'/N; % [0 1] fix it
    P_error(i)=(FP+FN)/N;
end

figure(2);% Plot ROC Curve
plot(FPR_new, TPR_new, '-o');
xlabel('False Positive Rate (FPR)');
ylabel('True Positive Rate (TPR)');
title('ROC Curve with incorrect Covariance ');
grid on;
hold on;

best_gamma=find(P_error==min(P_error));
gamma2_best = gamma1(best_gamma);

for i=1:size(best_gamma)
plot( FPR_new(best_gamma(i)),  TPR_new(best_gamma(i)), 'kp', 'MarkerSize', 15, 'MarkerFaceColor', 'r'); 
hold on;
end

min_Perror_new = min(P_error);

%--------------------------------------------------------------------------
%% Part C Linear Discriminant Analysis LDA 
%--------------------------------------------------------------------------
load('/Users/ln915/Documents/Lvx/2024_Fall/Intro2ML/Assignments/1/data/erm_test_data.mat')

mu1hat = mean(x(:,label==0),2);
mu2hat = mean(x(:,label==1),2);
Sb=(mu1hat-mu2hat)*(mu1hat-mu2hat)';
Sw=cov(x(:,label==0)')+cov(x(:,label==1)'); % equal weights

[V, D] = eig(Sw \ Sb);
[~, max_idx] = max(diag(D));  % Find the eigenvector corresponding to the largest eigenvalue
w_LDA = V(:, max_idx);        % Fisher LDA projection vector

if (mean(w_LDA' * x(:,label==0))>mean(w_LDA' * x(:,label==1)))
    w_LDA=-w_LDA;
end

y = (w_LDA)' * x;  

discriminator=y;

[sortedScores,ind] = sort(discriminator,'ascend');
thresholdList = [min(sortedScores)-eps,(sortedScores(1:end-1)+sortedScores(2:end))/2, max(sortedScores)+eps];

% thresholdList=[-1000:1:1000];% my threshold for plotting ROC curve
n_gamma = length(thresholdList);

TPR = zeros(1, n_gamma); % True Positive Rate, y-axis of the ROC curve
FPR = zeros(1, n_gamma); % False Positive Rate, x-axis of the ROC curve
P_error=zeros(1, n_gamma); 
true_positives = [];
false_positives = [];

for i = 1: n_gamma
    decision = (discriminator>=thresholdList(i));
    TP = sum(decision == 1 & label == 1); % True Positives
    FP = sum(decision == 1 & label == 0); % False Positives
    FN = sum(decision == 0 & label == 1); % False Negatives
    TN = sum(decision == 0 & label == 0); % True Negatives
    TPR(i) = TP / (TP + FN); % record the current value
    FPR(i) = FP / (FP + TN);
    P_error(i)=(FP+FN)/N;
end

figure(3);% Plot ROC Curve
plot(FPR, TPR, 'o');
xlabel('False Positive Rate (FPR)');
ylabel('True Positive Rate (TPR)');
title('ROC Curve for Fisher LDA Classifier (Equal Class Weights)');
grid on;
hold on;

best_gamma=find(P_error==min(P_error));
gamma3_best = thresholdList(best_gamma);

for i=1:size(best_gamma)
plot( FPR(best_gamma(i)),  TPR(best_gamma(i)), 'kp', 'MarkerSize', 15, 'MarkerFaceColor', 'y'); 
hold on;
end
min_Perror_3 = min(P_error);


function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
C = ((2*pi)^n * det(Sigma))^(-1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end







