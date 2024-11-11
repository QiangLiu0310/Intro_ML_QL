clear;close all;clc;
N=10;
figure(1), clf,
delta = 1e-2;
regWeight = 1e-10;

alpha_true = [0.2, 0.25, 0.25, 0.3];
mu_true = [-10, -5, 0, 10; 0, 0, 0, 0];
Sigma_true(:,:,1) = [3 1; 1 20];
Sigma_true(:,:,2) = [5 1; 1 5];
Sigma_true(:,:,3) = [5 1; 1 5];
Sigma_true(:,:,4) = [4 -1; 1 8];

x = randGMM(N,alpha_true,mu_true,Sigma_true);
[d,Mtrue] = size(mu_true)

M = 1;

alpha = ones(1,M)/M;
shuffledIndices = randperm(N);
mu = x(:,shuffledIndices(1:M)); 
[~,assignedCentroidLabels] = min(pdist2(mu',x'),[],1); 
for m = 1:M
    Sigma(:,:,m) = cov(x(:,find(assignedCentroidLabels==m))') + regWeight*eye(d,d);
end
t = 0;

Converged = 0; 
while ~Converged
    for l = 1:M
        temp(l,:) = repmat(alpha(l),1,N).*evalGaussian(x,mu(:,l),Sigma(:,:,l));
    end
    plgivenx = temp./sum(temp,1);
    alphaNew = mean(plgivenx,2);
    w = plgivenx./repmat(sum(plgivenx,2),1,N);
    muNew = x*w';
    for l = 1:M
        v = x-repmat(muNew(:,l),1,N);
        u = repmat(w(l,:),d,1).*v;
        SigmaNew(:,:,l) = u*v' + regWeight*eye(d,d); % adding a small regularization term
    end
    Dalpha = sum(abs(alphaNew-alpha'));
    Dmu = sum(sum(abs(muNew-mu)));
    DSigma = sum(sum(abs(abs(SigmaNew-Sigma))));
    Converged = ((Dalpha+Dmu+DSigma)<delta); % Check if converged
    alpha = alphaNew; mu = muNew; Sigma = SigmaNew;
    t = t+1;
    displayProgress(t,x,alpha,mu,Sigma);
end

function displayProgress(t,x,alpha,mu,Sigma)
figure(1),
if size(x,1)==2
    subplot(1,2,1), cla,
    plot(x(1,:),x(2,:),'b.');
    xlabel('x_1'), ylabel('x_2'), title('Data and Estimated GMM Contours'),
    axis equal, hold on;
    rangex1 = [min(x(1,:)),max(x(1,:))];
    rangex2 = [min(x(2,:)),max(x(2,:))];
    [x1Grid,x2Grid,zGMM] = contourGMM(alpha,mu,Sigma,rangex1,rangex2);
    contour(x1Grid,x2Grid,zGMM); axis equal,
    subplot(1,2,2),
end
logLikelihood = sum(log(evalGMM(x,alpha,mu,Sigma)));
plot(t,logLikelihood,'b.'); hold on,
xlabel('Iteration Index'), ylabel('Log-Likelihood of Data'),
drawnow; pause(1),
end

function x = randGMM(N,alpha,mu,Sigma)
d = size(mu,1); % dimensionality of samples
cum_alpha = [0,cumsum(alpha)];
u = rand(1,N); x = zeros(d,N); labels = zeros(1,N);
for m = 1:length(alpha)
    ind = find(cum_alpha(m)<u & u<=cum_alpha(m+1));
    x(:,ind) = randGaussian(length(ind),mu(:,m),Sigma(:,:,m));
end
end

function x = randGaussian(N,mu,Sigma)
n = length(mu);
z =  randn(n,N);
A = Sigma^(1/2);
x = A*z + repmat(mu,1,N);
end

function [x1Grid,x2Grid,zGMM] = contourGMM(alpha,mu,Sigma,rangex1,rangex2)
x1Grid = linspace(floor(rangex1(1)),ceil(rangex1(2)),101);
x2Grid = linspace(floor(rangex2(1)),ceil(rangex2(2)),91);
[h,v] = meshgrid(x1Grid,x2Grid);
GMM = (evalGMM([h(:)';v(:)'],alpha, mu, Sigma));
zGMM = reshape(GMM,91,101);
end

function gmm = evalGMM(x,alpha,mu,Sigma)
gmm = zeros(1,size(x,2));
for m = 1:length(alpha)
    gmm = gmm + alpha(m)*evalGaussian(x,mu(:,m),Sigma(:,:,m));
end
end

function g = evalGaussian(x,mu,Sigma)
[n,N] = size(x);
invSigma = inv(Sigma);
C = (2*pi)^(-n/2) * det(invSigma)^(1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(invSigma*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end
