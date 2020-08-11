clc;
clear;
load('COIL20.mat')
warning('off');
X = fea';label = gnd;
class_num = length(unique(label));
[d, n] = size(X);
rand('twister',5489);
FeaNumCandi = 20;
alpha = 10000;
gamma = 10000;
beta = 1;
m = 45;
%% run SLUFS
[W, Y, P, M] = USFS(X, class_num, m, alpha, beta,gamma);
W1 = [];
for k = 1:d
    W1 = [W1 norm(P(k,:),2)];
end
[~,index] = sort(W1,'descend');
new_fea = X(index(1:FeaNumCandi),:);
idx = kmeans(new_fea', class_num);
result = ClusteringMeasure(label, idx);