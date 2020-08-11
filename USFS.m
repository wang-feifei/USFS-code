function [W, Y, P, M] = USFS(X, class_num, m, alpha, beta,gamma)
%% input:
%       X(d*n): data matrix, each column is a data point
%       m: d*m projection matrix
%       class_num: the number of class class_num
%       alpha: parameter for updateing Y
%       beta: parameter for updateing W
%       gamma:parameter for updateing P
%%  output:
%       projective matrix W(d*m)
%       cluster soft label matrix Y(n*c)
%       feature selection matrix P(d*c)
%       the center matrix of clustering M(m*c)
%% initial
[d, n] = size(X);
epsil = 0.01; %
INTER = 5;
P = rand(d, class_num);
M = rand(m, class_num);
Y = initfcm(class_num, n);
Y = Y';
%% optimization
for counts = 1:INTER
    %% update W
    %     for i=1:n
    %         B(i,i) = sum(Y(i,:));
    %     end
    B = eye(n,n);
    for i=1:class_num
        D(i,i) = sum(Y(:,i));
    end
    Sw = X*(B-(Y*D*Y'))*X';
    [vec,val] = eig(Sw);
    [~,di] = sort(diag(val));
    W = vec(:,di(1:m));
    if ~isreal(W)
        W = abs(W);
    end
    %% update M
    for j=1:class_num
        aa = 0;
        for i=1:n
            aa = aa + Y(i,j)*W'*X(:,i);
        end
        M(:,j)= aa/sum(Y(:,j));
    end
    %% update P
    temp_gamma = zeros(d,1);
    for i = 1:d
        temp_gamma(i) = 1/(2*sqrt(P(i,:)*P(i,:)')+epsil);
    end
    Gamma  = diag(temp_gamma);
    P = (X*X' + (gamma/alpha)*Gamma)\X*Y;
    %% update Y
    for i = 1:n
        for j = 1:class_num
            distance(i,j) = mydist(W'*X(:,i),M(:,j));
            distance(i,j) = distance(i,j)^2;
        end
        ad = 0.5*(alpha*(X(:,i)'*P) -(distance(i,:)/(2*beta)));
        Y(i,:) = EProjSimplex_new(ad);
    end
end
end
