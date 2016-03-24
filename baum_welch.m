function [A, B, px0, c, a, b] = baum_welch(y, M, K, varargin)
% y: emissions
% M: number of states
% K: size of emission alphabet
% varargin: initializations for A, B, px0
% A: estimated transition matrix
% B: estimated emission matrix
% px0: estimated initial distribution
% c: estimated marginals/normalized gamma
% a, b: estimated alpha, beta

% choose parameter initialization
if(length(varargin) >= 1)
    A = varargin{1};
else
    A = rnd_stoch_matrix(M,M);    
end
if(length(varargin) >= 2)
    B = varargin{2};
else
    B = rnd_stoch_matrix(K,M);
end
if(length(varargin) >= 3)
    px0 = varargin{3};
else
    px0  = rnd_stoch_matrix(M,1);
end

N = length(y);
T = 1000;
e = 0.0001; % stop threshold

c = Inf;
for t=1:T
    A_old = A;
    c_old = c;
    % estimate state
    [c, a, b] = forward_backward(A, B, px0, y);
    py = sum(a(:,1) .* b(:,1));
    for i=1:N-1
        xi(:,:,i) = b(:,i+1) * a(:,i)' .* A .* repmat( B(y(i+1), :)', 1, M );
    end
    xi = xi / py; 

    % estimate parameters
    px0 = c(:,1);
    A = sum(xi, 3);
    A = A ./ repmat(sum(A), M, 1);
    for i=1:K
        B(i, :) = sum(c(:,find(y == i)), 2)';
    end
    B = B ./ repmat(sum(B), K, 1);
    if(max(max(abs(A - A_old))) < e & max(max(abs(c - c_old))) < e)
        break;
    end
end

function M = rnd_stoch_matrix(r,c)
% generates a random stochastic matrix of size r-by-c
M = rand(r,c);
M = M ./ repmat(sum(M), r, 1);