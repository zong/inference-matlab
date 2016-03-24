function [xhat, cost] = hmm_viterbi(A, B, px0, y)
% A: transition matrix
% B: emission matrix
% px0: initial state probability
% y: emissions
% xhat: most likely state path
% cost: final minimal cost

N = length(y);
M = size(px0, 1);
paths = zeros(M,N-1); % defined by states transited
costs = zeros(M,N); % associated with the paths
S = -log(A);
T = -log(B);

% cost minimization
costs(:,1) = -log(px0); % costs of the initial states
for i=1:N-1
    costs(:,i) = costs(:,i) + T(y(i),:)';
    [costs(:,i+1), paths(:,i)] = min( (repmat(costs(:,i),1,M) + S')', [], 2 );
end
[cost, xhat(N)] = min(costs(:,N) + T(y(N),:)');

% backtracking
for i=N-1:-1:1
    xhat(i) = paths(xhat(i+1),i);
end