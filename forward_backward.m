function [c, a, b] = forward_backward(A, B, px0, y)
% A: transition matrix
% B: emission matrix
% px0: initial state probability
% y: emissions
% c: marginals, i.e. normalized gamma
% a, b: alpha and beta

N = length(y);

a = zeros(size(px0,1), N);
b = a;
c = a;
% forwards pass
a(:,1) = diag(B(y(1),:)) * px0;
for i=1:N-1
    a(:,i+1) = sum(diag(B(y(i+1),:)) * A * diag(a(:,i)), 2);
end

% backwards pass
b(:,N) = 1;
for i=N-1:-1:1
    b(:,i) = sum(diag(B(y(i+1),:)) * diag(b(:,i+1)) * A, 1)';
end

% compute marginals
c = a .* b;
c = c / sum(c(:,1));