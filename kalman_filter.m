function [x_p, Pe_p, x_u, Pe_u] = kalman_filter(A, B, C, Pv, Pw, Px0, y)

% This implements the DT Kalman filter for the system described by
%
% x(:,n+1) = A(:,:,n)x(:,n) + B(:,:,n)v(:,n)
% y(:,n) = C(:,:,n)x(:,n) + w(:,n)
%
% where Pv(:,:,n), Pw(:,:,n) are the covariances of v(:,n) and w(:,n)
% and Px0 is the initial state covariance.
%
% v(:,n), w(:,n), x(:,1) are assumed to be zero-mean.
%
% Return values are
% x_p: state estimates given the past 
% Pe_p: error covariance estimates given the past
% x_u: state updates given the data
% Pe_u: error covariance updates given the data

N = length(y);  % number of time samples in the data
x_p = zeros( size(A,2), N+1 );
x_u = zeros( size(A,2), N );
Pe_p = zeros( size(A,2), size(A,2), N+1 );
Pe_u = zeros( size(A,2), size(A,2), N );
x_p(:,1) = 0;
Pe_p(:,:,1) = Px0;

for n=1:N
    [x_u(:,n), Pe_u(:,:,n)] = ...
        kalman_update(x_p(:,n), Pe_p(:,:,n), ...
        C(:,:,min(size(C,3),n)), Pw(:,:,min(size(Pw,3),n)), y(:,n));
    [x_p(:,n+1), Pe_p(:,:,n+1)] = ...
        kalman_predict(x_u(:,n), Pe_u(:,:,n), ...
        A(:,:,min(size(A,3),n)), B(:,:,min(size(B,3),n)), ...
        Pv(:,:,min(size(Pv,3),n)));
end
    
function [x_u, Pe_u] = kalman_update(x_p, Pe_p, C, Pw, y)
% The Kalman update step that finds the state estimate based on new data
    G = Pe_p * C' * (C * Pe_p * C' + Pw)^-1;
    x_u = x_p + G * (y - C * x_p);
    Pe_u = Pe_p - G * C * Pe_p;
end

function [x_p, Pe_p] = kalman_predict(x_u, Pe_u, A, B, Pv)
% The Kalman prediction step that implements the tracking system
    x_p = A * x_u;
    Pe_p = A * Pe_u * A' + B * Pv * B';
end
