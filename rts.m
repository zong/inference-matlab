function [x_s, Pe_s, x_u, Pe_u, x_p, Pe_p, Pu] = rts(A, B, C, Pv, Pw, Px0, y)

% This implements the RTS smoothing algorithm for the system described by
%
% x(:,n+1) = A(:,:,n)x(:,n) + B(:,:,n)v(:,n)
% y(:,n) = C(:,:,n)x(:,n) + w(:,n)
%
% It calls the Kalman filter function for the forward pass, then computes
% the backward pass without the data.

% get filtered values
[x_p, Pe_p, x_u, Pe_u] = kalman_filter(A, B, C, Pv, Pw, Px0, y);

N = length(x_u);  % number of filtered values
x_s = zeros( size(x_p) );
x_s(:,N+1) = x_p(:,N+1);    % initialize x_s
Pe_s = zeros( size(Pe_u) );
Pu = zeros( size(y,1), size(y,1), N+1 );

for n=N:-1:1
    An = A(:,:,min(size(A,3),n));
    Bn = B(:,:,min(size(B,3),n));
    Cn = C(:,:,min(size(C,3),n));
    Pvn = Pv(:,:,min(size(Pv,3),n));
    Pwn = Pw(:,:,min(size(Pw,3),n));

    % calculate smoothed state estimates
    Kn_tilde = An^-1 * Bn * Pvn * Bn' * Pe_p(:,:,n+1)^-1;
    Fn_tilde = An^-1 - Kn_tilde;
    x_s(:,n) = Fn_tilde * x_s(:,n+1) + Kn_tilde * x_p(:,n+1);

    % calculate error covariances of smoothing
    Pzn = Cn * Pe_p(:,:,n) * Cn' + Pwn;
    Kn = Pe_p(:,:,n) * Cn' * Pzn^-1;
    Fn = An - An * Kn * Cn;
    Pu(:,:,n) = Fn' * Pu(:,:,n+1) * Fn + Cn' * Pzn^-1 * Cn;
    Pe_s(:,:,n) = Pe_p(:,:,n) - Pe_p(:,:,n) * Pu(:,:,n) * Pe_p(:,:,n);
end

x_s(:,end) = [];