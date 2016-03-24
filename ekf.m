function [x_p, Pe_p, x_u, Pe_u] = ekf(f,df,G,h,dh,Pv,Pw,x0,P0,y)

% This implements the extended Kalman filter specialized for the system: 
% x(:,t+1) = f(x(:,t)) + G*v(:,t)
% y(:,t) = h(x(:,t)) + w(:,t)
%
% Non-linear (also time-invariant) functions f, df, h, dh are
% passed in as inlined matlab functions (see the 'inline' command).
%
% df, dh: derivatives of f and h, respectively
% Pv, Pw: covariances of v(:,t) and w(:,t)
% x0: initial state
% P0: initial state covariance
% y: observed data
%
% Returned values are
% x_p: state estimates given the past
% Pe_p: error covariance estimates given the past
% x_u: state updates given the data
% Pe_u: error covariance updates given the data

s = size(x0,1);	% state size
T = length(y);	% number of time samples in the data

x_p = zeros(s,T+1);
Pe_p = zeros(s,s,T+1);

% initialization
x_p(:,1) = x0;
Pe_p(:,:,1) = P0;

for t = 1:T
    % update step
    K = Pe_p(:,:,t)*dh(x_p(:,t))'* ...
        inv(dh(x_p(:,t))*Pe_p(:,:,t)*dh(x_p(:,t))' + Pw);
    x_u = x_p(:,t) + K*(y(:,t) - h(x_p(:,t)));
    Pe_u = Pe_p(:,:,t) - K*dh(x_p(:,t))*Pe_p(:,:,t);
    
    % prediction step
    x_p(:,t+1) = f(x_u);
    Pe_p(:,:,t+1) = df(x_u)*Pe_u*df(x_u)' + G*Pv*G';
end
