function [particles,weights] = particle_filter(f,G,h,Pv,Pw,particles0,y,threshold)

% This implements the particle filter specialized for the system given by:
% x(t+1) = f(x(t)) + G*v(t)
% y(t) = h(x(t)) + w(t)
% where state and measurement are scalar, and driving noise and
% measurement noise are Gaussian.
%
% Non-linear (also time-invariant) function f and h are passed 
% in as inlined matlab functions (see the 'inline' command).
%
% Pv, Pw: variances of v(t) and w(t)
% particles0: initial particles
% y: observed data
% threshold: resampling threshold (N_eff/N)
%
% Returned values are
% particles, weights: estimated final particles and weights

N = size(particles0,1); % number of particles
T = length(y);  % number of data samples

particles = zeros(N,T+1);
weights = zeros(size(particles));

% initialization
particles(:,1) = particles0;
weights(:,1) = 1/N;

for t = 1:T
    % update Step
    updated_weights = weights(:,t) .* ...
        (1/sqrt(2*pi*Pw) * exp(-(y(t)-h(particles(:,t))).^2/(2*Pw)));
    updated_weights = updated_weights./sum(updated_weights);
    
    % resampling 
    if (1/sum(updated_weights.^2))/N < threshold
        disp(['resamp: t = ' num2str(t)]);
        cutoff = cumsum(updated_weights);
        resamp_idx = zeros(N,1);
        for j = 1:N
            resamp_idx(j) = N-sum(rand<cutoff)+1;
        end
        resamp_particles = particles(resamp_idx,t);
        updated_weights = ones(N,1)/N;
    else
        resamp_particles = particles(:,t);
    end
    
    % prediction Step
    particles(:,t+1) = f(resamp_particles) + G*(sqrt(Pv)*randn(N,1)); % simulate the Markov process
    weights(:,t+1) = updated_weights; % the weights are not modified through prediction step 
end
