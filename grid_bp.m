function B = grid_bp(npot, epot_v, epot_h, varargin)

% This implements loopy BP on a 2D grid version of the HMM graph, 
% with discrete states, suitable for image processing tasks.
%
% The grid has h x w nodes, and M is the number of states
% npot: node potentials, h x w
%
% epot_v: vertical edge potentials, (h-1) x w x M x M
% epot_h: horizontal edge potentials, h x (w-1) x M x M
%
% The 3rd and 4th dimensions of epot_v and epot_h form a transition matrix
% where the transition on the graph is defined from a top to bottom and left to right.
%
% Optional parameters:
% N: number of iterations to run
% alpha: message update regularization factor
% 
% Returns:
% B, the belief at each node, size h x w x M.

if(length(varargin) >= 1)
    N = varargin{1};
else
    N = 30;
end
if(length(varargin) >= 2)
    alpha = varargin{2};
else
    alpha = 1;
end
[height, width, M] = size(npot);

% Initialize messages: M{dir}, where {dir} = d (down), u (up), r (right), l
% (left)
Md=ones(height-1,width,M)/M;
Mu=ones(height-1,width,M)/M;
Mr=ones(height,width-1,M)/M;
Ml=ones(height,width-1,M)/M;
Md1=Md;
Mu1=Mu;
Mr1=Mr;
Ml1=Ml;

for k=1:N
    % pass messages
    disp(['Iteration ' num2str(k)]);

    % top to bottom
    temp=npot(1:end-1,:,:);
    temp(2:end,:,:)=temp(2:end,:,:).*Md(1:end-1,:,:);
    temp(:,2:end,:)=temp(:,2:end,:).*Mr(1:end-1,:,:);
    temp(:,1:end-1,:)=temp(:,1:end-1,:).*Ml(1:end-1,:,:);
    for j=1:M
        Md1(:,:,j) = sum(squeeze(epot_v(:,:,j,:)) .* temp, 3);
    end

    % bottom to top
    temp=npot(2:end,:,:);
    temp(1:end-1,:,:)=temp(1:end-1,:,:).*Mu(2:end,:,:);
    temp(:,2:end,:)=temp(:,2:end,:).*Mr(2:end,:,:);
    temp(:,1:end-1,:)=temp(:,1:end-1,:).*Ml(2:end,:,:);
    for j=1:M
        Mu1(:,:,j) = sum(epot_v(:,:,:,j) .* temp, 3);
    end

    % left to right
    temp=npot(:,1:end-1,:);
    temp(:,2:end,:)=temp(:,2:end,:).*Mr(:,1:end-1,:);
    temp(2:end,:,:)=temp(2:end,:,:).*Md(:,1:end-1,:);
    temp(1:end-1,:,:)=temp(1:end-1,:,:).*Mu(:,1:end-1,:);
    for j=1:M
        Mr1(:,:,j) = sum(squeeze(epot_h(:,:,j,:)) .* temp, 3);
    end
    
    % right to left
    temp=npot(:,2:end,:);
    temp(:,1:end-1,:)=temp(:,1:end-1,:).*Ml(:,2:end,:);
    temp(2:end,:,:)=temp(2:end,:,:).*Md(:,2:end,:);
    temp(1:end-1,:,:)=temp(1:end-1,:,:).*Mu(:,2:end,:);
    for j=1:M
        Ml1(:,:,j) = sum(epot_h(:,:,:,j) .* temp, 3);
    end

    % normalize new messages
    temp=repmat(sum(Md1,3),[1,1,M]);
    Md1=Md1./temp;
    temp=repmat(sum(Mu1,3),[1,1,M]);
    Mu1=Mu1./temp;
    temp=repmat(sum(Mr1,3),[1,1,M]);
    Mr1=Mr1./temp;
    temp=repmat(sum(Ml1,3),[1,1,M]);
    Ml1=Ml1./temp;

    % update messsage
    Md=Md1*alpha+Md*(1-alpha);
    Mu=Mu1*alpha+Mu*(1-alpha);
    Mr=Mr1*alpha+Mr*(1-alpha);
    Ml=Ml1*alpha+Ml*(1-alpha);
    
    % compute beliefs
    B=npot;
    B(2:end,:,:)=B(2:end,:,:).*Md;
    B(1:end-1,:,:)=B(1:end-1,:,:).*Mu;
    B(:,2:end,:)=B(:,2:end,:).*Mr;
    B(:,1:end-1,:)=B(:,1:end-1,:).*Ml;
    
    temp=repmat(sum(B,3),[1,1,M]);
    B=B./temp;
end
