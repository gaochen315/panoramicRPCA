function [X, stats] = tvdn(Y,lambda,varargin)
%
% Syntax:       X = tvdn(Y,lambda);
%               X = tvdn(Y,lambda,opts);
%               [X, stats] = tvdn(Y,lambda);
%               [X, stats] = tvdn(Y,lambda,opts);
%               
% Description:  Solves the subsampled total-variation with denoising (TVDN)
%               problem
%               
%               X = \argmin_{X} 0.5\|Y - X\|_F^2 + \lambda TV(X)
%               
%                   where
%                   
%               TV(X) = \sum_{ijk} (w^y_{ijk}(X_{i+1jk} - X_{ijk}) + 
%                                   w^x_{ijk}(X_{ij+1k} - X_{ijk}) + 
%                                   w^z_{ijk}(X_{ijk+1} - X_{ijk}))
%               
%               is the weighted total variation penalty that omits first
%               differences involving pixels such that M(i,j,k) = false,
%               and
%               
%                   dim = 2: omit differences between slices X(:,:,k). In
%                            other words, w^z_{ijk} == 0
%               
%                   dim = 3: include differences between slices X(:,:,k)
%               
% Inputs:       Y is an m x n x p data array
%               
%               lambda >= 0 is the total-variation (TV) regularization
%               parameter
%               
%               [OPTIONAL] opts is a struct containing one or more of the
%               following fields. The default values are in ()
%                   
%                   opts.M (true(m,n,p)) is an m x n x p logical array
%                   specifying the coordinates of X whose roughness should
%                   be penalized
%                   
%                   opts.dim (2) can be {2,3} and controls whether to
%                   compute
%                   
%                       dim = 2: 2D differences on each slice X(:,:,k) of X
%                       dim = 3: 3D differences over X
%                   
%                   opts.nIters (50) is the number of ADMM iterations to
%                   perform
%                   
%                   opts.X0 (Y) is an m x n x p array specifying the
%                   initial X iterate
%                   
%                   opts.Xtrue (nan) is the ground truth X array to use 
%                   for NRMSE calculations
%                   
%                   opts.NRMSEfcn (@computeNRMSE) is the function to use
%                   when computing the NRMSE of X after each iteration
%                   
%                   opts.rho (1) specifies the ADMM rho parameter to use
%                   
%                   opts.flag (1) determines what status updates to print
%                   to the command window. The choices are
%                   
%                       flag = 0: no printing
%                       flag = 1: print iteration status
%               
% Outputs:      X is the estimated m x n x p array
%               
%               stats is a struct containing the following fields:
%               
%                   stats.nIters is the number of iterations performed
%                   
%                   stats.cost is a 1 x nIters vector containing the cost
%                   function values at each iteration
%                   
%                   stats.nrmse is a 1 x nIters vector containing the NRMSE
%                   of X with respect to Xtrue at each iteration
%                   
%                   stats.delta is a 1 x nIters vector containing the
%                   relative convergence of X at each iteration, defined as
%                   \|X_{k + 1} - X_k\|_F / \|X_k\|_F
%                   
%                   stats.time is a 1 x nIters vector containing the time,
%                   in seconds, required to perform each iteration
%               
% Date:         December 2, 2017
%

% Parse inputs
[M, dim, nIters, X, Xtrue, NRMSEfcn, rho, flag] = ...
                                                parseInputs(Y,varargin{:});
PRINT_STATS   = (flag > 0);
COMPUTE_STATS = PRINT_STATS || (nargout == 2);
DIFFS_3D      = (dim == 3);

% Parse data
x = X(:);
y = Y(:);

% Initialize stats
if COMPUTE_STATS
    % Cost function
    Psi = @(x,WCx) 0.5 * norm(y - x)^2 + lambda * norm(WCx,1);
    
    % Stats-printing function
    iterFmt = sprintf('%%0%dd',ceil(log10(nIters + 1)));
    out     = printFcn('Iter' ,iterFmt,'cost','%.2f','nrmse','%.3f', ...
                       'delta','%.3e' ,'time','%.2fs');
    
    % Initialize stats
    cost  = nan(1,nIters);
    nrmse = nan(1,nIters);
    delta = nan(1,nIters);
    time  = nan(1,nIters);
end

% TV weights
[m, n, p] = size(M);
Wy = cat(1,double(M(2:m,:,:) & M(1:(m - 1),:,:)),zeros(1,n,p));
Wx = cat(2,double(M(:,2:n,:) & M(:,1:(n - 1),:)),zeros(m,1,p));
w = [Wy(:); Wx(:)];
if DIFFS_3D
    Wz = cat(3,double(M(:,:,2:p) & M(:,:,1:(p - 1))),zeros(m,n,1));
    w = [w; Wz(:)];
end

% First differences matrix
[C, osiz] = Cc(m,n,p,dim);

% FFT-based x-update
% Computes (I + rho * C' * C)^{-1} x
if DIFFS_3D
    % We could compute this in closed-form...
    Hf = fftn(reshape(full(C' * C(:,1)),[m, n, p]));
    
    Bf = 1 ./ (1 + rho * Hf);
    fftUpdate = @(x) real(vec(ifftn(fftn(reshape(x,[m, n, p])) .* Bf)));
else
    % We could compute this in closed-form...
    C2 = Cc(m,n,1,2);
    Hf = fft2(reshape(full(C2' * C2(:,1)),[m, n]));
    
    Bf = repmat(1 ./ (1 + rho * Hf),1,1,p);
    fftUpdate = @(x) real(vec(ifft2(fft2(reshape(x,[m, n, p])) .* Bf)));
end

% Total variation denoising
if PRINT_STATS
    fprintf('***** %dD total variation denoising *****\n',dim);
end
z = zeros(osiz);
u = zeros(osiz);
for it = 1:nIters
    % Initialize iteration
    itimer = tic;
    if COMPUTE_STATS
        % Save last iterate
        xlast = x;
    end
    
    % x update
    x = fftUpdate(y + rho * (C' * (z - u)));
    Cx = C * x;
    
    % z update
    z = soft(Cx + u,(lambda / rho) * w);
    
    % u update
    u = u + Cx - z;
    
    % Record stats
    if COMPUTE_STATS
        cost(it)  = Psi(x,w .* Cx);
        nrmse(it) = NRMSEfcn(x,Xtrue(:));
        delta(it) = computeNRMSE(x,xlast);
        time(it)  = toc(itimer);
        if PRINT_STATS
            out(it,cost(it),nrmse(it),delta(it),time(it)); 
        end
    end
end

% Reshape output 
X = reshape(x,size(Y));

% Return stats
if COMPUTE_STATS
    stats.nIters = nIters;
    stats.cost   = cost;
    stats.nrmse  = nrmse;
    stats.delta  = delta;
    stats.time   = time;
end


% Circulant dim-D first differences matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [C, osiz] = Cc(m,n,p,dim)
C = [kron(speye(p),kron(speye(n),Dc(m)));         % columns
     kron(speye(p),kron(Dc(n),speye(m)))];        % rows
if dim == 3
    C = [C; kron(Dc(p),kron(speye(n),speye(m)))]; % frames
end
osiz = [size(C,1), 1];


% Circulant 1D first differences matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function D = Dc(n)
D = spdiags([-ones(n,1), ones(n,1)],[0, 1],n,n);
D(n,1) = 1;


% Vectorize input
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x = vec(X)
x = X(:);


% Soft thresholding
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Y = soft(X,lambda)
Y = sign(X) .* max(abs(X) - lambda,0);


% Print function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function out = printFcn(varargin)
str = [sprintf('%s[%s] ',varargin{:}), '\n'];
out = @(varargin) fprintf(str,varargin{:});

% Compute NRMSE
function err = computeNRMSE(Xhat,X)
denom = norm(X(:));
if isnan(denom)
    err = nan;
elseif denom == 0
    err = 0;
else
    err = norm(Xhat(:) - X(:)) / denom;
end


% Parse inputs                                                
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [M, dim, nIters, X0, Xtrue, NRMSEfcn, rho, flag] = ...
                                                        parseInputs(Y,opts)
if ~exist('opts','var') || isempty(opts)
    opts = struct();
end

% Standard args
M        = parseField(opts,'M',true(size(Y)));
dim      = parseField(opts,'dim',2);
nIters   = parseField(opts,'nIters',50);
X0       = parseField(opts,'X0',Y);
Xtrue    = parseField(opts,'Xtrue',nan);
NRMSEfcn = parseField(opts,'NRMSEfcn',@computeNRMSE);
rho      = parseField(opts,'rho',1);
flag     = parseField(opts,'flag',1);


% Parse struct field
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function val = parseField(stats,field,default)
if isfield(stats,field)
    val = stats.(field);
else
    val = default;
end
