function [L, E, S, stats] = augRobustPCA(Y,lambdaL,lambdaE,lambdaS,varargin)
%
% Syntax:       [L, E, S] = augRobustPCA(Y,lambdaL,lambdaE,lambdaS);
%               [L, E, S] = augRobustPCA(Y,lambdaL,lambdaE,lambdaS,opts);
%               [L, E, S, stats] = augRobustPCA(Y,lambdaL,lambdaE,lambdaS);
%               [L, E, S, stats] = augRobustPCA(Y,lambdaL,lambdaE,lambdaS,opts);
%
% Description:  Solves the Augmented Robust PCA problem:
%               
%          \min_{L,E,S} 0.5\|\sqrt{M} \odot (Y - A(L + E + S))\|_F^2 + 
%                         \lambda_L \|L\|_{\star} +
%                         \lambda_E \|T E\|_1 +
%                         \lambda_S \mathbf{TV}_W(S)
%
% Inputs:       Y is an m x n observed data matrix
%               
%               lambdaL >= 0 is the low-rank regularization parameter.
%               Note that lambdaL is ignored when ~isnan(r)
%               
%               lambdaE >= 0 is the sparsity regularization parameter 
%
%               lambdaS >= 0 is the total variation regularization
%               parameter
%              
%               [OPTIONAL] opts is a struct containing one or more of the
%               following fields. The default values are in ()
%                   
%                   opts.A (1) is the m x d system matrix
%                   
%                   opts.M is the m x n observation mask
%  
%                   opts.M0 (nan) is the m x n missing data observation
%                   mask
%
%                   opts.T (1) is an optional d x d unitary sparsifying
%                   transform
%                   
%                   opts.r (nan) is the OptShrink rank parameter. When a
%                   non-nan r is specified, lambdaL is ignored and
%                   OptShrink is used in place of SVT for all L updates
%
%                   opts.dimS (2) can be {2,3} and controls whether to
%                   compute
%                   
%                       dimS = 2: 2D differences on each slice of S
%                       dimS = 3: 3D differences over S
%                   
%                   opts.rhoS (1) specifies the ADMM rho parameter to use
%                   
%                   opts.nIters (50) is the number of outer iterations to
%                   perform
%                   
%                   opts.nItersS (10) is the number of inner TV iterations
%                   to peform
%
%                   opts.L0 (A' * Y) is a p x n matrix containing the
%                   initial low-rank iterate
%                   
%                   opts.E0 (zeros(p,n)) is a p x n matrix containing the
%                   initial sparse iterate
%                   
%                   opts.S0 (zeros(p,n)) is a p x n matrix containing the
%                   initial smooth iterate
%                   
%                   opts.Xtrue (nan) is the ground truth X = L + S matrix
%                   to use for NRMSE calculations
%                   
%                   opts.NRMSEfcn (@computeNRMSE) is the function to use
%                   when computing the NRMSE of X = L + S after each
%                   iteration for the outer iterations
%                   
%                   opts.accel (true) specifies whether to use Nesterov's
%                   acceleration scheme
%                   
%                   opts.tau (0.99 + ~accel) / (3 * norm(A)^2) is the step
%                   size to use, and should satisfy
%                   
%                       tau <= 1 / (3 * norm(A)^2), when accel = true
%                       tau <  2 / (3 * norm(A)^2), when accel = falses
%                   
%                   opts.height is the height of registered video
%
%                   opts.width is the width of registered video
%                   
%                   opts.m is the embedding mask of panorama
%
%                   opts.flag (1) determines what status updates to print
%                   to the command window. The choices are
%                   
%                       flag = 0: no printing
%                       flag = 1: print outer iteration status
%                       flag = 2: print inner and outer iteration status
%               
% Outputs:      L is a p x n matrix containing the estimated low-rank
%               component
%               
%               E is a p x n matrix containing the estimated sparse
%               component
%               
%               S is a p x n matrix containing the estimated smooth
%               component
%               
%               stats is a statistics struct containing the following
%               fields:
%               
%                   stats.nIters is the number of iterations performed
%                   
%                   stats.cost is a 1 x nIters vector containing the value
%                   of the cost function at each iteration
%                   
%                   stats.nrmse is a 1 x nIters vector containing the NRMSE
%                   of X = L + S with respect to Xtrue at each iteration
%                   
%                   stats.delta is a 1 x nIters vector containing the
%                   relative convergence of X = L + S + E at each
%                   iteration, defined as \|X_{k + 1} - X_k\|_F / \|X_k\|_F
%                   
%                   stats.time is a 1 x nIters vector containing the time,
%                   in seconds, required to perform each iteration
%
% References:   B. E. Moore, C. Gao, and R. R. Nadakuditi, "Panoramic
%               robust PCA for foreground-background separation on noisy,
%               free-motion camera video," arXiv:1712.06229, 2017.
%
%               C. Gao, B. E. Moore, and R. R Nadakuditi, "Augmented
%               robust PCA for foreground-background separation on noisy,
%               moving camera video," in IEEE Global Conference on Signal
%               and Information Processing (GlobalSIP), November 2017, pp.
%               1240-1244.
%
% Date:         May 18, 2018
%

% Parse inputs
[A, M, T, W, r, dimS, rhoS, nIters, nItersS, L, E, S, Xtrue, NRMSEfcn, ...
   accel, tau, height, width, m, flag, isRGB] = parseInputs(Y,varargin{:});
PRINT_STATS = (flag > 0);
COMPUTE_STATS = PRINT_STATS || (nargout == 4);
USE_OPTSHRINK = ~isnan(r);

% Initialize stats
if COMPUTE_STATS
    % Cost function
    MC = MCm(logical(W),dimS);
    if ~USE_OPTSHRINK
        Psi = @(X,sL,E,S) 0.5 * norm(vec(sqrt(M) .* (Y - A * X)))^2 + ...
            lambdaL * sum(sL) + ...
            lambdaE * norm(vec(T * E),1) + ...
            lambdaS * norm(vec(MC * vec(S)),1);
    end
    
    % Stats-printing function
    iterFmt = sprintf('%%0%dd',ceil(log10(nIters + 1)));
    if USE_OPTSHRINK
        out = printFcn('Iter' ,iterFmt,'nrmse','%.3f', ...  % no cost
                       'delta','%.3e' ,'time','%.2fs');
    else
        out = printFcn('Iter' ,iterFmt,'cost','%.2f','nrmse','%.3f', ...
                       'delta','%.3e' ,'time','%.2fs');
    end
    
    % Initialize stats
    X     = L + E + S;
    cost  = nan(1,nIters);
    nrmse = nan(1,nIters);
    delta = nan(1,nIters);
    time  = nan(1,nIters);
end

% Total variation options
optsS.M      = W;
optsS.dim    = dimS;
optsS.nIters = nItersS;
optsS.rho    = rhoS;
optsS.flag   = flag - 1;

% Augmented Robust PCA
if accel
    % Initialize accelerated method
    t = 0;
    Llast = L;
    Elast = E;
    Slast = S;
end
for it = 1:nIters
    % Initialize iteration
    itimer = tic;
    
    if isRGB == 0
        Sorig      = zeros(height * width,size(Y,2));
        Sorig(m,:) = S;
        optsS.X0   = reshape(Sorig,height,width,[]);
    else
        Sorig      = zeros(height * width * 3,size(Y,2));
        Sorig(m,:) = S;
        optsS.X0   = reshape(Sorig,height,width,3,[]);
    end
    
    % Proximal gradient update
    if accel
        % Accelerated proximal gradient step
        tlast  = t;
        t      = 0.5 * (1 + sqrt(1 + 4 * t^2));
        Lbar   = L + ((tlast - 1) / t) * (L - Llast);
        Ebar   = E + ((tlast - 1) / t) * (E - Elast);
        Sbar   = S + ((tlast - 1) / t) * (S - Slast);
        Llast  = L;
        Elast  = E;
        Slast  = S;
        Z      = A' * (M .* (A * (Lbar + Ebar + Sbar) - Y));
        if USE_OPTSHRINK
            % OptShrink
            [L, sL] = OptShrink(Lbar - tau * Z,r);
        else
            % SVT
            [L, sL] = SVT(Lbar - tau * Z,tau * lambdaL);
        end
        E      = T' * soft(T * (Ebar - tau * Z),tau * lambdaE); 
        
        if isRGB == 0
            Sorig      = zeros(height * width,size(Y,2));
            Sorig(m,:) = Sbar - tau * Z;
            Sorig      = reshape(Sorig,height,width,[]);
            S          = tvdn(Sorig,tau * lambdaS,optsS);
            S          = reshape(S,[],size(Y,2));
            S          = S(m,:);       
        else
            Sorig      = zeros(height * width * 3,size(Y,2));
            Sorig(m,:) = Sbar - tau * Z;
            Sorig      = reshape(Sorig,height,width,3,[]);
            S          = tvdn(Sorig,tau * lambdaS,optsS);
            S          = reshape(S,[],size(Y,2));
            S          = S(m,:); 
        end
    else
        % Standard step
        Z = A' * (M .* (A * (L + S + E) - Y));
        if USE_OPTSHRINK
            % OptShrink
            [L, sL] = OptShrink(L - tau * Z,r);
        else
            % SVT
            [L, sL] = SVT(L - tau * Z,tau * lambdaL);
        end
        E = T' * soft(T * (E - tau * Z),tau * lambdaE);
        
        if isRGB == 0
            Sorig      = zeros(height * width,size(Y,2));
            Sorig(m,:) = S - tau * Z;
            Sorig      = reshape(Sorig,height,width,[]);
            S          = tvdn(Sorig,tau * lambdaS,optsS);
            S          = reshape(S,[],size(Y,2));
            S          = S(m,:);
        else
            Sorig      = zeros(height * width * 3,size(Y,2));
            Sorig(m,:) = S - tau * Z;
            Sorig      = reshape(Sorig,height,width,3,[]);
            S          = tvdn(Sorig,tau * lambdaS,optsS);
            S          = reshape(S,[],size(Y,2));
            S          = S(m,:);
        end
    end
    
    % Record stats
    if COMPUTE_STATS
        Xlast = X;
        X = L + E + S;
        if ~USE_OPTSHRINK
            cost(it) = Psi(X,sL,E,Sorig);
        end
        nrmse(it) = NRMSEfcn(X,Xtrue);        
        delta(it) = computeNRMSE(X,Xlast);
        time(it)  = toc(itimer);
        if PRINT_STATS
            if USE_OPTSHRINK
                out(it,nrmse(it),delta(it),time(it));
            else
                out(it,cost(it),nrmse(it),delta(it),time(it));
            end
        end
    end
end

% Return stats
if COMPUTE_STATS
    stats.nIters = nIters;
    stats.cost   = cost;
    stats.nrmse  = nrmse;
    stats.delta  = delta;
    stats.time   = time;
end


% Subsampled dim-D first differences matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function MC = MCm(M,dim)
[m, n, p] = size(M);
MC = sparse(0,m * n * p);
if m > 1
    Cy = kron(speye(p),kron(speye(n),Dnc(m)));  % columns
    My = M(2:m,:,:) & M(1:(m - 1),:,:);
    MC = [MC; Cy(My(:),:)];
end
if n > 1
    Cx = kron(speye(p),kron(Dnc(n),speye(m)));  % rows
    Mx = M(:,2:n,:) & M(:,1:(n - 1),:);
    MC = [MC; Cx(Mx(:),:)];
end
if (p > 1) && (dim == 3)
    Cz = kron(Dnc(p),kron(speye(n),speye(m)));  % frames
    Mz = M(:,:,2:p) & M(:,:,1:(p - 1));
    MC = [MC; Cz(Mz(:),:)];
end


% Non-circulant 1D first differences matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function D = Dnc(n)
D = spdiags([-ones(n - 1,1),ones(n - 1,1)],[0, 1],n - 1,n);


% Singular value thresholding
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Y, sY] = SVT(X,lambda)
[UX, SX, VX] = svd(X,'econ');
sY = soft(diag(SX),lambda);
Y  = UX * diag(sY) * VX';


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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function err = computeNRMSE(Xhat,X)
denom = norm(X(:));
if isnan(denom)
    err = nan;
elseif denom == 0
    err = 0;
else
    err = norm(Xhat(:) - X(:)) / denom;
end


% Vectorize input
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x = vec(X)
x = X(:);


% Parse inputs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [A, M, T, W, r, dimS, rhoS, nIters, nItersS, L0, E0, S0, ...
    Xtrue, NRMSEfcn, accel, tau, height, width, m, flag, isRGB] = ...
                                                        parseInputs(Y,opts)
if ~exist('opts','var')
    opts = struct();
end

% Standard args
A            = parseField(opts,'A',1);
M            = parseFieldr(opts,'M');
M0           = parseField(opts,'M0',nan);
T            = parseField(opts,'T',1);
r            = parseField(opts,'r',nan);
dimS         = parseField(opts,'dimS',2);
rhoS         = parseField(opts,'rhoS',1);
nIters       = parseField(opts,'nIters',200);
nItersS      = parseField(opts,'nItersS',10);
L0           = parseField(opts,'L0',nan);
E0           = parseField(opts,'E0',nan);
S0           = parseField(opts,'S0',nan);
Xtrue        = parseField(opts,'Xtrue',nan);
NRMSEfcn     = parseField(opts,'NRMSEfcn',@computeNRMSE);
accel        = parseField(opts,'accel',true);
tau          = parseField(opts,'tau',nan);
height       = parseFieldr(opts,'height');
width        = parseFieldr(opts,'width');
m            = parseFieldr(opts,'m');
flag         = parseField(opts,'flag',1);
isRGB        = parseFieldr(opts,'isRGB');

% Expensive defaults
if isnan(L0),       L0 = A' * Y;                                end
if isnan(S0),       S0 = zeros(size(L0));                       end
if isnan(E0),       E0 = zeros(size(L0));                       end
if isnan(tau),      tau = (0.99 + ~accel) / (3 * norm(A)^2);    end

if isRGB == 0
    W      = zeros(height * width,size(Y,2));
    W(m,:) = M;
    W      = reshape(W,height,width,[]);
else
    W      = zeros(height * width * 3,size(Y,2));
    W(m,:) = M;
    W      = reshape(W,height,width,3,[]);
end

% M0 is the mask of missing data.
if ~isnan(M0),      M = M0;                                     end


% Parse struct field
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function val = parseField(stats,field,default)
if isfield(stats,field)
    val = stats.(field);
else
    val = default;
end


% Parse required field
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function val = parseFieldr(stats,field)
if isfield(stats,field)
    val = stats.(field);
else
    error('Required field %s not provided',field);
end
