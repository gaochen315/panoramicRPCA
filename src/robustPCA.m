function [L, S, stats] = robustPCA(Y,r,lambdaS,varargin)
%
% Syntax:       [L, S] = robustPCA(Y,r,lambdaS);
%               [L, S] = robustPCA(Y,r,lambdaS,opts);
%               [L, S, stats] = robustPCA(Y,r,lambdaS);
%               [L, S, stats] = robustPCA(Y,r,lambdaS,opts);
%               
% Description:  Approximately solves the Robust PCA problem:
%               
%               \min_{L,S} 0.5\|\sqrt{M} \odot (Y - A(L + S))\|_F^2 + 
%                          \lambda_L \|L\|_{\star} + \lambda_S \|S\|_1
%               
%               using an OptShrink-based low-rank update. Note that the
%               parameter \lambda_L does not appear in this implementation
%               as the OptShrink update is not derived directly from the
%               cost function.
%               
% Inputs:       Y is an m x n observed data matrix
%               
%               r >= 0 is the desired rank of the low-rank component
%               
%               lambdaS >= 0 is the sparsity regularization parameter
%               
%               [OPTIONAL] opts is a struct containing one or more of the
%               following fields. The default values are in ()
%                   
%                   opts.A (1) is the m x d system matrix
%                   
%                   opts.M (1) is the m x n observation mask
%                   
%                   opts.T (1) is the d x d unitary sparsifying transform
%                   
%                   opts.L0 (A' * Y) is the initial low-rank iterate
%                   
%                   opts.S0 (zeros(d,n)) is the initial sparse iterate
%                   
%                   opts.accel (true) specifies whether to use Nesterov's
%                   acceleration scheme
%                   
%                   opts.tau (1) is the step size parameter. To guarantee
%                   convergence, tau should be in (0,1]. The actual step
%                   size used is
%                       tau / (2 * norm(A)^2), when accel = true
%                       tau /      norm(A)^2 , when accel = false
%                   
%                   opts.nIters (100) is the number of iterations to
%                   perform
%                   
%                   opts.flag (true) determines whether to print iteration
%                   stats to the command window
%               
% Outputs:      L is a d x n matrix containing the estimated low-rank
%               component
%               
%               S is a d x n matrix containing the estimated sparse
%               component
%               
%               stats is a statistics struct containing the following
%               fields:
%               
%                   stats.nIters is the number of iterations performed
%                   
%                   stats.deltaL is a 1 x nIters vector containing the
%                   relative convergence of the low-rank component:
%                   \|L_{k + 1} - L_k\|_F / \|L_k\|_F
%                   
%                   stats.deltaS is a 1 x nIters vector containing the
%                   relative convergence of the sparse component:
%                   \|S_{k + 1} - S_k\|_F / \|S_k\|_F
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
[A, M, T, L0, S0, accel, tau, nIters, flag] = parseInputs(Y,varargin{:});
PRINT_STATS = (flag > 0);
COMPUTE_STATS = PRINT_STATS || (nargout == 3);

% Initialize stats
if COMPUTE_STATS
    % Stats-printing function
    iterFmt = sprintf('%%0%dd',ceil(log10(nIters + 1)));
    out = printFcn('Iter',iterFmt,'deltaL','%.3e','deltaS','%.3e', ...
                   'time','%.2fs');

    % Initialize stats
    deltaL = nan(1,nIters);
    deltaS = nan(1,nIters);
    time   = nan(1,nIters);
end

% Robust PCA
t     = 0;
L     = L0;
S     = S0;
Llast = L0;
Slast = S0;
for it = 1:nIters
    % Initialize iteration
    itimer = tic;
    
    if accel
        % Accelerated proximal gradient step
        tlast = t;
        t     = 0.5 * (1 + sqrt(1 + 4 * t^2));
        Lbar  = L + ((tlast - 1) / t) * (L - Llast);
        Sbar  = S + ((tlast - 1) / t) * (S - Slast);
        Llast = L;
        Slast = S;
        Z     = A' * (M .* (A * (Lbar + Sbar) - Y));
        L     = OptShrink(Lbar - tau * Z,r);
        S     = T' * soft(T * (Sbar - tau * Z),tau * lambdaS);
    else
        % Proximal gradient step
        Llast = L;
        Slast = S;
        Z     = A' * (M .* (A * (L + S) - Y));
        L     = OptShrink(L - tau * Z,r);
        S     = T' * soft(T * (S - tau * Z),tau * lambdaS);
    end
    
    % Record stats
    if COMPUTE_STATS
        deltaL(it) = nrmse(L,Llast);
        deltaS(it) = nrmse(S,Slast);
        time(it)   = toc(itimer);
        if PRINT_STATS
            out(it,deltaL(it),deltaS(it),time(it)); 
        end
    end
end

% Return stats
if COMPUTE_STATS
    stats.nIters = nIters;
    stats.deltaL = deltaL;
    stats.deltaS = deltaS;
    stats.time   = time;
end


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
function err = nrmse(X,Xlast)
denom = norm(Xlast(:));
if denom == 0
    err = 0;
else
    err = norm(X(:) - Xlast(:)) / denom;
end


% Parse inputs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [A, M, T, L0, S0, accel, tau, nIters, flag] = parseInputs(Y,opts)
if ~exist('opts','var')
    opts = struct();
end
A        = parseField(opts,'A',1);
M        = parseField(opts,'M',1);
T        = parseField(opts,'T',1);
L0       = parseField(opts,'L0',A' * Y);
S0       = parseField(opts,'S0',zeros(size(L0)));
accel    = parseField(opts,'accel',true);
tau      = parseField(opts,'tau',1) / ((1 + accel) * norm(A)^2);
nIters   = parseField(opts,'nIters',100);
flag     = parseField(opts,'flag',true);


% Parse struct field
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function val = parseField(stats,field,default)
if isfield(stats,field)
    val = stats.(field);
else
    val = default;
end
