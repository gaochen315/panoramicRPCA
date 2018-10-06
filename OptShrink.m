function [Xhat, sX, MSE, RMSE] = OptShrink(Y,r)
%
% Syntax:       Xhat = OptShrink(Y,r);
%               [Xhat, sX] = OptShrink(Y,r);
%               [Xhat, sX, MSE, RMSE] = OptShrink(Y,r);
%               
% Inputs:       Y is an m x n matrix
%               
%               r > 0 is the desired rank
%               
% Outputs:      Xhat is an m x n rank-r matrix obtained by applying the
%               OptShrink algorithm to Y
%               
%               sX are the singular values of Xhat
%               
%               MSE is an estimate of the mean-squared error (MSE)
%               norm(Xhat - X,'fro')^2, where X is the underlying true
%               low-rank matrix
%               
%               RMSE is an estimate of the relative MSE (RMSE)
%               norm(Xhat - X,'fro')^2 / norm(X,'fro')^2, where X is the
%               underlying true low-rank matrix
%               
% References:   Nadakuditi, R., "Optshrink: An algorithm for improved
%               low-rank signal matrix denoising by optimal, data-driven
%               singular value shrinkage", IEEE Transactions on Information
%               Theory, vol. 60, no. 5, pp. 3002-3018, 2014
%               
%               Moore, B., Nadakuditi, R., Fessler, J., "Improved
%               robust PCA using low-rank denoising with optimal singular
%               value shrinkage," Proc. SSP, July 2014
%               
% Date:         February 1, 2017
%

% Parse inputs
[m, n] = size(Y);

% Compute SVD
[Uy, Sy, Vy] = svd(Y,'econ');
sY = diag(Sy);

% Optimal shrinkage
[sX, MSE, RMSE] = optimalShrinkage(sY,m,n,r);

% Construct estimate
Xhat = Uy(:,1:r) * diag(sX) * Vy(:,1:r)';


% Data-driven estimate of optimal shrinkage
% See Algorithm 1 of [1] for details
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [w, MSE, RMSE] = optimalShrinkage(s,m,n,r)

% Parse inputs
q = min(m,n);

% Signal singular values
ss = s(1:r); % column vector

% Noise singular values
sn2 = s((r + 1):q).^2; % column vector
ssH = [sn2; zeros(m - q,1)]'; % row vector
sHs = [sn2; zeros(n - q,1)]'; % row vector

% Numerical approximation of D transform
ss2        = ss.^2;
ss2mssH    = bsxfun(@minus,ss2,ssH);
ss2msHs    = bsxfun(@minus,ss2,sHs);
s1oss2mssH = sum(1 ./ ss2mssH,2);
s1oss2msHs = sum(1 ./ ss2msHs,2);
phimss     = (ss / (m - r)) .* s1oss2mssH; % \phi_m(ss)
phinss     = (ss / (n - r)) .* s1oss2msHs; % \phi_n(ss)
Dss        = phimss .* phinss; % D(ss)

% Numerical approximation of D transform derivative
phimpss = (1 / (m - r)) * ... % \phi'_m(ss)
          (sum(((-2 * bsxfun(@rdivide,ss,ss2mssH).^2)),2) + s1oss2mssH);
phinpss = (1 / (n - r)) * ... % \phi'_n(ss)
          (sum(((-2 * bsxfun(@rdivide,ss,ss2msHs).^2)),2) + s1oss2msHs); 
Dpss    = phimss .* phinpss + phinss .* phimpss; % D'(ss)

% Optimal shrinkage
w = -2 * (Dss ./ Dpss);
w(isnan(w)) = 0;

% MSE estimate
tmp = sum(1 ./ Dss);
MSE = tmp - sum(w.^2);

% RMSE estimate
RMSE = MSE / tmp;
