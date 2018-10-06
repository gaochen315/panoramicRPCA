function [L, S] = adjustLS(L,S,M)
%
% Adjusts the low-rank and sparse components by moving the pixelwise
% median of the sparse component into the low-rank component.
%
% The total-variation penalty is invariant to constant offset, so this
% operation helps remove any residual offset from the sparse (foreground)
% component.
%
% L: (ny x nx [x 3] x nt) background
% S: (ny x nx [x 3] x nt) foreground
% M: (ny x nx [x 3] x nt) frame masks
%

% Parse inputs
fd = ndims(L);

% Compute per-pixel adjustment
X = S;
X(~M) = nan;
Delta = nanmedian(X,fd);
Delta(~any(M,fd)) = 0;
Delta = repmat(Delta,[ones(1,fd - 1), size(L,fd)]);

% Adjust components
L = L + Delta;
S = S - Delta .* M;
