function L = correctLighting(L,M)
%
% Attempts to correct non-uniform lighting effects in the given panorama
% by iteratively computing scaling factors to apply to the pixels in the 
% difference of the supports between consecutive frame masks
%
% L: (ny x nx) panorama
% M: (ny x nx x nt) frame masks
%

% Parse inputs
nt = size(M,3);
cIdx = round(0.5 * nt);

% Sort frames by x-midpoint
idx = sortFrames(M);
M = M(:,:,idx);

% Correct lighting
for i = (cIdx - 1):-1:1
    % center -> left
    L = deltaCorrect(L,M(:,:,i),M(:,:,i + 1));
end
for i = (cIdx + 1):nt
    % center -> right
    L = deltaCorrect(L,M(:,:,i),M(:,:,i - 1));
end


% Correct lighting of frame-delta
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function L = deltaCorrect(L,Mnew,Mref)
%    L: (ny x nx) panorama
% Mnew: (ny x nx) new frame mask
% Mref: (ny x nx) reference frame mask

% Get edge pixels
newIdx = find(Mnew & ~Mref);
[refIdx, adjIdx] = getBorder(Mref);
useIdx = ismember(adjIdx,newIdx);
ref = L(refIdx(useIdx)); % Reference pixels
adj = L(adjIdx(useIdx)); % Adjacent pixels

% Correct lighting
if ~isempty(ref)
    alpha = computeAlpha(ref,adj);
    L(newIdx) = alpha * L(newIdx);
end


% Compute alpha
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function alpha = computeAlpha(ref,adj)
% ref:   (p x 1) reference pixels
% adj:   (p x 1) adjacent pixels
% alpha: (1 x 1) correction factor for adjacent region

% Possible cost functions
%cost = @(a) mean((ref - a * adj).^2);
cost = @(a) mean(abs(ref - a * adj));
%cost = @(a) median(ref - a * adj);

% Compute alpha
alpha = fminsearch(cost,1);


% Get border
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [refIdx, adjIdx] = getBorder(M)
%      M: (ny x nx) frame mask
% refIdx:   (p x 1) linear indices of reference (border) pixels
% adjIdx:   (p x 1) linear indices of adjacent (out-of-mask) pixels

% Parse inputs
[ny, nx] = size(M);
xg = (1:nx)';
yg = (1:ny)';

% Upper border
[~, yuIdx] = max(M,[],1);
uIdx = sub2ind([ny, nx],yuIdx(:),xg);
uIdx = uIdx(yuIdx > 1);

% Bottom border
[~, ybIdx] = max(flipud(M),[],1);
ybIdx = ny + 1 - ybIdx;
bIdx = sub2ind([ny, nx],ybIdx(:),xg);
bIdx = bIdx(ybIdx < ny);

% Left border
[~, xlIdx] = max(M,[],2);
lIdx = sub2ind([ny, nx],yg,xlIdx(:));
lIdx = lIdx(xlIdx > 1);

% Right border
[~, yrIdx] = max(fliplr(M),[],2);
yrIdx = nx + 1 - yrIdx;
rIdx = sub2ind([ny, nx],yg,yrIdx(:));
rIdx = rIdx(yrIdx < nx);

% Reference indices
refIdx = [uIdx; rIdx; bIdx; lIdx];

% Adjacent indices
adjIdx = [(uIdx - 1); (rIdx + ny); (bIdx + 1); (lIdx - ny)];


% Sort frames
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function idx = sortFrames(M)
%   M: (ny x nx x nt) frame masks
% idx:       (1 x nt) sorted (left -> right) frame indices

% x-extents
Mx = squeeze(any(M,1));

% Left edge
[~, le] = max(Mx,[],1);

% Right edge
[~, re] = max(flipud(Mx),[],1);
re = size(Mx,1) + 1 - re;

% Frame centers
mm = 0.5 * (le + re);

% Sort frames from left to right
[~, idx] = sort(mm);
