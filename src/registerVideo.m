function [Yreg, mask, T, height, width] = registerVideo(Y,method,T)
%
% Registers the frames of the input video to a common reference
% perspective.
%
%       Y: (m x n x t) input video
%  method: 'temporal', 'spatial', or <#>
%       T: (nt x 1) per-frame homographies
%
%    Yreg: (nd x nt) registered video frames (vectorized as columns)
%    mask: (nd x nt) registered frame masks (vectorized as columns)
%       T: (nt x 1) per-frame homographies
%  height: the height of the regsitered frames
%   width: the width of the regsitered frames
%

% Parse inputs
isRGB = ndims(Y) == 4;

if ~isa(T,'projective2d')
    % Compute homographies
    
    % Read first frame
    if isRGB
        nf = size(Y,4);
        img = rgb2gray(Y(:,:,:,1));
    else
        nf = size(Y,3);
        img = Y(:,:,1);
    end
    
    % Process first frame
    points = detectSURFFeatures(img);
    [features, points] = extractFeatures(img,points);
    T = repmat(projective2d(eye(3)),nf,1);
    
    % Process remaining frames
    for k = 2:nf
        pointsPrev = points;
        featuresPrev = features;
        
        % Read frame
        if isRGB
            img = rgb2gray(Y(:,:,:,k));
        else
            img = Y(:,:,k);
        end
        
        % Process frame
        points = detectSURFFeatures(img);
        [features, points] = extractFeatures(img,points);
        indexPairs = matchFeatures(features,featuresPrev,'Unique',true);
        matchedPoints = points(indexPairs(:,1),:);
        matchedPointsPrev = pointsPrev(indexPairs(:,2),:);
        T(k) = estimateGeometricTransform( ...
            matchedPoints,matchedPointsPrev,'projective','Confidence',...
            99.9,'MaxNumTrials',2000);
        
        % Compute transformation w.r.t. first frame
        T(k).T = T(k - 1).T * T(k).T;
    end
    
    imgSize = size(img);
    for k = 1:nf
        [xlims(k,:), ylims(k,:)] = outputLimits( ...
            T(k),[1, imgSize(2)],[1, imgSize(1)]); %#ok
    end
    
    if isnumeric(method)
        anchorIdx = method;
    elseif strcmp('temporal',method)
        anchorIdx = floor(nf / 2);
    elseif strcmp('spatial',method)
        avgXLim = mean(xlims,2);
        avgYLim = mean(ylims,2);
        avgLim = [avgXLim, avgYLim];
        center = mean(avgLim,1);
        distance = sum((avgLim - repmat(center,size(avgLim,1),1)) .^ 2,2);
        [~, inds] = sort(distance);
        anchorIdx = inds(1);        
    else
        error('Unsupported method %s',method);
    end
    
    % Map homographies to anchor frame
    Tinv = invert(T(anchorIdx));
    for k = 1:numel(T)
        T(k).T = Tinv.T * T(k).T;
    end
else
    % Use provided homographies
    if isRGB
        img = rgb2gray(Y(:,:,:,1));
    else
        img = Y(:,:,1);
    end
    imgSize = size(img);
end

% Compute output view
for k = 1:numel(T)
    [xlims(k,:), ylims(k,:)] = outputLimits( ...
        T(k),[1, imgSize(2)],[1, imgSize(1)]);
end
xmin = floor(min([1; xlims(:)]));
xmax = ceil(max([imgSize(2); xlims(:)]));
ymin = floor(min([1; ylims(:)]));
ymax = ceil(max([imgSize(1); ylims(:)]));
width = xmax - xmin;
height = ymax - ymin;
oview = imref2d([height, width],[xmin, xmax],[ymin, ymax]);

% Register frames
Yreg = [];
mask = [];
if isRGB
    panorama = zeros([height, width, 3],'like',img);
else
    panorama = zeros([height, width],'like',img);
end
blender = vision.AlphaBlender( ...
    'Operation','Binary mask','MaskSource','Input port');
for k = 1:numel(T)
    if isRGB
        img = Y(:,:,:,k);
        warpedImg = imwarp(img,T(k),'OutputView',oview);
        Yreg(:,k) = warpedImg(:); %#ok
        panorama = step(blender,panorama,warpedImg,warpedImg(:,:,1,1));
        Mk = imwarp(ones(size(img)),T(k),'OutputView',oview);
        Mk(Mk ~= 0) = 1;
        mask(:,k) = Mk(:); %#ok
    else
        img = Y(:,:,k);
        warpedImg = imwarp(img,T(k),'OutputView',oview);
        Yreg(:,k) = warpedImg(:); %#ok
        panorama = step(blender,panorama,warpedImg,warpedImg(:,:,1));
        Mk = imwarp(ones(size(img)),T(k),'OutputView',oview);
        Mk(Mk ~= 0) = 1;
        mask(:,k) = Mk(:); %#ok 
    end
end
