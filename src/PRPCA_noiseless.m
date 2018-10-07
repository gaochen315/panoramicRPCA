function [pano, L, S, Lreg, Sreg] = PRPCA_noiseless(Y,varargin)
%
% Syntax:       [pano, L, S, Lreg, Sreg] = PRPCA_noiseless(Y);
%               [pano, L, S, Lreg, Sreg] = PRPCA_noiseless(Y,opts);
%               
% Description:  Performs foreground-background separation via the
%               Noiseless Panoramic Robust PCA (Noiseless PRCPA) method
%
% Inputs:       Y can be one of the following:
%                   
%                   a height x width x frames grayscale video tensor
%                   
%                   a height x width x 3 x frames RGB video tensor
%
%               [OPTIONAL] opts is a struct containing one or more of the
%               following fields. The default values are in ()
%
%                   opts.lambdaS is the sparsity regularization parameter
%                   
%                   opts.nIters (100) is the number of iterations to
%                   perform
%                   
%                   opts.cleanLS (True) determines whether apply
%                   post-processing cleanup to L and S
%                   
%                   opts.method ('temporal') is the method to use to
%                   select the anchor frame, and can be:
%                   
%                       'temporal': the frame at the midpoint of the video
%                       
%                       'spatial': the frame at the spatial midpoint of
%                       horizontal span of the registered frames
%
%                       #: the frame number to use as an anchor frame
%
% Outputs:      pano: the estimated background-only panorama
%
%               L is a height x width [x 3] x frames tensor containing the
%               estimated background (low-rank) component of the video
%
%               S is a height x width [x 3] x frames tensor containing the
%               estimated foreground (sparse) component of the video
%
%               Lreg is a tensor containing the estimated background
%               (low-rank) component of the video in the registered
%               coordinates
%
%               Sreg is a tensor containing the estimated foreground
%               (sparse) component of the video in the registered
%               coordinates
%
% Dependencies: MATLAB Computer Vision Toolbox
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

% Parse registration inputs
[method, T] = parseRegistrationInputs(varargin{:});

% Register video
[Yreg, mask, ~, height, width] = registerVideo(Y,method,T);

% Parse RPCA inputs
[lambdaS, nIters, cleanLS] = parseRPCAInputs(Yreg,varargin{:});
isRGB = ndims(Y) == 4;

% Extract active region (for efficiency)
m = any(mask,2);
Ytil = Yreg(m,:);
Mtil = mask(m,:);

% Perform Robust PCA
opts.M = Mtil;
opts.nIters = nIters;
[Ltil, Stil] = robustPCA(Ytil,1,lambdaS,opts);

% Embed components back into full space
Lhat = zeros(size(Yreg));
Shat = zeros(size(Yreg));
Lhat(m,:) = Ltil;
Shat(m,:) = Stil;

if isRGB
    %
    % Color images
    %
    
    % Generate forground/background video
    Lreg = reshape(Lhat,[height, width, 3, size(Y,4)]);
    Sreg = reshape(Shat,[height, width, 3, size(Y,4)]);
    M = logical(reshape(mask,[height, width, 3, size(Y,4)]));

    if cleanLS
        [Lreg, Sreg] = adjustLS(Lreg,Sreg,M);
    end

    L = pano2video_RGB(Lreg,mask,height,width,size(Y));
    S = pano2video_RGB(Sreg,mask,height,width,size(Y));

    % Generate panorama
    [uhat, ~, ~] = svds(reshape(Lreg,[],size(Y,4)),1);
    pano = reshape(uhat,height,width,3);
    pano = uint8(round(pano * 255 / max(pano(:))));
else
    %
    % Grayscale images
    %
    
    % Generate forground/background video
    Lreg = reshape(Lhat,[height, width, size(Y,3)]);
    Sreg = reshape(Shat,[height, width, size(Y,3)]);
    M = logical(reshape(mask,[height, width, size(Y,3)]));

    if cleanLS
        [Lreg, Sreg] = adjustLS(Lreg,Sreg,M);
        Lreg = cleanBackground(Lreg,M);
    end

    [Lreg, ~, Sreg] = formatForDisplay(Lreg,[],Sreg,M);
    L = pano2video(Lreg,mask,height,width,size(Y));
    S = pano2video(Sreg,mask,height,width,size(Y));

    % Generate panorama
    [uhat, ~, ~] = svds(reshape(Lreg,[],size(Y,3)),1);
    pano = reshape(uhat,[height, width]);
end


% Parse registration inputs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [method, T] = parseRegistrationInputs(opts)
if ~exist('opts','var')
    opts = struct();
end
method   = parseField(opts,'method','temporal');
T        = parseField(opts,'T',1);


% Parse RPCA inputs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [lambdaS, nIters, cleanLS] = parseRPCAInputs(Yreg,opts)
if ~exist('opts','var')
    opts = struct();
end
lambdaS = parseField(opts,'lambdaS',1 / sqrt(max(size(Yreg))));
nIters  = parseField(opts,'nIters',100);
cleanLS = parseField(opts,'cleanLS',true);


% Parse struct field
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function val = parseField(stats,field,default)
if isfield(stats,field)
    val = stats.(field);
else
    val = default;
end
