function [pano, L, E, S, Lreg, Ereg, Sreg] = PRPCA(Y,varargin)
%
% Syntax:       [pano, L, E, S, Lreg, Ereg, Sreg] = PRPCA(Y);
%               [pano, L, E, S, Lreg, Ereg, Sreg] = PRPCA(Y,opts);
%               
% Description:  Performs foreground-background separation via the
%               Panoramic Robust PCA (PRCPA) method
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
%                   opts.lambdaL is the low-rank regularization parameter.
%                   Note that lambdaL is ignored when ~isnan(r)
%
%                   opts.lambdaE is the sparsity regularization parameter
%                   
%                   opts.lambdaS is the TV regularization parameter
%
%                   opts.r (1) is the OptShrink rank parameter. When a
%                   non-nan r is specified, lambdaL is ignored and
%                   OptShrink is used in place of SVT for all L updates
%                   
%                   opts.nIters (200) is the number of outer iterations to
%                   perform
%                   
%                   opts.nItersS (10) is the number of inner TV iterations
%                   to peform
%
%                   opts.dimS (2) can be {2,3} and controls whether to
%                   compute
%                   
%                       dimS = 2: 2D differences on each slice of S
%                       dimS = 3: 3D differences over S
%
%                   opts.flag (1) determines what status updates to print
%                   to the command window. The choices are
%                   
%                       flag = 0: no printing
%                       flag = 1: print outer iteration status
%                       flag = 2: print inner and outer iteration status
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
%               E is a height x width [x 3] x frames tensor containing the
%               estimated sparse component of the video
%
%               S is a height x width [x 3] x frames tensor containing the
%               estimated foreground (smooth) component of the video
%
%               Lreg is a tensor containing the estimated background
%               (low-rank) component of the video in the registered
%               coordinates
%
%               Ereg is a tensor containing the estimated sparse component
%               of the video in the registered coordinates
%
%               Sreg is a tensor containing the estimated foreground
%               (smooth) component of the video in the registered
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
[lambdaL, lambdaE, lambdaS, r, nIters, nItersS, dimS, flag, cleanLS] ...
                                       = parseRPCAInputs(Yreg,varargin{:});
isRGB = ndims(Y) == 4;

% Extract active region (for efficiency)
m = any(mask,2);
Ytil = Yreg(m,:);
Mtil = mask(m,:);

% Perform Augmented Robust PCA
opts.M       = Mtil;
opts.nIters  = nIters;  
opts.nItersS = nItersS;
opts.height  = height;
opts.width   = width;
opts.m       = m;
opts.dimS    = dimS; 
opts.flag    = flag;
opts.r       = r;
opts.isRGB   = isRGB;
[Ltil, Etil, Stil] = augRobustPCA(Ytil,lambdaL,lambdaE,lambdaS,opts);

% Embed components back into full space
Lhat = zeros(size(Yreg));
Ehat = zeros(size(Yreg));
Shat = zeros(size(Yreg));
Lhat(m,:) = Ltil;
Ehat(m,:) = Etil;
Shat(m,:) = Stil;

if isRGB
    %
    % Color images
    %
    
    % Generate forground/background video
    Lreg = reshape(Lhat,[height, width, 3, size(Y,4)]);
    Ereg = reshape(Ehat,[height, width, 3, size(Y,4)]);
    Sreg = reshape(Shat,[height, width, 3, size(Y,4)]);
    M = logical(reshape(mask,[height, width, 3, size(Y,4)]));

    [Lreg, Sreg] = adjustLS(Lreg,Sreg,M);
    
    L = pano2video_RGB(Lreg,mask,height,width,size(Y));
    E = pano2video_RGB(Ereg,mask,height,width,size(Y));
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
    Ereg = reshape(Ehat,[height, width, size(Y,3)]);
    Sreg = reshape(Shat,[height, width, size(Y,3)]);
    M = logical(reshape(mask,[height, width, size(Y,3)]));

    if cleanLS
        [Lreg, Sreg] = adjustLS(Lreg,Sreg,M);
        Lreg = cleanBackground(Lreg,M);
    end

    [Lreg, Ereg, Sreg] = formatForDisplay(Lreg,Ereg,Sreg,M);
    L = pano2video(Lreg,mask,height,width,size(Y));
    E = pano2video(Ereg,mask,height,width,size(Y));
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
T        = parseField(opts,'T',nan);


% Parsed RPCA inputs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [lambdaL, lambdaE, lambdaS, r, nIters, nItersS, dimS, flag, ...
                                      cleanLS] = parseRPCAInputs(Yreg,opts)
if ~exist('opts','var')
    opts = struct();
end
lambdaL  = parseField(opts,'lambdaL',[]);
lambdaE  = parseField(opts,'lambdaE',9.5 / sqrt(max(size(Yreg))));
lambdaS  = parseField(opts,'lambdaS',9.5 / sqrt(max(size(Yreg))));
r        = parseField(opts,'r',1);
nIters   = parseField(opts,'nIters',200);
nItersS  = parseField(opts,'nItersS',10);
dimS     = parseField(opts,'dimS',2);
flag     = parseField(opts,'flag',1);
cleanLS  = parseField(opts,'cleanLS',true);


% Parse struct field
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function val = parseField(stats,field,default)
if isfield(stats,field)
    val = stats.(field);
else
    val = default;
end
