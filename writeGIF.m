function writeGIF(X,filename,fps)
%
% Sytnax:       writeGIF(X,filename);
%               writeGIF(X,filename,fps);
%               
% Description:  Writes the input data to a GIF file.
%               
% Inputs:       X can be one of the following:
%               
%               (a) An m x n x 1 x p cube containg p frames of grayscale
%                   data, each with height m pixels and width n pixels
%               
%               (b) A m x n x 3 x p hypercube containing p frames of RGB
%                   data, each with height m pixels and width n pixels
%               
%               See imwrite() for valid class types and data ranges for X
%               
%               filename is the desired output filename
%               
%               [OPTIONAL] fps is the desired frame rate of the GIF. The
%               default value is fps = 12
%               
% Example:      % Random GIF
%               X = rand(64,64,3,15);
%               writeGIF(X,'random.gif',5);
%               
% Date:         December 5, 2016
%

% Parse inputs
if ~exist('fps','var') || isempty(fps)
    % Default fps
    fps = 12;
end

% Write GIF
p = size(X,4);
[X1, map1] = convertToIndexedImage(X(:,:,:,1));
imwrite(X1,map1,filename,'gif','DelayTime',1 / fps, ...
                               'WriteMode','overwrite', ...
                               'LoopCount',inf);
for i = 2:p
    [Xi, mapi] = convertToIndexedImage(X(:,:,:,i));
    imwrite(Xi,mapi,filename,'WriteMode','append', ...
                             'DelayTime',1 / fps);
end


% Convert to indexed image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [X, map] = convertToIndexedImage(I)
[X, map] = rgb2ind(I,256);
