function L = cleanBackground(L,M)
%
% Cleans the panoramic background video with the associated per-frame
% observation masks by taking the following actions:
%   - Computes a panorama image from the video via per-pixel averages
%   - Attempts to correct non-uniform lighting effects in the panroama
%     via correctLighting()
%
% L: (ny x nx x nt): background
% M: (ny x nx x nt): frame masks
%

% Compute panorama
Lp = mean(L,3);

% Correct lighting
Lc = correctLighting(Lp,M);

% Convert back to video
L = repmat(Lc,[1, 1, size(M,3)]);
