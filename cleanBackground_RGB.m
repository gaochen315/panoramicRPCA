function L = cleanBackground_RGB(L,M)
%
% Syntax: L = cleanBackground_RGB(L,M);
%
% L: (ny x nx x 3 x nt): background
% M: (ny x nx x 3 x nt): frame masks
%

% Compute panorama
Lp = mean(L,4);

% Correct lighting
Lc = correctLighting(Lp,M);

% Convert back to video
L = repmat(Lc,[1, 1, 1 size(M,4)]);
