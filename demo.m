%%
% Example uses of the Panoramic Robust PCA (PRPCA) method to perform
% robust foreground-background separation from video with arbitrary camera
% motion.
%
% May 18, 2018
%
% B. E. Moore, C. Gao, and R. R. Nadakuditi, "Panoramic robust PCA for
% foreground-background separation on noisy, free-motion camera video,"
% arXiv:1712.06229, 2017.
%
% C. Gao, B. E. Moore, and R. R Nadakuditi, "Augmented robust PCA for
% foreground-background separation on noisy, moving camera video," in
% IEEE Global Conference on Signal and Information Processing (GlobalSIP),
% November 2017, pp. 1240-1244.
%

%% Noiseless (grayscale)

% Load data
Ytrue = [];
contents = dir('data/tennis/*.jpg');
for i = 1:numel(contents)
    img = imread(sprintf('data/tennis/%s',contents(i).name));
    Ytrue(:,:,i) = im2double(imresize(rgb2gray(img),0.5)); %#ok
end

% Perform Noiseless PRPCA
opts = struct();
opts.nIters = 25;
[pano, L, S, Lreg, Sreg] = PRPCA_noiseless(Ytrue,opts);

% Visualize registered decomposition
movie = [Lreg; Sreg];
opts2 = struct();
opts2.ylabels = {'foreground', 'backgrond'};
PlayMovie(movie,opts2);

% Visualize decomposition from original persepctive
movie = [Ytrue; L; S];
opts3 = struct();
opts3.ylabels = {'foreground', 'background', 'observations'};
PlayMovie(movie,opts3);

% Visualize panorama
figure();
imshow(pano,[]);
title('Panoramic background');

%% Noiseless (color)

% Load data
Ytrue = [];
contents = dir('data/tennis/*.jpg');
for i = 1:numel(contents)
    img = imread(sprintf('data/tennis/%s',contents(i).name));
    Ytrue(:,:,:,i) = im2double(imresize(img,0.5)); %#ok
end

% Perform Noiseless PRPCA
opts = struct();
opts.nIters = 25;
[pano, L, S, Lreg, Sreg] = PRPCA_noiseless(Ytrue,opts);

% Visualize panorama
figure();
imshow(pano,[]);
title('Panoramic background');

% Play results
movie = [Lreg; Sreg];
opts2 = struct();
opts2.ylabels = {'foreground', 'background'};
PlayMovie(movie,opts2);

% Play results in original ratio
movie = [Ytrue; L; S];
opts3 = struct();
opts3.ylabels = {'foreground', 'background', 'observations'};
PlayMovie(movie,opts3);

%% Noisy (grayscale)

rng(1);

% Load homographies
load('data/tennis/Tgray.mat')

% Load data
Ytrue = [];
contents = dir('data/tennis/*.jpg');
for i = 1:numel(contents)
    img = imread(sprintf('data/tennis/%s',contents(i).name));
    Ytrue(:,:,i) = im2double(imresize(rgb2gray(img),0.5)); %#ok
end

% Generate noisy, corrupted data
p = 0.2;  % salt and pepper noise probability
M0 = (rand(size(Ytrue)) < 0.5 * p);
M1 = (rand(size(Ytrue)) < 0.5 * p);
Ynoisy = Ytrue;
Ynoisy(M0) = 1;                        
Ynoisy(M1) = 0; 
PlayMovie(Ynoisy)

% Perform PRPCA
opts = struct();
opts.nIters = 150;
opts.nItersS = 10;
opts.T = T;
opts.lambdaS = 0.02;
opts.lambdaE = 0.02;
[pano, L, E, S, Lreg, Ereg, Sreg] = PRPCA(Ynoisy,opts);

% Visualize registered decomposition
movie = [Lreg; Sreg; Ereg];
opts2 = struct();
opts2.ylabels = {'sparse', 'foreground', 'background'};
PlayMovie(movie,opts2);

% Visualize decomposition from original persepctive
movie = [Ynoisy; L; S; E; L + S];
opts3 = struct();
opts3.ylabels = {'reconstruction', 'sparse', 'foreground', 'background', 'observations'};
PlayMovie(movie,opts3);

% Visualize panorama
figure();
imshow(pano,[]);
title('Panoramic background');

% Visualize key frames of decomposition
LS = L + S;
keyframes = [
    [Ytrue(:,:,1);  Ynoisy(:,:,1);  L(:,:,1) ; E(:,:,1);  S(:,:,1);  LS(:,:,1)] ...
    [Ytrue(:,:,17); Ynoisy(:,:,17); L(:,:,17); E(:,:,17); S(:,:,17); LS(:,:,17)]
];
figure();
imshow(keyframes);
title('Key frames of decomposition');

%% Noisy (color)
 
rng(1);

% Load homographies
load('data/tennis/Tcolor.mat')

% Load data
Ytrue = [];
contents = dir('data/tennis/*.jpg');
for i = 1:numel(contents)
    img = imread(sprintf('data/tennis/%s',contents(i).name));
    Ytrue(:,:,:,i) = im2double(imresize(img,0.5)); %#ok
end

% Generate noisy, corrupted data
p = 0.2;  % salt and pepper noise probability
M0 = (rand(size(Ytrue)) < 0.5 * p);
M1 = (rand(size(Ytrue)) < 0.5 * p);
Ynoisy = Ytrue;
Ynoisy(M0) = 1;                        
Ynoisy(M1) = 0; 

% Perform PRPCA
opts = struct();
opts.nIters = 150;
opts.nItersS = 10;
opts.T = T; 
opts.lambdaS = 0.02;
opts.lambdaE = 0.02;
[pano, L, E, S, Lreg, Ereg, Sreg] = PRPCA(Ynoisy,opts);

% Visualize registered decomposition
movie = [Lreg; Sreg; Ereg];
opts2 = struct();
opts2.ylabels = {'sparse', 'foreground', 'background'};
PlayMovie(movie,opts2);

% Visualize decomposition from original persepctive
movie = [Ynoisy; L; S; E; L + S];
opts3 = struct();
opts3.ylabels = {'reconstruction', 'sparse', 'foreground', 'backgrond', 'observations'};
PlayMovie(movie,opts3);

% Visualize panorama
figure();
imshow(pano,[]);
title('Panoramic background');

% Visualize key frames of decomposition
LS = L + S;
keyframes = [
    [Ytrue(:,:,:,1);  Ynoisy(:,:,:,1);  L(:,:,:,1);  E(:,:,:,1);  S(:,:,:,1);  LS(:,:,:,1)] ...
    [Ytrue(:,:,:,17); Ynoisy(:,:,:,17); L(:,:,:,17); E(:,:,:,17); S(:,:,:,17); LS(:,:,:,17)]
];
figure();
imshow(keyframes);
title('Key frames of decomposition');

