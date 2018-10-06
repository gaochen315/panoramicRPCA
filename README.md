# Panoramic Robust PCA

This package contains the implementation and example usages of the 
Panoramic Robust Principal Component Analysis (PRPCA) method, an algorithm
for producing robust estimates of the foreground and background of a scene
from possibly corrupted video with arbitrary camera motion.

See the [project page](https://gaochen315.github.io/pRPCA/) for more details. Please contact Chen Gao (chengao@vt.edu) if you have any questions.

<img src='Tennis_PRPCA.gif'>
<img src='Tennis_Pano.gif'>

## Quickstart

To generate some reconstructions using PRPCA, simply run the examples in
`demo.m`. The script contains examples of processing both grayscale and
color videos in the noiseless and noisy (salt and pepper) regimes.


## Contents

Data

 - `data/tennis`: 35 frames of the Tennis sequence. Color images with
    resolution 480 x 854. Also contains homographies computed by the
    `registerVideo` function for mapping each frame to the coordinates of
    the 17th frame of the sequence

Scripts

- `demo.m`: Example uses of PRPCA to perform robust foreground-background
    separation from video with arbitrary camera motion

Main Functions

- `PRPCA.m`: Main function for the PRCPA method as described in the papers
    cited below
- `PRPCA_noiseless.m`: An implementation of PRPCA intended for the
    noiseless (possibly moving camera) setting. This implementation uses
    ell-1 based regularization for the foreground and omits the total
    variation-regularized component, which is not essential when there is
    no corruption to disentangle from the foreground component

Other Functions

- `adjustLS.m`: Adjusts the low-rank and sparse components of a
    total variation-regularized foreground-background decomposition to 
    account for the constant-offset-invariance of the TV penalty
- `augRobustPCA.m`: Solves the Augmented Robust PCA problem with low-rank
    background, total variation-regularized foreground, and an addition
    sparse component to capture residual corruptions
- `cleanBackground.m`: Cleans the background component of a panoramic
    foreground-background grayscale video reconstruction
- `cleanBackground_RGB.m`: Cleans the background component of a panoramic
    foreground-background RGB video reconstruction
- `correctLighting.m`: Attempts to correct non-uniform lighting effects in
    a panoramic image
- `formatForDisplay.m`: Formats a decomposition for display by appropriately
    scaling and clamping the components to [0, 1]
- `OptShrink.m`: Implementation of the data-driven OptShrink low-rank
    matrix estimator
- `pano2video.m`: Transforms a panoramic grayscale video back to its
    native perspective
- `pano2video_RGB.m`: Transforms a panoramic RGB video back to its native
    perspective
- `PlayMovie.m`: Utility function for visualizing a data tensor as a video
- `registerVideo.m`: registers the frames of a video to a common reference
    perspective
- `robustPCA.m`: Approximately solves the Robust PCA problem using an
    OptShrink-based low-rank update
- `tvdn.m`: Solves the subsampled total-variation with denoising (TVDN)
    problem
- `writeGIF.m`: Utility function for writing a GIF image.


## License

This code is made available with the MIT license. See `LICENSE` for
details.


## References

If you use this code, we request that you cite the following papers:

```
@inproceedings{AugRPCA,
    author    = {Gao, Chen and Moore, Brian E and Nadakuditi, Raj Rao}, 
    title     = {Augmented robust PCA for foreground-background separation on noisy, moving camera video}, 
    booktitle = {Signal and Information Processing (GlobalSIP), 2017 IEEE Global Conference on}
    month     = {November},
    year      = {2017},
    pages     = {1240-1244}
}
@inproceedings{PRPCA,
    author    = {Moore, Brian E* and Gao, Chen* and Nadakuditi, Raj Rao}, 
    title     = {Panoramic Robust PCA for Foreground-Background Separation on Noisy, Free-Motion Camera Video}, 
    journal   = {arXiv:1712.06229}
    year      = {2017}
}
```
