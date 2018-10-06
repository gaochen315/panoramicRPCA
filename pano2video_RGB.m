function video = pano2video_RGB(frames,mask,height,width,videoSize)
%
% Transforms a panoramic RGB video back to its native perspective.
%
%    frames: (nd x nframes) panoramic video (columns = vectorized frames)
%      mask: (nd x nframes) frame masks (columns = vectorized frames)
%    height: height of the panoramic frames
%     width: width of the panoramic frames
% videoSize: the [height, width, 3, nframes] of the native video
%
%     video: (height x width x 3 x nframes) video from native perspective
%

video = zeros(videoSize);
for k = 1:videoSize(4)
    maskk = reshape(mask(1:height * width,k),height,width);
    [I, J] = find(maskk > max(maskk(:)) / 2);
    IJ = [I, J];
    [~, idx] = min(IJ * [1, 1; -1, -1; 1, -1; -1, 1].');
    corners = IJ(idx,:);
    
    movingPoints = corners([1, 4, 2, 3],[2, 1]);
    fixedPoints = [1, 1; 1, height; width, height; width, 1];
    Tk = fitgeotrans(movingPoints,fixedPoints,'projective');
    
    R = imref2d([height, width],[1, width],[1, height]);
    imgWarped = imwarp(frames(:,:,:,k),R,Tk,'OutputView',R);
    video(:,:,1,k) = imresize(imgWarped(:,:,1),videoSize(1:2));
    video(:,:,2,k) = imresize(imgWarped(:,:,2),videoSize(1:2));
    video(:,:,3,k) = imresize(imgWarped(:,:,3),videoSize(1:2));
end
