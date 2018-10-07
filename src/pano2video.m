function video = pano2video(frames,mask,height,width,videoSize)
%
% Transforms a panoramic grayscale video back to its native perspective.
%
%    frames: (nd x nframes) panoramic video (columns = vectorized frames)
%      mask: (nd x nframes) frame masks (columns = vectorized frames)
%    height: height of the panoramic frames
%     width: width of the panoramic frames
% videoSize: the [height, width, nframes] of the native video
%
%     video: (height x width x nframes) video from native perspective
%

video = zeros(videoSize);
for k = 1:videoSize(3)
    maskk = reshape(mask(:,k),height,width);
    [I, J] = find(maskk > max(maskk(:)) / 2);
    IJ = [I, J];
    [~, idx] = min(IJ * [1, 1; -1, -1; 1, -1; -1, 1].');
    corners = IJ(idx,:);
    
    movingPoints = corners([1, 4, 2, 3],[2, 1]);
    fixedPoints = [1, 1; 1, height; width, height; width, 1];
    Tk = fitgeotrans(movingPoints,fixedPoints,'projective');
    
    R = imref2d([height width],[1 width],[1 height]);
    imgWarped = imwarp(frames(:,:,k),R,Tk,'OutputView',R);
    video(:,:,k) = imresize(imgWarped, [videoSize(1) videoSize(2)]);
end
