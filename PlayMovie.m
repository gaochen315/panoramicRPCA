function varargout = PlayMovie(movie,opts)
%
% Syntax:       PlayMovie(movie);
%               PlayMovie(movie,opts);
%               P = PlayMovie(movie);
%               P = PlayMovie(movie,opts);
%               
% Inputs:       movie is a struct with the following fields
%               
%                   movie.video can be one of the following:
%                   
%                       An (ny x nx x nt) cube containg nt frames, each
%                       with height ny pixels and width nx pixels. The data
%                       is auto-scaled to [0 1] and shown on grayscale
%                       
%                       An (ny x nx x 3 x nt) hypercube containing nt color
%                       frames, each with height ny pixels, width nx pixels
%                       and RGB values specified along the third dimension.
%                       If class(video) == 'double', the values should be
%                       in [0 1]. Otherwise, class(video) should be 'uint8'
%                       and the values should be in [0 255]
%                   
%        [OPTIONAL] movie.Fv is the video frame rate. The default value is
%                   Fv = 24
%                   
%        [OPTIONAL] movie.sound is either (i) a (1 x Ns) vector of audio
%                   samples, or (ii) a (1 x Na) cell array whose ith entry
%                   contains the (1 x Nsi) vector of audio samples for the
%                   ith audio stream
%                   
%        [OPTIONAL] movie.Fa is a (1 x Na) whose ith entry contains the
%                   audio sampling rate of the ith audio stream.
%                   
%                       NOTE: When sound data is provided, Fa and Fv *MUST*
%                             be specified
%                   
%        [OPTIONAL] movie.audioLabels is either (i) a string, or (ii) a 
%                   (1 x Na) cell array of strings containing the names of
%                   the audio streams. The default value is
%                   audioLabels = {'Stream 1','Stream 2',...}
%                   
%               opts is a struct with the following fields:
%                   
%        [OPTIONAL] opts.repeat = {true false} determines whether to play
%                   the movie on repeat. The default value is repeat =
%                   true
%                   
%        [OPTIONAL] opts.mag is a scalar magnification factor. The default
%                   value is mag = 1
%                   
%        [OPTIONAL] opts.dim = [width height] are the desired frame
%                   dimensions during playback, in pixels. When either
%                   value is NaN, the appropriate aspect ratio-preserving
%                   value is used
%                   
%        [OPTIONAL] opts.xlabels is either (i) a string, or (ii) a cell
%                   array of strings to display centered (and evenly
%                   spaced, if necessary) above/below the movie. The
%                   default value is xlabels = {}
%                   
%        [OPTIONAL] opts.ylabels is either (i) a string, or (ii) a cell
%                   array of strings to display centered (and evenly
%                   spaced, if necessary) to left/right of movie. The
%                   default value is ylabels = {}
%                   
%        [OPTIONAL] opts.fontSize specifies the font size (in points) to
%                   use for the x/y labels. The default value is
%                   fontSize = 12
%                   
%        [OPTIONAL] opts.gap specifies the label gap (in pixels) to use for
%                   the x/y labels. The default value is gap = 15
%                   
% Outputs:      P is a struct of methods that support external GUI control.
%               The supported methods are:
%               
%                      P.Start();           % Start movie
%                      P.Stop();            % Stop movie
%                      P.Beginning();       % Go to beginning of movie
%                      P.End();             % Go to end of movie
%                      P.Repeat(bool);      % Set repeat flag
%               bool = P.IsPlaying();       % Return playing status
%                      P.SetFrame(idx);     % Set movie frame
%                      P.SetFrameRate(Fv);  % Set frame rate
%                      P.SetAudio(idx);     % Set audio stream
%                      P.AudioOff();        % Turn off audio
%                      P.Close();           % Close player
%                      P.SaveMovie(path);   % Save movie to file
%                      P.SaveGIF(path);     % Save movie as GIF
%                      P.SaveFrame(path);   % Save current frame to file
%               
% Description:  This function plays back the input data as a movie
%               
% Example:      % Knobs
%               T  = 8;                             % Movie length
%               Fv = 24;                            % Video frame rate
%               
%               % Load music
%               load handel;
%               
%               % Generate movie frames
%               A = [0.25 0.50; 0.75 1.00];
%               B = rand(128,128,3,T * Fv);
%               M = bsxfun(@times,kron(A,ones(64)),B);
%               
%               % Movie data
%               movie.video = M;                    % Movie frames
%               movie.Fv    = Fv;                   % Video frame rate
%               movie.sound = y(1:(T * Fs));        % Audio samples
%               movie.Fa    = Fs;                   % Audio sampling rate
%               
%               % Options
%               opts.mag     = 5;                   % 5x magnification
%               opts.xlabels = {'Left' 'Right'};    % x-labels
%               opts.ylabels = {'Bottom' 'Top'};    % y-labels
%               
%               % Play movie
%               PlayMovie(movie,opts);
%               
% Hot Keys:     {Backspace,Enter,Delete}    Start/stop movie playback
%               {Left,Up} Arrow             Back one frame
%               {Right,Down} Arrow          Advance one frame
%               
% Mouse:        Mouse click                 Start/stop movie playback
%               Scroll up                   Back one frame
%               Scroll down                 Advance one frame
%               
% Date:         December 6, 2016
%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Knobs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Video constants
FPS_MEMORY = 5;                     % Frame rate calculation memory

% Choose which label mode you prefer:
%   (a) FIXED                       % Label size is fixed
%   (b) NORMALIZED                  % Labels grow prop. to figure height
FIXED       = 1;
NORMALIZED  = 2;
LABEL_MODE  = FIXED;
%LABEL_MODE = NORMALIZED;

% Label constants
switch LABEL_MODE
    case FIXED
        % Constant-sized labels
        DEFAULT_GAP       = 15;     % Default label gap (pixels)
        DEFAULT_FONT_SIZE = 12;     % Default font size (points)
    case NORMALIZED
        % Normalized labels
        DEFAULT_GAP       = 0.05;   % Default label gap (normalized units)
        DEFAULT_FONT_SIZE = 0.8;    % Default font size (% of gap)
end

% Misc constants
AUDIOPLAYER_LATENCY = 0.025;        % Audioplayer latency
DEFAULT_ALOCK       = true;         % Default aspect ratio lock
NCOLORS             = 256;          % Number of grayscale colors (<= 256)
CM                  = @gray;        % Grayscale colormap
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parse inputs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parse video
if ~isstruct(movie)
    % Convert video matrix to struct
    movie = struct('video',movie);
end
if ~isreal(movie.video)
    % Convert complex-valued data to magnitude
    movie.video = abs(movie.video);
end
[ny, nx, sz3, sz4] = size(movie.video);
if sz4 == 1
    % Grayscale image
    isColorImage = false;
    args = {':', ':'};
    nt   = sz3;
    
    % Scale data
    M = double(movie.video);
    M = M - min(M(:));
    M = floor(NCOLORS * (M / max(M(:))));
    M(M == NCOLORS) = (NCOLORS - 1);
    movie.video = M;
    clear M;
else
    % Color image
    isColorImage = true;
    args = {':' ':' ':'};
    nt   = sz4;
end

% Parse video frame rate
if ~isfield(movie,'Fv') || isempty(movie.Fv)
    % Default frame rate
    movie.Fv = 24;
end

% Parse opts
if (nargin < 2)
    opts = struct();
end
if isfield(opts,'repeat') && ~isempty(opts.repeat)
    % User-specified repeat flag
    repeat = opts.repeat;
else
    % Default repeat flag
    repeat = true;
end
if isfield(opts,'mag') && ~isempty(opts.mag)
    % Magnification factor
    dim = round(opts.mag * [nx, ny]);
elseif isfield(opts,'dim') && ~isempty(opts.dim) && ~all(isnan(opts.dim))
    % Dimensions specified
    if isnan(opts.dim(1))
        % Compute aspect-preserving width
        dim = opts.dim(2) * [round(nx / ny), 1];
    elseif isnan(opts.dim(2))
        % Compute aspect-preserving height
        dim = opts.dim(1) * [1, round(ny / nx)];
    else
        % User-specified dimensions
        dim = opts.dim;
    end
else
    % Default dimensions
    dim = [nx, ny];
end
if isfield(opts,'fontSize') && ~isempty(opts.fontSize)
    % User-specified font size
    fontSize = opts.fontSize;
else
    % Default font size
    fontSize = DEFAULT_FONT_SIZE;
end
if isfield(opts,'gap') && ~isempty(opts.gap)
    % User-specified gap
    gap = opts.gap;
else
    % Default gap
    gap = DEFAULT_GAP;
end

% Parse xlabels/ylabels
haveXlabels = isfield(opts,'xlabels') && ~isempty(opts.xlabels);
haveYlabels = isfield(opts,'ylabels') && ~isempty(opts.ylabels);
haveLabels  = [haveYlabels, haveXlabels];
switch LABEL_MODE
    case FIXED
        % Fixed-size labels
        fw = dim + 2 * gap * haveLabels;
    case NORMALIZED
        % Normalized labels
        fw = dim .* (1 + 2 * gap * haveLabels);
end

% Parse sound
if isfield(movie,'sound')
    % Parse audio streams
    if ~iscell(movie.sound)
        movie.sound = {movie.sound};
    end
    Na       = numel(movie.sound);
    audioobj = cell(1,Na);
    Ns       = zeros(1,Na);
    for i = 1:Na
        % Initialize ith audio player
        Ns(i)       = length(movie.sound{i});
        audioobj{i} = audioplayer(movie.sound{i},movie.Fa(i)); %#ok
    end
    
    % Parse stream labels
    if ~isfield(movie,'audioLabels') || isempty(movie.audioLabels)
        movie.audioLabels = {};
    elseif ischar(movie.audioLabels)
        movie.audioLabels = {movie.audioLabels};
    end
    for i = (numel(movie.audioLabels) + 1):Na
        movie.audioLabels{i} = sprintf('Stream %i',i);
    end
else
    % No sound data
    Na = 0;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize GUI
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize figure
scrsz = get(0,'Screensize');
fig   = figure('MenuBar','None', ...
               'NumberTitle','off', ...
               'DockControl','off', ...
               'Color',[1, 1, 1], ...
               'Name','Movie Player', ...
               'Position',[0.5 * (scrsz(3:4) - fw) fw], ...
               'WindowButtonDownFcn',@(s,e)togglePlayback(), ...
               'WindowScrollWheelFcn',@(s,e)scroll(e), ...             
               'KeyPressFcn',@(s,e)handleKeyPress(e), ...
               'ResizeFcn',@(s,e)resizeFcn(), ...
               'CloseRequestFcn',@(s,e)close(), ...
               'Visible','off');

% File menu
filem = uimenu(fig,'Label','File');
uimenu(filem,'Label','Save Movie...', ...
             'Callback',@(s,e)saveMovie(), ...
             'Accelerator','S');
uimenu(filem,'Label','Save GIF...', ...
             'Callback',@(s,e)saveGIF(), ...
             'Accelerator','G');
uimenu(filem,'Label','Save Frame...', ...
             'Callback',@(s,e)saveFrame(), ...
             'Accelerator','F');
uimenu(filem,'Label','Close', ...
             'Callback',@(s,e)close(), ...
             'Accelerator','W', ...
             'Separator','on');

% Play menu
playm = uimenu(fig,'Label','Play');
uimenu(playm,'Label','Start', ...
             'Callback',@(s,e)startMovie(), ...
             'Accelerator','1');
uimenu(playm,'Label','Stop', ...
             'Callback',@(s,e)stopMovie(), ...
             'Accelerator','2');
uimenu(playm,'Label','Beginning', ...
             'Callback',@(s,e)setFrame(1), ...
             'Accelerator','A', ...
             'Separator','on');
uimenu(playm,'Label','End', ...
             'Callback',@(s,e)setFrame(nt), ...
             'Accelerator','Z');
repeatm = uimenu(playm,'Label','Repeat', ...
             'Callback',@(s,e)toggleRepeat(), ...
             'Accelerator','R', ...
             'Separator','on');
uimenu(playm,'Label','Frame Rate...', ...
             'Callback',@(s,e)setFrameRate(), ...
             'Separator','on');

% Sound menu
soundm = uimenu(fig,'Label','Sound');
audiom = zeros(1,Na + 1);
for i = 1:Na
    audiom(i) = uimenu(soundm,'Label',movie.audioLabels{i}, ...
                              'Callback',@(s,e)changeAudio(i));
end
audiom(Na + 1) = uimenu(soundm,'Label','(None)', ...
                               'Callback',@(s,e)changeAudio(Na + 1), ...
                               'Separator','on');

% View menu
viewm   = uimenu(fig,  'Label','View');
lockm   = uimenu(viewm,'Label','Locked aspect ratio', ...
                       'Callback',@(s,e)aspectRatio(), ...
                       'Accelerator','L');
unlockm = uimenu(viewm,'Label','Unlocked aspect ratio', ...
                       'Callback',@(s,e)aspectRatio(), ...
                       'Accelerator','U');

% Status menus
framem = uimenu(fig,'Enable','off');
fpsecm = uimenu(fig,'Label','','Enable','off');
writem = uimenu(fig,'Label','','Enable','off');

% Initialize axis
switch LABEL_MODE
    case FIXED
        % Fixed-size labels
        ax   = axes('Units','pixels');
        xlim = [1 dim(1)] - 0.5;
        ylim = [1 dim(2)] - 0.5;
        imh  = image(xlim,ylim,movie.video(args{:},1),'Parent',ax);
    case NORMALIZED
        % Normalized labels
        axpos = [((haveLabels * gap) ./ (1 + 2 * gap)) (dim ./ fw)];
        ax    = axes('Units','normalized','Position',axpos);
        imh   = image([0 1],[0 1],movie.video(args{:},1),'Parent',ax);
end

% Initialize timer
timerobj = timer('ExecutionMode','FixedRate', ...
                 'TasksToExecute',Inf, ...
                 'TimerFcn',@(s,e)updateFrame());

% Format grayscale movie
if ~isColorImage
    % Set colormap
    colormap(ax,CM(NCOLORS));
    set(imh,'CDataMapping','direct');
end

% Add labels
switch LABEL_MODE
    case FIXED
        % Fixed-size labels
        fSize  = fontSize;
        fUnits = 'points';
    case NORMALIZED
        % Normalized labels
        fSize  = fontSize * gap;
        fUnits = 'normalized';
end
txt = @(x,y,theta,str) text(x,y,str,'HorizontalAlignment','center', ...
                                    'VerticalAlignment','middle', ...
                                    'FontUnits',fUnits, ...
                                    'FontName','Helvetica', ...
                                    'FontSize',fSize, ...
                                    'FontWeight','normal', ...
                                    'Rotation',theta);
if haveXlabels
    % x-labels
    if ischar(opts.xlabels)
        xlabels = {opts.xlabels};
    elseif isVector(opts.xlabels)
        xlabels = opts.xlabels(:)';
    else
        xlabels = opts.xlabels;
    end
    xDim = size(xlabels);
    xh   = zeros(xDim);
    for i = 1:xDim(1)
        for j = 1:xDim(2)
            switch LABEL_MODE
                case FIXED
                    % Fixed-size labels
                    x       = (j - 0.5) * dim(1) / xDim(2);
                    xh(i,j) = txt(x,nan,0,xlabels{i,j}); % y set later
                case NORMALIZED
                    % Normalized labels
                    x       = (j - 0.5) / xDim(2);
                    y       = (i - 1) + (2 * i - 3) * (0.6 * gap);
                    xh(i,j) = txt(x,y,0,xlabels{i,j});
            end
        end
    end
end
if haveYlabels
    % y-labels
    if ischar(opts.ylabels)
        ylabels = flipud({opts.ylabels});
    elseif isVector(opts.ylabels)
        ylabels = flipud(opts.ylabels(:));
    else
        ylabels = flipud(opts.ylabels);
    end
    yDim = size(ylabels);
    yh   = zeros(yDim);
    for i = 1:yDim(1)
        for j = 1:yDim(2)
            theta = 90 + 180 * (j == 2);
            switch LABEL_MODE
                case FIXED
                    % Fixed-size labels
                    y       = (i - 0.5) * dim(2) / yDim(1);
                    yh(i,j) = txt(nan,y,theta,ylabels{i,j}); % x set later
                case NORMALIZED
                    % Normalized labels
                    x       = (j - 1) + (2 * j - 3) * (0.6 * gap);
                    y       = (i - 0.5) / yDim(1);
                    yh(i,j) = txt(x,y,theta,ylabels{i,j});
            end
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Declare global variables
alock    = DEFAULT_ALOCK;
audioIdx = 1;
Fv       = movie.Fv;
idx      = [];
idx0     = [];
fps      = [];
ftimer   = [];
stimer   = [];
block    = false;

% Finalize GUI
setFrame(1);
setFrameRate(Fv);
updateMenus();
resizeFcn();
axis(ax,'off');
set(fig,'Visible','on');

% If external control was requested
if nargout > 0
    % Return control struct
    varargout{1} = struct( ...
                   'Start',       @startMovie, ...
                   'Stop',        @stopMovie, ...
                   'Beginning',   @()setFrame(1), ...
                   'End',         @()setFrame(nt), ...
                   'Repeat',      @(varargin)toggleRepeat(varargin{:}), ...
                   'IsPlaying',   @isPlaying, ...
                   'SetFrame',    @(idx)setFrame(idx), ...
                   'SetFrameRate',@(Fv)setFrameRate(Fv), ...
                   'SetAudio',    @(idx)changeAudio(idx), ...
                   'AudioOff',    @()changeAudio(Na + 1), ...
                   'Close',       @close, ...
                   'SaveMovie',   @(path)saveMovie(path), ...
                   'SaveGIF',     @(path)saveGIF(path), ...
                   'SaveFrame',   @(path)saveFrame(path));
else
    % Start movie
    startMovie();
end


% Start movie
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function startMovie(newBlock)
    if isPlaying() || (~repeat && (idx >= nt))
        % Nothing to do
        return;
    end
    
    % If audio stream is active
    if audioIdx <= Na
        % Start audio @ current frame
        Fa    = movie.Fa(audioIdx);
        range = [round(1 + (idx - 1) * Fa / Fv) Ns(audioIdx)];
        play(audioobj{audioIdx},range);
        pause(AUDIOPLAYER_LATENCY); % HACK: match audioplayer latency
    end
    
    % Update blcok
    block = (nargin == 1) && newBlock;
    
    % Start animation
    idx0   = idx;
    fps    = nan(1,FPS_MEMORY);
    ftimer = tic;
    stimer = tic;
    start(timerobj);
end


% Stop movie
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function stopMovie(newBlock)
    % If animation is running
    if isPlaying()
        % Stop animation
        set(fpsecm,'Label','','ForegroundColor',[0 0 0]);
        stop(timerobj);
    end
    
    % If audio is playing
    if (audioIdx <= Na) && isplaying(audioobj{audioIdx})
        % Stop audio
        stop(audioobj{audioIdx});
        drawnow(); pause(0.01); % HACK: give audioplayer time to cool down
    end
    
    % Update blcok
    block = (nargin == 1) && newBlock;
end


% Toggle repeat
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function toggleRepeat(bool)
    % Parse inputs
    if nargin == 0
        bool = ~repeat;
    end
    
    % Set repeat flag
    repeat = bool;
    updateMenus();
end


% Toggle playback
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function togglePlayback()
    % If movie is playing
    if isPlaying()
        % Stop movie
        stopMovie();
    else
        % Start movie
        startMovie();
    end
end


% Set frame rate
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function setFrameRate(newFv)
    % Stop movie
    shouldRestart = isPlaying();
    stopMovie();
    
    % Parse inputs
    if ~exist('newFv','var') || isempty(newFv)
        % Ask user for frame rate
        newFvStr = inputdlg({'Desired frame rate, in fps:'}, ...
                             'Set frame rate',1, ...
                            {sprintf('%d',Fv)},'on');
        drawnow();
        
        % Parse frame rate
        newFv = str2double(newFvStr);
        if isempty(newFv)
            % Quick return
            return;
        end
    end
    
    % Set frame rate
    Fv     = newFv;
    period = round(1000 / Fv) / 1000; % timer() can only do ms precision
    set(timerobj,'StartDelay',period,'Period',period);
    
    % Scale audio rates accordingly
    for ii = 1:Na
        set(audioobj{ii},'SampleRate',movie.Fa(ii) * (Fv / movie.Fv));
    end
    
    % Restart movie, if necessary
    if shouldRestart
        startMovie();
    end
end


% Determine if movie is playing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function bool = isPlaying()
    % Get animiation status
    bool = strcmpi(get(timerobj,'Running'),'on');
end


% Change audio stream
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function changeAudio(newIdx)
    % Make sure index has changed
    if (newIdx == audioIdx)
        % Quick return
        return;
    end
    
    % Stop movie
    shouldRestart = isPlaying();
    stopMovie();
    
    % Change audio stream
    audioIdx = min(max(1,newIdx),Na + 1);
    updateMenus();
    
    % Restart movie, if necessary
    if shouldRestart
        startMovie();
    end
end


% Set frame
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function setFrame(newIdx)
    % Stop movie
    stopMovie();
    
    % Display frame
    displayFrame(newIdx);
end


% Update frame
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function updateFrame()
    % If we're ready for a frame update
    if block || ~ishandle(fig)
        return;
    end
    
    % Set busy lock
    block = true;
    
    % Display new frame
    newIdx     = idx0 + round(toc(stimer) * Fv);
    reachedEnd = (newIdx > nt);
    if repeat
        newIdx = 1 + mod(newIdx - 1,nt);
    end
    displayFrame(newIdx);
    
    % Update frame rate menu
    fps    = [fps(2:end), (1 / toc(ftimer))];
    ftimer = tic;
    cfps   = nanmedian(fps); % Current frame rate
    ftext  = sprintf('%.01f Hz',cfps);
    if cfps < (Fv - 1)
        % Frames were dropped
        ftext = ['* ', ftext, ' *'];
    end
    set(fpsecm,'Label',ftext);
    
    % If we reached the last frame
    if reachedEnd
        stopMovie(true);
        if repeat
            startMovie(true);
        end
    end
    
    % Release busy lock
    block = false;
end


% Display frame
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function displayFrame(newIdx)
    % Display frame
    idx = min(max(newIdx,1),nt);
    set(imh,'CData',movie.video(args{:},idx));
    set(framem,'Label',sprintf('Frame %i/%i',idx,nt));
end


% Handle aspect ratio menu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function aspectRatio()
    % Toggle autoplay lock
    alock = ~alock;
    resizeFcn();
    
    % Update menus
    updateMenus();
end


% Update menus
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function updateMenus()
    % Update play menu
    if repeat
        set(repeatm,'Checked','on');
    else
        set(repeatm,'Checked','off');
    end
    
    % Update view menu
    if alock
        % Aspect ratio currently locked
        set(lockm,'Checked','on');
        set(unlockm,'Checked','off');
    else
        % Aspect ratio currently unlocked
        set(lockm,'Checked','off');
        set(unlockm,'Checked','on');
    end
    
    % Update audio menu
    offinds = setdiff(1:(Na + 1),audioIdx);
    set(audiom(audioIdx),'Checked','on');
    set(audiom(offinds),'Checked','off');
end


% Handle scroll
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function scroll(e)
    % Handle scroll
    switch sign(e.VerticalScrollCount)
        case -1
            % Previous frame
            setFrame(idx - 1);
        case 1
            % Next frame
            setFrame(idx + 1);
    end
end


% Handle figure keypress
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function handleKeyPress(e)
    % Parse keypress
    keyChar = e.Character;
    if isempty(keyChar)
        % Quick return
        return;
    end
    
    % Handle keypress
    switch double(keyChar)
        case {8 13 127} % Backspace/enter/delete
            % Toggle playback
            togglePlayback();
        case {28 30} % Left/up arrow
            % Previous frame
            setFrame(idx - 1);
        case {29 31} % Right/down arrow
            % Next frame
            setFrame(idx + 1);
    end
end


% Handle figure resize
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function resizeFcn()
    % Update figure
    cpos = get(fig,'Position');
    if alock
        % Maintain aspect ratio
        kappa = dim(1) / dim(2);
        if LABEL_MODE == FIXED
            w = kappa * (cpos(4) - 2 * gap * haveXlabels) + ...
                2 * gap * haveYlabels;
        else
            w = kappa * cpos(4);
        end
        npos = [cpos(1:2), w, cpos(4)];
        set(fig,'Position',npos);
    else
        % Don't change figure position
        npos = cpos;
    end
    
    % Update labels
    switch LABEL_MODE
        case FIXED
            % Update fixed-size label positions
            xy0   = (1 + gap * haveLabels);
            axdim = (npos(3:4) - 2 * gap * haveLabels);
            set(ax,'Position',[xy0 axdim]);
            if haveXlabels
                dy = 0.6 * gap * (dim(2) / axdim(2));
                for ii = 1:xDim(1)
                    for jj = 1:xDim(2)
                        posxij    = get(xh(ii,jj),'Position');
                        posxij(2) = (ii - 1) * dim(2) + (2 * ii - 3) * dy;
                        set(xh(ii,jj),'Position',posxij);
                    end
                end
            end
            if haveYlabels
                dx = 0.6 * gap * (dim(1) / axdim(1));
                for ii = 1:yDim(1)
                    for jj = 1:yDim(2)
                        posyij    = get(yh(ii,jj),'Position');
                        posyij(1) = (jj - 1) * dim(1) + (2 * jj - 3) * dx;
                        set(yh(ii,jj),'Position',posyij);
                    end
                end
            end
        case NORMALIZED
            % Empty
    end
end


% Handle figure close
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function close()
    % Stop movie
    stopMovie();
    delete(timerobj);
    for ii = 1:Na
        delete(audioobj{ii});
    end
    
    % Delete GUI
    if ishandle(fig)
        delete(fig);
    end
end


% Save movie
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function saveMovie(path)
    % Stop movie
    shouldRestart = isPlaying();
    stopMovie();
    
    % Parse input
    if ~exist('path','var') || isempty(path)
        % Ask user for path
        path = inputdlg('Output path (.avi or .mp4):', ...
                        'Save Movie',[1, 50],{'./movie.avi'});
        drawnow();
        if isempty(path)
            % Quick return
            return;
        end
        path = path{1};
    end
    
    % Parse extension
    [base, ~, ext] = fileparts(path);
    switch ext
        case '.mp4'
            % MP4
            format = 'MPEG-4';
        case '.avi'
            % JPEG-compressed AVI
            format = 'Motion JPEG AVI';
        case '.aviu'
            % Uncompressed AVI
            format = 'Uncompressed AVI';
            path   = path(1:(end - 1)); % Remove trailing 'u'
        otherwise
            % Unsupported extension
            error('Unsupported extension "%s"',ext);
    end
    
    % Create output directory, if necessary
    if ~exist(base,'dir')
        mkdir(base);
    end
    
    % Initialize writer
    set(writem,'Label','*** WRITING ***');
    vidObj = VideoWriter(path,format);
    vidObj.FrameRate = Fv;
    vidObj.open();
    
    % Record movie
    fprintf('*** Writing movie\n');
    stimer = tic();
    idx = 0;
    while idx < nt
        % Display frame
        displayFrame(idx + 1);
        
        % Write frame to file
        drawnow(); pause(0.01);
        vidObj.writeVideo(getframe(fig));
    end
    vidObj.close();
    set(writem,'Label','');
    fprintf('Movie "%s" written [Time = %.2fs]\n',path,toc(stimer));
    
    % Restart movie, if necessary
    if shouldRestart
        startMovie();
    end
end


% Save GIF
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function saveGIF(path)
    % Stop movie
    shouldRestart = isPlaying();
    stopMovie();
    
    % Parse input
    if ~exist('path','var') || isempty(path)
        % Ask user for path
        path = inputdlg('Output path (.gif):', ...
                        'Save GIF',[1, 50],{'./movie.gif'});
        drawnow();
        if isempty(path)
            % Quick return
            return;
        end
        path = path{1};
    end
    
    % Parse extension
    [base, ~, ext] = fileparts(path);
    if ~strcmp(ext,'.gif')
        % Unsupported extension
        error('Received extension "%s", expected ''.gif''',ext);
    end
    
    % Create output directory, if necessary
    if ~exist(base,'dir')
        mkdir(base);
    end
    
    % Write GIF
    set(writem,'Label','*** WRITING ***');
    fprintf('*** Writing GIF\n');
    stimer = tic();
    idx = 0;
    X   = [];
    while idx < nt
        % Display frame
        displayFrame(idx + 1);
        
        % Save frame data
        drawnow(); pause(0.01);
        s = getframe(fig);
        if isempty(X)
            X = s.cdata;
        else
            X(:,:,:,idx) = s.cdata; %#ok
        end
    end
    writeGIF(X,path,Fv);
    set(writem,'Label','');
    fprintf('GIF "%s" written [Time = %.2fs]\n',path,toc(stimer));
    
    % Restart movie, if necessary
    if shouldRestart
        startMovie();
    end
end


% Save current frame
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function saveFrame(path)
    % Stop movie
    shouldRestart = isPlaying();
    stopMovie();
    
    % Parse inputs
    if ~exist('path','var') || isempty(path)
        % Ask user for path
        path = inputdlg('Output path (.png, .jpg, or .tiff):', ...
                        'Save Frame',[1, 50],{'./frame.png'});
        drawnow();
        if isempty(path)
            % Quick return
            return;
        end
        path = path{1};
    end
    
    % Parse path
    [base, ~, ext] = fileparts(path);
    
    % Create output directory, if necessary
    if ~isempty(base) && ~exist(base,'dir')
        mkdir(base);
    end
    
    % Save frame
    stimer = tic();
    set(writem,'Label','*** WRITING ***');
    drawnow(); pause(0.01);
    f = getframe(fig);
    imwrite(f.cdata,path,ext(2:end));
    set(writem,'Label','');
    fprintf('Frame "%s" saved [Time = %.2fs]\n',path,toc(stimer));
    
    % Restart movie, if necessary
    if shouldRestart
        startMovie();
    end
end

end


% Determine if input is a vector
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function bool = isVector(X)
    bool = (ndims(X) - nnz(size(X) == 1)) == 1;
end
