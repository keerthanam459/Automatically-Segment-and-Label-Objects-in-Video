clc; clear; close all;

cfg.videoPath      = 'C:\MATLAB Drive\MatHack\New.mp4';   
cfg.outputRoot     = 'C:\MATLAB Drive\MatHack\results';      
cfg.frameSize      = [416 416];      % Must match YOLO input (416x416)
cfg.frameRate      = 25;             % Output video FPS
cfg.scoreThreshold = 0.45;          % Confidence threshold (0-1)
cfg.iouThreshold   = 0.45;          % NMS IoU threshold
cfg.targetClasses  = {'person', 'car', 'truck', 'bus', 'motorbike'}; % COCO labels
cfg.maxFrames      = Inf;            

% Color map for labels
cfg.colorMap = containers.Map( ...
    {'person',  'car',    'truck',  'bus',    'motorbike'}, ...
    {'yellow',  'cyan',   'green',  'magenta','red'});

cfg.framesDir = fullfile(cfg.outputRoot, 'frames');
cfg.labelDir  = fullfile(cfg.outputRoot, 'annotated');
cfg.videoOut  = fullfile(cfg.outputRoot, 'labeled_output.avi');

for d = {cfg.framesDir, cfg.labelDir}
    if ~exist(d{1}, 'dir'), mkdir(d{1}); end
end

fprintf('\n[INFO] Output root: %s\n', cfg.outputRoot);

fprintf('[INFO] Loading YOLOv4 detector (coco)...\n');
try
    detector = yolov4ObjectDetector('csp-darknet53-coco');
    fprintf('[INFO] YOLOv4 loaded successfully.\n');
catch ME
    error('[ERROR] Could not load YOLOv4: %s\nEnsure Deep Learning Toolbox + Computer Vision Toolbox are installed.', ME.message);
end

fprintf('[INFO] Reading video: %s\n', cfg.videoPath);
vidObj   = VideoReader(cfg.videoPath);
totalEst = floor(vidObj.Duration * vidObj.FrameRate);
fprintf('[INFO] Estimated frames: %d at %.1f FPS\n', totalEst, vidObj.FrameRate);

frameIdx = 0;
while hasFrame(vidObj)
    frameIdx = frameIdx + 1;
    if frameIdx > cfg.maxFrames, break; end

    frame        = readFrame(vidObj);
    frameResized = imresize(frame, cfg.frameSize);  % RGB

    framePath = fullfile(cfg.framesDir, sprintf('frame_%05d.png', frameIdx));
    imwrite(frameResized, framePath);
end
numFrames = frameIdx;
fprintf('[INFO] Extracted %d frames.\n', numFrames);

fprintf('[INFO] Running detection on %d frames...\n', numFrames);

allBboxes    = cell(numFrames, 1);
allLabels    = cell(numFrames, 1);
allScores    = cell(numFrames, 1);
detCountLog  = zeros(numFrames, 1);

frameFiles = dir(fullfile(cfg.framesDir, '*.png'));
[~, sortIdx] = sort({frameFiles.name});
frameFiles   = frameFiles(sortIdx);

for i = 1:numFrames
    frame = imread(fullfile(cfg.framesDir, frameFiles(i).name));

    %YOLOv4 detection
    [bboxes, scores, labels] = detect(detector, frame, ...
        'Threshold',     cfg.scoreThreshold, ...
        'SelectStrongest', true);                     % Built-in NMS

    %Filtering to target classes only
    if ~isempty(labels)
        mask   = ismember(cellstr(labels), cfg.targetClasses);
        bboxes = bboxes(mask, :);
        scores = scores(mask);
        labels = labels(mask);
    end

    %Annotate frame
    annotated = frame;
    if ~isempty(bboxes)
        for k = 1:size(bboxes,1)
            lbl   = char(labels(k));
            score = scores(k);
            clr   = 'white';
            if isKey(cfg.colorMap, lbl)
                clr = cfg.colorMap(lbl);
            end
            caption = sprintf('%s %.0f%%', lbl, score*100);
            annotated = insertObjectAnnotation(annotated, 'rectangle', ...
                bboxes(k,:), caption, ...
                'Color', clr, 'LineWidth', 2, 'FontSize', 12);
        end
    end

    % Annotated frames
    imwrite(annotated, fullfile(cfg.labelDir, frameFiles(i).name));

    allBboxes{i}   = bboxes;
    allLabels{i}   = labels;
    allScores{i}   = scores;
    detCountLog(i) = size(bboxes, 1);

    if mod(i, 10) == 0 || i == numFrames
        fprintf('  Frame %4d / %4d | Detections: %d\n', i, numFrames, detCountLog(i));
    end
end

% Detection results
save(fullfile(cfg.outputRoot, 'detection_results.mat'), ...
    'allBboxes', 'allLabels', 'allScores');
fprintf('[INFO] Detection complete. Results saved.\n');

fprintf('[INFO] Generating evaluation plots...\n');

figure('Name', 'Detection Summary', 'NumberTitle', 'off', ...
       'Position', [100 100 1000 400]);

%Detection count per frame
subplot(1,2,1);
plot(1:numFrames, detCountLog, 'b-', 'LineWidth', 1.5);
xlabel('Frame Number');
ylabel('Number of Detections');
title('Detections Per Frame');
grid on;

%Class distribution
subplot(1,2,2);
allLabelsFlat = vertcat(allLabels{:});
if ~isempty(allLabelsFlat)
    labelCounts = countcats(categorical(cellstr(allLabelsFlat)));
    labelNames  = categories(categorical(cellstr(allLabelsFlat)));
    bar(categorical(labelNames), labelCounts, 'FaceColor', [0.2 0.5 0.8]);
    xlabel('Class'); ylabel('Total Detections');
    title('Class Distribution (All Frames)');
    grid on;
else
    text(0.5, 0.5, 'No detections found', 'HorizontalAlignment','center');
    title('Class Distribution');
end

saveas(gcf, fullfile(cfg.outputRoot, 'detection_summary.png'));
fprintf('[INFO] Evaluation plot saved.\n');


fprintf('[INFO] Compiling output video...\n');

annotatedFiles = dir(fullfile(cfg.labelDir, '*.png'));
[~, sortIdx]   = sort({annotatedFiles.name});
annotatedFiles = annotatedFiles(sortIdx);

vWriter            = VideoWriter(cfg.videoOut, 'Motion JPEG AVI');
vWriter.FrameRate  = cfg.frameRate;
vWriter.Quality    = 90;
open(vWriter);

for i = 1:numel(annotatedFiles)
    frame = imread(fullfile(cfg.labelDir, annotatedFiles(i).name));
    writeVideo(vWriter, frame);
end
close(vWriter);

fprintf('[INFO] Video saved: %s\n', cfg.videoOut);


fprintf('\n========================================\n');
fprintf('  PIPELINE COMPLETE — SUMMARY\n');
fprintf('========================================\n');
fprintf('  Total frames processed : %d\n',   numFrames);
fprintf('  Total detections       : %d\n',   sum(detCountLog));
fprintf('  Avg detections/frame   : %.2f\n', mean(detCountLog));
fprintf('  Max detections/frame   : %d\n',   max(detCountLog));
fprintf('  Output video           : %s\n',   cfg.videoOut);
fprintf('  Results MAT file       : %s\n',   fullfile(cfg.outputRoot,'detection_results.mat'));
fprintf('========================================\n');

