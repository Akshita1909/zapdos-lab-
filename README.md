Red Laser Boundary Detector
Zapdos Labs — Founding Computer Vision Engineer Technical Challenge

Overview
This project implements a Computer Vision pipeline to detect and mask the red laser safety boundary projected around forklifts in a warehouse environment. Given a video clip, the pipeline outputs a binary (black & white) mask video where the laser-bounded region is filled white, at a minimum of 2 FPS.
The solution is implemented as a Google Colab notebook (laser_boundary_detector.ipynb) using classical CV techniques — no heavy models required, making it fast and interpretable.

Approach
The pipeline follows 4 stages per frame:
Input Frame
    │
    ▼
① HSV Red Segmentation      — isolate red laser pixels
    │
    ▼
② Morphological Denoising   — remove speckle, fill line gaps
    │
    ▼
③ Convex Hull Fitting        — produce a clean filled boundary polygon
    │
    ▼
④ Temporal Smoothing (EMA)  — stabilise mask across frames
    │
    ▼
Output Binary Mask Frame
Stage 1 — HSV Red Segmentation
Red is tricky in HSV because it wraps around hue 0/180. Two separate inRange masks are computed and OR'd:

Range 1: Hue 0–10 (lower red)
Range 2: Hue 165–180 (upper red)

Saturation and value floors are set high enough to reject floor glare and ambient reddish tones.
Stage 2 — Morphological Denoising
Three sequential morphological ops clean up the raw mask:
OpKernelPurposeOPEN3×3 ellipseRemoves dust/speckleCLOSE21×21 rectBridges gaps between laser line segmentsDILATE5×5 ellipseSlightly thickens lines before hull fitting
Stage 3 — Convex Hull Boundary Fitting
All sufficiently large contours (≥500 px²) are merged and a convex hull is computed over their combined points. The hull is filled solid white on a black canvas — producing a clean filled rectangle/polygon even under perspective warp.

If the laser boundary is non-convex (e.g. L-shaped), minAreaRect can be used instead — noted in the tuning cheatsheet at the bottom of the notebook.

Stage 4 — Temporal EMA Smoothing
Frame-to-frame flicker is reduced with an exponential moving average blend between the current and previous mask (alpha=0.6). The blended float mask is thresholded at 127 to stay binary.

How to Run (Google Colab)

Open laser_boundary_detector.ipynb in Google Colab
Run Cell 1 (Setup) — installs dependencies and clones the data repo
Run Cell 2 (Core Pipeline) — defines all processing functions
Run Cell 3 (Visualise Stages) — inspect the pipeline on a single frame
(Optional) Run Cell 4 (Tune HSV) — tweak thresholds interactively
Run Cell 5 (Process All Clips) — generates mask .mp4 files in /content/output_masks/
Run Cell 6 (Visual QA) — side-by-side review of input vs mask
Run Cell 7 (Download ZIP) — downloads all outputs as laser_masks.zip


Key Parameters (Config class)
ParameterDefaultEffectRED_LOW_1 / HIGH_1[0,120,80] / [10,255,255]Lower red HSV rangeRED_LOW_2 / HIGH_2[165,120,80] / [180,255,255]Upper red HSV rangeOPEN_KERNEL3Speckle removal sizeCLOSE_KERNEL21Gap-filling size between laser segmentsDILATE_KERNEL5Final thickening before hullMIN_AREA500 px²Minimum blob size to include in hullTEMPORAL_ALPHA0.6EMA weight (higher = less smoothing)

Tuning Cheatsheet
ProblemFixLaser not detectedWiden HSV ranges (lower sat/val floor)Floor glare bleeding inRaise MIN_AREA, narrow hue rangeGaps in the hullIncrease CLOSE_KERNEL (try 25–35)Flickery outputLower TEMPORAL_ALPHA toward 0.4Too slowAdd frame = cv2.resize(frame, (640, 360))Non-convex laser shapeUse minAreaRect instead of convexHull

Design Decisions & Trade-offs
Why classical CV over a model?
The laser is a strongly saturated, consistent red — HSV thresholding is highly reliable, interpretable, and runs well above the 2 FPS requirement on Colab CPU. Running a segmentation model (SAM, etc.) on every frame would be unnecessarily expensive for a signal this clean.
Why convex hull over raw contours?
The laser projects as line segments, not a filled region. Connecting all detected segments into their convex hull gives a clean, filled mask that handles perspective warp naturally without needing a homography or depth estimate.
Why EMA smoothing over no smoothing?
Single-frame masks flicker due to exposure variation and partial occlusion. EMA with alpha=0.6 provides ~2-frame memory, significantly reducing flicker without introducing noticeable lag.

Dependencies
opencv-python-headless
numpy
matplotlib
All installed automatically in Cell 1.

Output Format

File: <clip_name>_mask.mp4
Codec: mp4v
Channels: Grayscale (single channel binary)
FPS: Matches input clip FPS
Resolution: Matches input clip reso
