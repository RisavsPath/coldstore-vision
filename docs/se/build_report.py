"""Builds docs/Project_Report_ColdStore_Vision.pdf.   python docs/se/build_report.py"""
from __future__ import annotations

from common import ROOT, arrow, box, figure, image, page, svg, table, to_pdf

A = ROOT / "video" / "web" / "assets"


def pipeline() -> str:
    names = [("Recordings", "Day 1, Day 2"), ("Keyframes", "sharp, spaced"), ("Alignment", "CLIP + LoFTR"),
             ("Segmentation", "YOLO11s-seg")]
    names2 = [("Lighting", "histogram match"), ("Matching", "Hungarian"), ("Diagnosis", "colour, texture, flow"), ("Outputs", "5 files")]
    parts = []
    w, h, gap = 158, 58, 28
    for i, n in enumerate(names):
        x = 12 + i * (w + gap)
        parts.append(box(x, 12, w, h, *n, bold_first=True))
        if i:
            parts.append(arrow(x - gap, 41, x, 41))
    for i, n in enumerate(reversed(names2)):
        x = 12 + i * (w + gap)
        parts.append(box(x, 120, w, h, *n, bold_first=True))
        if i:
            parts.append(arrow(x, 149, x - gap, 149))
    last = 12 + 3 * (w + gap) + w / 2
    parts.append(arrow(last, 70, last, 120))
    return svg(760, 190, *parts)


body = f"""
<div class="cover">
<p><b>KALINGA INSTITUTE OF INDUSTRIAL TECHNOLOGY</b><br>School of Computer Engineering</p>
<p style="margin-top:22mm">Project Report</p>
<h1>ColdStore-Vision</h1>
<p>Automated Cold-Storage Sack Inspection from Video</p>
<p style="margin-top:22mm"><b>Submitted by</b><br><mark>[Name] - [Roll number]</mark><br><mark>[Name] - [Roll number]</mark><br><mark>[Name] - [Roll number]</mark></p>
<p><b>Group:</b> <mark>[Group name]</mark> &nbsp; <b>Section:</b> <mark>[Section]</mark></p>
<p><b>Course:</b> <mark>[Course name and code]</mark><br><b>Submitted to:</b> <mark>[Faculty name]</mark></p>
<p>November 2026</p>
</div>

<h1>Abstract</h1>
<p>Sacks in cold stores are checked by hand, which is slow and irregular. ColdStore-Vision compares two phone-video walkthroughs of the same stack, recorded on different days, and gives each sack a status with a written reason. It selects sharp keyframes, aligns the two days using CLIP retrieval and LoFTR matching, outlines sacks with a custom-trained YOLO11s-seg model, compensates for lighting, matches sacks with the Hungarian algorithm, and measures colour, texture and motion.</p>
<p>The detector reached a mask mAP@50 of 0.797 and precision of 0.798 on its test set, just below the targets of 0.85 and 0.80. On real footage from a working godown, the system tracked 22 onion sacks: 4 NORMAL, 4 WARNING, 1 MISSING and 13 UNVERIFIED. It flagged a sack that had been swapped between the days and did not report sacks hidden behind a pillar as missing. The high unverified count came from filming each day from a different viewpoint. All 16 automated tests pass. The main conclusion is that the software works, and that consistent capture is now the limiting factor.</p>

<h1>1. Introduction</h1>
<h2>1.1 Background</h2>
<p>Onions, potatoes and garlic are stored in stacked jute or mesh sacks. Spoilage spreads between sacks, stacks settle, and stock moves without always being recorded. Walking the aisles and looking is the usual check, but it is slow, and it leaves no record that can be compared later.</p>
<h2>1.2 Problem statement</h2>
<p>Given two videos of one stack filmed on different days, decide for each visible sack whether it is unchanged, has changed in appearance, has settled, or has gone, and say why. The method must work with an ordinary phone and require no markers on the racks.</p>
<h2>1.3 Objectives</h2>
<ol><li>Train an instance segmentation model that separates touching sacks.</li>
<li>Align two recordings taken from different viewpoints and in different light.</li>
<li>Identify each sack across days and diagnose changes with stated reasons.</li>
<li>Avoid false alarms: say UNVERIFIED if the evidence is too weak.</li>
<li>Produce outputs that a store manager can read without technical knowledge.</li>
<li>Test the system on both synthetic and real footage, and report the results honestly.</li></ol>
<h2>1.4 Scope</h2>
<p>The system processes recordings offline on a laptop CPU. A concept animation shows how a drone could capture the footage later, but no drone was built.</p>

<h1>2. Background and related methods</h1>
{table(["Topic", "Method used", "Why it suits this problem"], [
    ["Blur detection", "Variance of the Laplacian", "Cheap, and handheld walks blur on each step"],
    ["Local contrast", "CLAHE [1]", "Dim scenes; avoids blowing out lamps"],
    ["Feature matching", "SIFT with Lowe's ratio test [2]", "Scale and rotation invariant; sufficient within one walk"],
    ["Robust fitting", "RANSAC [3], MAGSAC [4]", "Identical-looking sacks produce many false matches"],
    ["Dense matching", "LoFTR [5]", "Survives simultaneous change in angle and light"],
    ["Image retrieval", "CLIP embeddings [6]", "Reduces the pairs LoFTR must process"],
    ["Instance segmentation", "YOLO11s-seg [7]", "Separates same-coloured touching sacks; runs on CPU"],
    ["Label drafting", "SAM 2.1 [8], FastSAM", "One point prompt gives one mask"],
    ["Texture", "GLCM [9]", "Surface texture may change before colour"],
    ["Motion", "Farneb&auml;ck optical flow [10]", "Settling shows up as downward motion and compression"],
    ["Assignment", "Hungarian algorithm [11]", "Optimal one-to-one matching"],
    ["Panorama blending", "Multi-band blending [12]", "Hides exposure seams"],
], [22, 34, 44])}

<h1>3. System design</h1>
{figure(pipeline(), "Figure 1: Pipeline. Stages run in the order of the arrows.")}
<p>The code is split into layers: ingestion, registration, segmentation, analytics and reporting. All thresholds are in one YAML file, and the detector can be replaced by swapping one weights file. Detailed design is in Part B of the SRS.</p>
<p>Three design decisions shaped the system:</p>
<ul><li><b>Compare locally, not across the whole frame.</b> A stack is 3-D, so one homography cannot align all of it across viewpoints. Each matched sack is re-aligned on its own, and all measurements are taken inside its mask, eroded by 7 pixels.</li>
<li><b>Refuse to guess.</b> UNVERIFIED is a proper outcome. It is used when a view has too few matching sacks or when something blocks the sack.</li>
<li><b>Build in parallel.</b> A fallback detector and a colour-based test detector let the whole pipeline be built and tested before the trained model existed.</li></ul>

<h1>4. Implementation</h1>
<p>Delivered size is 4,128 lines of Python, excluding tests and the explainer video. Main libraries: OpenCV, PyTorch, ultralytics, kornia, OpenCLIP, SciPy, scikit-image.</p>
<h2>4.1 Keyframes</h2>
<p>Frames are resized to 1280 pixels wide. A keyframe is chosen every 0.5 s, or every 0.5 m when camera poses are available. If a frame's Laplacian variance is below 100, the sharpest frame within 5 frames replaces it.</p>
<h2>4.2 Alignment</h2>
<p>For each Day 1 keyframe, CLIP shortlists the 4 most similar Day 2 keyframes. LoFTR finds correspondences, and MAGSAC fits a homography, needing at least 40 inliers. Plain SIFT was tried first, but failed because angle, distance, resolution and white balance all changed at once. Within a walk, keyframes are chained with affine transforms and blended into a panorama.</p>
<h2>4.3 Dataset and detector</h2>
<p>No cold-store footage existed at the start, so images were collected from the web. They were deduplicated with perceptual hashing, filtered by CLIP score, and judged by AI reviewer agents. FastSAM, CLIP and YOLOE drafted the labels. Agents then pointed out errors on a grid, and SAM 2.1 drew corrected masks from those points.</p>
{table(["Item", "Value"], [
    ["Images reviewed for labelling", "241"],
    ["Images used", "152 (train 104, validation 32, test 16)"],
    ["Sack outlines", "2,782 (train 1,878, validation 587, test 317)"],
    ["Model and input size", "yolo11s-seg, 640 pixels"],
    ["Training", "Kaggle NVIDIA T4, stopped at 105 of up to 150 epochs"],
    ["Augmentation", "HSV shift, flip, rotation, scale, mosaic, mixup, copy-paste, grey-scale, blur"],
], [40, 60])}
<h2>4.4 Lighting compensation</h2>
<p>Day 2 colours are remapped so each channel's cumulative histogram matches Day 1, leaving out sacks that are missing or new. The median hue difference on the background gives the drift across the whole scene, and each sack's hue change is reported relative to it. Hue averages use a circular mean because hue is an angle.</p>
<h2>4.5 Matching</h2>
<p>The Hungarian algorithm assigns Day 1 sacks to Day 2 sacks. Under parallax, masks of the same sack may hardly overlap, so the cost is centre distance divided by sack size, plus a penalty for size mismatch. A match based only on centres must also look alike after local alignment. Otherwise, a removed sack could be matched to its neighbour.</p>
<h2>4.6 Diagnosis</h2>
{table(["Check", "Measure", "Result"], [
    ["Missing", "At least 45% of area changed (20% if uncovered), no match; or depth above 0.30 m", "MISSING"],
    ["Rot colour", "Pixels with hue 10-30, saturation above 60, value below 110: growth of area", "+3 points WARNING, +8 points CRITICAL"],
    ["Colour drift", "Relative hue change above 15 degrees", "WARNING"],
    ["Texture", "GLCM contrast up 35%, or homogeneity down 20%", "WARNING"],
    ["Settling", "Downward flow, negative divergence, top edge drop above 4 px (10 px critical)", "WARNING or CRITICAL (capped at WARNING across viewpoints)"],
    ["Swapped", "Hue shift of at least 10 degrees with at least 40% area changed", "WARNING, possibly replaced"],
    ["Blocked or unsure", "Hue jump above 90 degrees, or under 50% of the sacks in the view match", "UNVERIFIED"],
], [18, 52, 30])}
<p>Sacks seen in several pairs take their majority status. Sacks cut by the side of the frame are never called MISSING.</p>
<h2>4.7 Outputs</h2>
<p>An overlay video, an audit report (HTML and JSON), a side-by-side video, a digital twin page and a panorama slider.</p>

<h1>5. Testing</h1>
{table(["Level", "What was tested", "Result"], [
    ["Unit (8 tests)", "Homography, histogram matching, hue drift, rot and GLCM, erosion, flow, matching, bay index", "All pass"],
    ["Integration (6 tests)", "Keyframes, sack count, depth, bays, panorama alignment, Polycam conversion", "All pass"],
    ["System (2 tests)", "Synthetic aisle with a missing sack, two rotting, one slumped, one new, plus a lighting change", "All found, no false alarm"],
    ["Field run", "Real godown footage, onion stack", "See section 6"],
    ["Invalid input", "Non-existent path", "Clean error message"],
], [20, 60, 20])}
<p>The suite ran with pytest on 2 November 2026: <b>16 passed in 93.56 s.</b></p>

<h1>6. Results</h1>
<h2>6.1 Detector</h2>
{table(["Metric (test set)", "Value", "Target"], [
    ["Mask mAP@50", "0.797", "0.85 (not met)"], ["Mask precision", "0.798", "0.80 (just missed)"],
    ["Mask recall", "0.767", "-"], ["Mask mAP@50-95", "0.531", "-"], ["Box mAP@50", "0.801", "-"],
], [50, 20, 30])}
<p>The model is strong on dense stacks but was trained only on web images, and made some false detections, for example on an umbrella. A fine-tune on godown frames is the planned fix. Figure 2 shows it on a real frame, where it separated 44 touching sacks.</p>
{figure(image(A / "day1_detections.jpg", 92), "Figure 2: Detector output on a real Day 1 frame")}
<h2>6.2 Field experiment</h2>
{table(["Item", "Day 1 (21 Oct 2026)", "Day 2 (22 Oct 2026)"], [
    ["Recording", "1080p, 30 fps, 15 s, side-on", "4K, 30 fps, 17 s, closer and at a different angle"],
    ["Part used", "From 9.8 s (onion stack)", "From 9.8 s (onion stack)"],
    ["Keyframes", "11", "15"],
], [20, 40, 40])}
<p>10 keyframe pairs were aligned, and the run took 791 s on a laptop CPU. The potato stack could not be compared, as it was filmed too close on Day 2.</p>
{table(["Status", "Sacks"], [["NORMAL", 4], ["WARNING", 4], ["MISSING", 1], ["UNVERIFIED", 13], ["<b>Total</b>", "<b>22</b>"]], [70, 30])}
<p>Flagged sacks and the system's own reasons:</p>
{table(["Sack", "Status", "Reason reported"], [
    ["Sack_005", "WARNING", "Texture drift: contrast +134.6%, homogeneity -14.1%"],
    ["Sack_006", "WARNING", "Settled 5 px; top dropped 5 px. Viewpoint differs, verify on site."],
    ["Sack_011", "WARNING", "Hue -14.2 degrees, 49% of area differs: possibly swapped or replaced"],
    ["Sack_013", "MISSING", "57% of area changed, no match, only 15% covered by any Day 2 sack"],
    ["Sack_014", "WARNING", "Settled 31 px with compressive flow. Viewpoint differs, verify on site."],
], [15, 15, 70])}
<p>Sack_011 was checked by eye: as Figure 3 shows, an orange sack at the front on Day 1 is gone on Day 2. The other flags were not independently checked on site. Because no ground truth was recorded for this pair, <b>no accuracy figure can be claimed for the field run</b>. The settling flags are the least certain, since viewpoint differences look like motion. Sack_013 is reported as an unconfirmed MISSING.</p>
{figure(image(A / "swap_crop.png", 45), "Figure 3: Sack_011, Day 1 (left) and Day 2 (right)")}
{figure(image(A / "audit_report.png", 88), "Figure 4: Audit report")}
<h2>6.3 Discussion</h2>
<p>13 of 22 sacks were unverified, which is too many for practical use. The cause is known: each walk took a different path, and a pillar blocked part of the view on Day 2. Across-day alignment did work, the swapped sack was caught, and the pillar case did not cause false MISSING reports.</p>

<h1>7. Limitations</h1>
<ul><li>Detector accuracy is below target and has not been trained on godown frames.</li>
<li>Handheld footage from different viewpoints leaves most sacks unverified.</li>
<li>No ground truth for the real pair, and just one field experiment at one site.</li>
<li>When a removed sack is overlapped by a neighbour, a test gives WARNING rather than MISSING.</li>
<li>Thresholds were tuned on one site.</li>
<li>Only the visible surface is assessed; rot inside a sack with no visible sign cannot be found.</li>
<li>LiDAR depth is supported in code but was not used on real footage, because the capture was a room scan.</li>
<li>Tested on Windows only. Whole-aisle runs are untested.</li></ul>

<h1>8. Future work</h1>
<ol><li><b>Fixed route:</b> a drone or rail, for the same viewpoint each day.</li>
<li><b>Rack tags:</b> QR or barcode tags to anchor each bay.</li>
<li><b>LiDAR:</b> a proper rack-pass capture to confirm missing sacks by depth.</li>
<li><b>Model v2:</b> fine-tune on godown frames to reach the 0.85 target.</li>
<li><b>Ground truth trial:</b> record a pair with known changes and measure real precision and recall.</li>
<li><b>Engineering:</b> add version control and pin package versions.</li></ol>

<h1>9. Conclusion</h1>
<p>ColdStore-Vision turns two ordinary phone videos into a per-sack inspection report with reasons, using no markers on the racks. A trained detector, cross-day alignment and rule-based diagnosis were built, tested and run on real footage. The system caught a real change and avoided false alarms from a pillar, though accuracy targets were narrowly missed and more than half of the sacks were left unverified. The next step is not more algorithms but consistent capture and a measured trial.</p>

<h1>10. Use of AI tools</h1>
<p>Claude Code, an AI coding assistant, was used under human direction to write and revise code, scripts and documents, including this report. AI agents also reviewed training images and labels. The team set the scope, recorded the footage and checked the results.</p>

<h1>References</h1>
<ol style="font-size:10.5pt">
<li>K. Zuiderveld, "Contrast Limited Adaptive Histogram Equalization," in <i>Graphics Gems IV</i>, 1994.</li>
<li>D. Lowe, "Distinctive Image Features from Scale-Invariant Keypoints," <i>IJCV</i>, 2004.</li>
<li>M. Fischler and R. Bolles, "Random Sample Consensus," <i>Communications of the ACM</i>, 1981.</li>
<li>D. Barath, J. Matas and J. Noskova, "MAGSAC: Marginalizing Sample Consensus," <i>CVPR</i>, 2019.</li>
<li>J. Sun et al., "LoFTR: Detector-Free Local Feature Matching with Transformers," <i>CVPR</i>, 2021.</li>
<li>A. Radford et al., "Learning Transferable Visual Models From Natural Language Supervision," <i>ICML</i>, 2021.</li>
<li>Ultralytics, YOLO11 documentation and software.</li>
<li>N. Ravi et al., "SAM 2: Segment Anything in Images and Videos," 2024.</li>
<li>R. Haralick, K. Shanmugam and I. Dinstein, "Textural Features for Image Classification," <i>IEEE Trans. SMC</i>, 1973.</li>
<li>G. Farneb&auml;ck, "Two-Frame Motion Estimation Based on Polynomial Expansion," <i>SCIA</i>, 2003.</li>
<li>H. Kuhn, "The Hungarian Method for the Assignment Problem," <i>Naval Research Logistics Quarterly</i>, 1955.</li>
<li>P. Burt and E. Adelson, "A Multiresolution Spline with Application to Image Mosaics," <i>ACM Trans. Graphics</i>, 1983.</li>
<li>R. Mall, <i>Fundamentals of Software Engineering</i>, PHI Learning.</li>
<li>IEEE Std 830-1998, Recommended Practice for Software Requirements Specifications.</li>
</ol>
"""

if __name__ == "__main__":
    to_pdf(page("Project Report: ColdStore-Vision", body), "Project_Report_ColdStore_Vision.pdf")
