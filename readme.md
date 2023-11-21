Our three stage DLO perception pipeline.

1. To get SAM+Grounding DINO masks, please follow https://github.com/IDEA-Research/Grounded-Segment-Anything and use their grounding_dino_demo.py with text rope prompt.

2. To get post-processed SAM masks, please use sam_post_processing.py

3. To get 3D reconstruction, please use reconstruction_3d.py

4. To get DER smoothing, please check DER folder and check instruction.txt.

As for the test dataset, we have two datasets:

1. Synthetic dataset (https://drive.google.com/file/d/110YY_n6KPS_CbTmrJCuPVyvPRuBEa7tj/view?usp=sharing), we have a pair of <X_Y_mask.png, X_Y_node.npy, X_Y_pc.npy> for each image. X is the DLO id, Y is the time step from 0 to 300. The mask is the ground-truth mask as input, the node is the ground-truth keypoint of 60 points, pc is the point cloud.

2. Real dataset (https://drive.google.com/file/d/169wLkCgy5-9n7PcFkOf1X3SxLsknmzBo/view?usp=sharing), we have the image and the mask, under image folder, there are 7 DLO folders, each has A (unoccluded) and B with a, b, c, d, e, 5 different poses. In each folder of the pose, there is a pc folder containing the point cloud, and a rgb folder containing the RGB image. The first image is unoccluded, and the rest are randomly occuluded. The mask dataset follows the same structure but only have rgb folder, which contains the labelled masks.

We will further improve the code (more in-detailed comments, faster running speed) before we release on public, e.g., on GitHub. Please feel free to contact us (zhaole.sun@ed.ac.uk or zhaole.sun.97@gmail.com) if you have any questions. Thanks!
