# Third-Party Notices

This repository contains or derives from third-party open-source software.

## Upstream projects

### 1. HKUST-Aerial-Robotics / VINS-Fusion
- URL: https://github.com/HKUST-Aerial-Robotics/VINS-Fusion
- License: GNU General Public License v3.0 (GPL-3.0)
- Role in this repository: original VINS-Fusion codebase and upstream project lineage

### 2. zinuok / VINS-Fusion-ROS2
- URL: https://github.com/zinuok/VINS-Fusion-ROS2
- License: GPL-3.0
- Role in this repository: ROS 2 adaptation lineage

### 3. JanekDev / VINS-Fusion-ROS2-humble-arm
- URL: https://github.com/JanekDev/VINS-Fusion-ROS2-humble-arm
- License: GPL-3.0
- Role in this repository: ROS 2 fork/port lineage

### 4. cannnnxu / VINS-Fusion-ROS2-jazzy
- URL: https://github.com/cannnnxu/VINS-Fusion-ROS2-jazzy
- License: GPL-3.0
- Role in this repository: immediate upstream used for this research fork

### 5. cvg / LightGlue

* URL: https://github.com/cvg/LightGlue
* License: Apache License 2.0
* Role in this repository: optional visual feature matching / reliability pipeline dependency

LightGlue code and LightGlue pretrained weights are released under the Apache License 2.0 by the upstream authors.

Important note: Some optional feature extractors or pretrained weights referenced by LightGlue, such as SuperPoint, may follow different and more restrictive license terms. Users should check the corresponding upstream licenses before redistribution, publication, or commercial use.

## Local modifications in this fork

This fork includes research-oriented modifications such as:
- ROS 2 Jazzy / Ubuntu 24.04 build and compatibility adjustments
- logging and debugging extensions
- experiment-specific output handling
- visual admission / reliability related changes

## Important notice

This repository is an unofficial research fork.
All trademarks, project names, and author attributions remain the property of their respective owners.
Original copyright and license notices in upstream-derived source files should be preserved.

### LightGlue

* Upstream: https://github.com/cvg/LightGlue
* License: Apache License 2.0
* Role in this repository: optional visual feature matching / reliability pipeline component.

This repository vendors the LightGlue source code for reproducibility and ease of setup.

The LightGlue code and LightGlue pretrained weights are released under the Apache License 2.0 by the upstream authors. However, some optional local feature extractors or pretrained weights referenced by LightGlue, such as SuperPoint, may follow different and more restrictive license terms. Users should check the corresponding upstream licenses before redistributing weights, pretrained models, or using them commercially.
