# Combining Statistical and Graph-Based Approaches to Classification of Interstitial Pulmonary Diseases in High-Resolution Computed Tomography Images

## Abstract

Problems of texture classification are consistently challenging once the patterns of different instances can be very similar. Moreover, the descriptors need to be invariant to rotations, scale, and lighting variations. In the context of medical imaging, this group of methods can aid in diagnosing patients as part of the concept of Computer-Aided Diagnosis (CAD). In this paper, we propose a method for texture classification in the context of classifying Interstitial Pulmonary Diseases (IPDs) on high-resolution Computed Tomographies (CTs) using concepts of complex networks and statistical metrics. Our approach is based on mapping the input image into multiscale graphs and extracting the closeness centrality metric. We transform the multiscale closeness centrality images into one matrix that encapsulates local and global texture information. From the matrix, we extract a feature vector that represents an IPD pattern. A final feature vector combines it with Haralick and Local Binary Pattern descriptors. Once this process characterizes all the images, we go through a classification step to recognize the image. We analyze the proposed approach’s performance by comparing it with other methods and discussing its metrics for each class (IPD pattern) of the dataset. Based on the results, we can highlight our technique as an aid on the problem of diagnosing patients with COVID-19