# Pyhabitats

## What pyhabitats is
It is a Python library designed for habitat analysis of radiomic images. It provides an interface for such analysis pipeline from loading images from disk, through identifying habitats to radiomic features extraction. It offers predefined clusterization algorithms for habitat identification and support custom ones.

## Main features
Library provides the following features:

1. Loading medical images from disk. Supports wide range of formats designed for medical imaging (e.g. .nii.gz, .nrrd, .png and more).
2. Habitats identification. Uses clusterization algorithms to find similar sub-regions in ROI (region of interest). The number of habitats can be provided by user.
3. Radiomic features extraction. Utilizes pyradiomics library to extract radiomic features for each identified habitat.

## Contributing
[//]: # (Add link to gh issues)

## License
[//]: # (Add license)

[//]: # (Consider adding list of supported formats somewhere in the docs)