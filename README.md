This repository is an updated version of https://github.com/OmnesRes/ATGC_old. We have created a framework for performing multiple instance learning using ragged tensors, and applied the tool to somatic variants.  The code is written in Python 3 and was run with TensorFlow 2.2.0.

We've achieved best-in-class performance for every problem we've applied the tool to (tumor classification, clonality, microsatellite instability, tumor mutational burden), and will be creating a new branch for every publication that uses this tool.  For example, the method_paper branch corresponds to this manuscript: https://www.biorxiv.org/content/10.1101/2020.08.05.237206v4.  We will also be creating a zenodo release for each manuscript.

The easiest way to get started would be to follow along with the IPython notebook: https://github.com/OmnesRes/ATGC/blob/method_paper/examples/example.ipynb

Although applied to somatic mutations, this model could be used for any problem involving MIL given an appropriate encoder module and appropriate loss function.
