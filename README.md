This repository is an updated version of https://github.com/OmnesRes/ATGC, and represents the model and results for the most recent version of this preprint: https://doi.org/10.1101/2020.08.05.237206.

We have created a framework for performing multiple instance learning using ragged tensors.  The code is written in Python 3 and was run with TensorFlow 2.2.0.

The easiest way to get started would be to run one of the simulated experiments: https://github.com/OmnesRes/ATGC2/tree/master/figures/controls/samples/sim_data.

Although applied to somatic mutations, this model could be used for any problem involving MIL given an appropriate encoder module and appropriate loss function.
