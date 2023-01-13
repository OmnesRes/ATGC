# ATGC
## A multiple instance learning framework
Aggregation Tool for Genomic Concepts (ATGC) implements MIL in TensorFlow using ragged tensors.  Most implementations of MIL require stochastic gradient descent, in contrast our implementation allows users to train models as they would any other model.  Our MIL follows that in https://github.com/AMLab-Amsterdam/AttentionDeepMIL, but we calculate attention with a single network and an adaptive activation.  Our model allows for any number of attention heads, inputs, and outputs, albeit dynamic attention currently only supports a single attention head.

## Publication
We applied our tool to somatic mutations, and all the code to reproduce the following manuscript is included in this repository: "Aggregation Tool for Genomic Concepts (ATGC): A deep learning framework for sparse genomic measures and its application to somatic mutations": https://www.biorxiv.org/content/10.1101/2020.08.05.237206v4.  To apply the tool to your data all that is required is an appropriate encoder written in TensorFlow, see the "InstanceModels" located here: https://github.com/OmnesRes/ATGC/blob/method_paper/model/Sample_MIL.py.

## Dependencies
The model was run with Python 3.8, TensorFlow 2.7.0.

## Use
Running the model simply requires importing the code from the "model" folder.  An example with simulated somatic mutation data is available: https://github.com/OmnesRes/ATGC/blob/method_paper/examples/example.ipynb
