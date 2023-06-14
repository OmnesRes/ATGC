# ATGC
## A multiple instance learning framework
Aggregation Tool for Genomic Concepts (ATGC) implements MIL in TensorFlow using ragged tensors.  Most implementations of MIL require stochastic gradient descent, in contrast our implementation allows users to train models as they would any other model.  Our MIL follows that in https://github.com/AMLab-Amsterdam/AttentionDeepMIL, but we calculate attention with a single network and an adaptive activation.  Our model allows for any number of attention heads, inputs, and outputs.

## Publication
Currently in genomics when predictions are made at the sample level with sparse instance data static hand-crafted features are created, summarised with a mean or a sum, then a model is chosen for predictions.  In contrast, we encode the instances, aggregate, and make predictions with a single model.  This end-to-end approach allows for novel encodings that learn directly from the data, and allows for the calculation of attention values for each instance.  To show the benefit of this approach we applied this tool to somatic mutations, and all the code to reproduce the following manuscript is included in this repository: "Aggregation Tool for Genomic Concepts (ATGC): A deep learning framework for sparse genomic measures and its application to somatic mutations": https://www.biorxiv.org/content/10.1101/2020.08.05.237206v4. We developed a novel method for representing the sequence of somatic mutations, and applied an embedding strategy for genes and genomic position.  However the model can be used with established features and the attention our model calculates may lead to performance benefits or model explainability.  In particular the embedding strategy can allow for high-dimensional inputs that traditional neural nets would not be able to handle.

## Dependencies
The model was run with Python 3.8, TensorFlow 2.7.0.

## Use
Running the model simply requires importing the code from the "model" folder.  An example with simulated somatic mutation data is available: https://github.com/OmnesRes/ATGC/blob/method_paper/examples/example.ipynb
