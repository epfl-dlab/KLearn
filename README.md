# KLearn

Code accompanying the paper: `KLearn: Background Knowledge Inference from Summarization Data` published at *Findings of EMNLP 2020*

### Abstract
The goal of text summarization is to compress documents to the relevant information while excluding background information already known to the receiver.
So far, summarization researchers have given considerably more attention to relevance than to background knowledge.
In contrast, this work puts background knowledge in the foreground.
Building on the realization that the choices made by human summarizers and annotators contain implicit information about their background knowledge, we develop and compare techniques for inferring background knowledge from summarization data.
Based on this framework, we define summary scoring functions that explicitly model background knowledge, and show that these scoring functions fit human judgments significantly better than baselines.
We illustrate some of the many potential applications of our framework.
First, we provide insights into human information importance priors.
Second, we demonstrate that averaging the background knowledge of multiple, potentially biased annotators or corpora greatly improves summary-scoring performance.
Finally, we discuss potential applications of our framework beyond summarization.

----
# Installation
    virtualenv klearn-env
    source klearn-env/bin/activate
    pip install -r requirements.txt

----
# Usage

* Download the preprocessed data [here](https://drive.google.com/file/d/1Pi3dcJ9rLFSsP6LXeiEr4rbbIfmWnUod/view?usp=sharing)

* [model_comparison.py](model_comparison.py) is a script that test different algorithms on TAC datasets.

* To learn annotator- and domain-specific Ks, use the script in experiments

* The jupyter notebook [K_analysis](experiments/K_analysis.ipynb) helps to reproduce the analysis.

* The main code is in [klearn](klearn)

## Contact
maxime.peyrard@epfl.ch
