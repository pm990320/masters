# masters

Patrick Menlove (2250066M) masters project 2019-2020.

> Multivariate time-series anomaly detection applied to key business data

All code for the project is included here, except for that code used to generate graphs from Skyscanner's data.

## Structure

`dissertation/` - contains the LaTeX for the final report

`interimreport/` - contains the LaTeX for the interim report

`papers/` - contains copies of some of the papers used in literature review

`code/` - contains the evaluation framework code & models implementations. The `notebooks/` subfolder also contains Jupyter notebooks used for the creation of graphs and exploratory code.

## Setup

To run the code, you should use a Conda environment. You can either install the full Anaconda distribution or Miniconda.

```
conda env create -f environment.yml
```

## Run the Evaluation code

This will take some time to run!

```
cd code
make evaluate
```
