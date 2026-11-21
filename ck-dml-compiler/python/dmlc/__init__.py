"""dmlc: a small compiler for the inference path of linear Double ML.

Modules
-------
reference  NumPy implementation of every quantity (the correctness oracle)
data       synthetic data and the EconML model used as the external oracle
emit       writes a batch of queries as MLIR text in the `dml` dialect
pipeline   runs dml-opt, compiles with IREE, executes the result
"""
