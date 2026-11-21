#ifndef DML_OPS_TD
#define DML_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def DML_Dialect : Dialect {
  let name = "dml";
  let cppNamespace = "::dml";
  let summary = "Inference path of linear Double ML (post-fit effect queries)";
}

// EFFICIENCY (Pure ops enable CSE): every op is declared Pure, meaning it has
// no side effects. That one trait is what allows MLIR's stock --cse pass to
// merge repeated computations, with no analysis code of our own.
class DML_Op<string mnemonic> : Op<DML_Dialect, mnemonic, [Pure]> {
  // Shape rules are checked in lib/DML/DMLDialect.cpp.
  let hasVerifier = 1;
}

def F64Matrix : RankedTensorOf<[F64]>;

//===-- Query-level ops: what the analyst writes, one per question --------===//

def DML_EffectOp : DML_Op<"effect"> {
  let summary = "effect(X, T0, T1) = (T1 - T0) * theta(X)";
  let arguments = (ins F64Matrix:$X, F64Matrix:$beta, F64Attr:$delta);
  let results = (outs F64Matrix:$effect);
}

def DML_EffectIntervalOp : DML_Op<"effect_interval"> {
  let summary = "Normal confidence interval of effect(X, T0, T1)";
  let description = [{
    lo/hi = delta * theta(X) -/+ z * |delta| * stderr(theta(X)).
    `z` is the normal quantile for the requested confidence level.
  }];
  let arguments = (ins F64Matrix:$X, F64Matrix:$beta, F64Matrix:$cov,
                       F64Attr:$delta, F64Attr:$z);
  let results = (outs F64Matrix:$lo, F64Matrix:$hi);
}

def DML_AteOp : DML_Op<"ate"> {
  let summary = "Average effect over the population: mean_i(delta * theta(X_i))";
  let arguments = (ins F64Matrix:$X, F64Matrix:$beta, F64Attr:$delta);
  let results = (outs F64Matrix:$ate);
}

//===-- Building-block ops: produced by --dml-decompose -------------------===//

def DML_CateOp : DML_Op<"cate"> {
  let summary = "theta(X) = beta[0] + X @ beta[1:]";
  let arguments = (ins F64Matrix:$X, F64Matrix:$beta);
  let results = (outs F64Matrix:$theta);
}

def DML_CateStderrOp : DML_Op<"cate_stderr"> {
  let summary = "sqrt(diag([1,X] cov [1,X]^T)), computed row-wise";
  let arguments = (ins F64Matrix:$X, F64Matrix:$cov);
  let results = (outs F64Matrix:$stderr);
}

def DML_ScaleOp : DML_Op<"scale"> {
  let summary = "out = a * x";
  let arguments = (ins F64Matrix:$x, F64Attr:$a);
  let results = (outs F64Matrix:$out);
}

def DML_AxpbyOp : DML_Op<"axpby"> {
  let summary = "out = a * x + b * y";
  let arguments = (ins F64Matrix:$x, F64Matrix:$y, F64Attr:$a, F64Attr:$b);
  let results = (outs F64Matrix:$out);
}

def DML_MeanScaleOp : DML_Op<"mean_scale"> {
  let summary = "out = a * mean(x), rank-0 result";
  let arguments = (ins F64Matrix:$x, F64Attr:$a);
  let results = (outs F64Matrix:$out);
}

def DML_ColMeanOp : DML_Op<"col_mean"> {
  let summary = "xbar[j] = mean_i X[i, j]; depends on X only, so it is shared "
                "by every model and scenario scored on the same population";
  let arguments = (ins F64Matrix:$X);
  let results = (outs F64Matrix:$xbar);
}

def DML_AteFromMeanOp : DML_Op<"ate_from_mean"> {
  let summary = "ate = delta * (beta[0] + xbar . beta[1:])";
  let arguments = (ins F64Matrix:$xbar, F64Matrix:$beta, F64Attr:$delta);
  let results = (outs F64Matrix:$ate);
}

#endif // DML_OPS_TD
