// The dml dialect: registration and the shape rules (verifiers) for every op.
//
// Notation used in the messages: X is (n, d), beta is (p,) with p = d + 1
// (intercept first), cov is (p, p). A dynamic dimension ("?") skips the
// corresponding check, because its size is only known at run time.
#include "DML/DML.h"

#include "DML/DMLOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "DML/DMLOps.cpp.inc"

using namespace mlir;

void dml::DMLDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "DML/DMLOps.cpp.inc"
      >();
}

namespace {

RankedTensorType tensorTy(Value v) {
  return llvm::cast<RankedTensorType>(v.getType());
}

bool known(int64_t dim) { return !ShapedType::isDynamic(dim); }

/// `v` must have exactly `rank` dimensions.
LogicalResult checkRank(Operation *op, Value v, int64_t rank, StringRef name) {
  if (tensorTy(v).getRank() != rank)
    return op->emitOpError() << name << " must be a rank-" << rank
                             << " tensor, got rank " << tensorTy(v).getRank();
  return success();
}

/// Dimension `dimA` of `a` must equal dimension `dimB` of `b` plus `offset`.
LogicalResult checkDim(Operation *op, Value a, unsigned dimA, Value b,
                       unsigned dimB, int64_t offset, StringRef message) {
  int64_t sizeA = tensorTy(a).getDimSize(dimA);
  int64_t sizeB = tensorTy(b).getDimSize(dimB);
  if (known(sizeA) && known(sizeB) && sizeA != sizeB + offset)
    return op->emitOpError() << message << " (got " << sizeA << " and "
                             << sizeB << ")";
  return success();
}

/// X is (n, d) and beta is (d + 1,).
LogicalResult checkXBeta(Operation *op, Value X, Value beta) {
  if (failed(checkRank(op, X, 2, "X")) ||
      failed(checkRank(op, beta, 1, "beta")))
    return failure();
  return checkDim(op, beta, 0, X, 1, 1,
                  "beta must have one entry per feature plus an intercept: "
                  "len(beta) == X.shape[1] + 1");
}

/// X is (n, d) and cov is (d + 1, d + 1).
LogicalResult checkXCov(Operation *op, Value X, Value cov) {
  if (failed(checkRank(op, X, 2, "X")) || failed(checkRank(op, cov, 2, "cov")))
    return failure();
  StringRef message = "cov must be (p, p) with p == X.shape[1] + 1";
  if (failed(checkDim(op, cov, 0, X, 1, 1, message)) ||
      failed(checkDim(op, cov, 1, X, 1, 1, message)))
    return failure();
  return success();
}

/// A per-row result: rank 1 with as many entries as `rowsOf` has rows.
LogicalResult checkPerRow(Operation *op, Value result, Value rowsOf,
                          StringRef name) {
  if (failed(checkRank(op, result, 1, name)))
    return failure();
  return checkDim(op, result, 0, rowsOf, 0, 0,
                  "per-row result must have one entry per input row");
}

} // namespace

//===-- Query ops ---------------------------------------------------------===//

LogicalResult dml::EffectOp::verify() {
  Operation *op = getOperation();
  if (failed(checkXBeta(op, getX(), getBeta())))
    return failure();
  return checkPerRow(op, getEffect(), getX(), "effect");
}

LogicalResult dml::EffectIntervalOp::verify() {
  Operation *op = getOperation();
  if (failed(checkXBeta(op, getX(), getBeta())) ||
      failed(checkXCov(op, getX(), getCov())) ||
      failed(checkPerRow(op, getLo(), getX(), "lo")) ||
      failed(checkPerRow(op, getHi(), getX(), "hi")))
    return failure();
  if (getZ().convertToDouble() <= 0.0)
    return emitOpError("z must be positive");
  return success();
}

LogicalResult dml::AteOp::verify() {
  Operation *op = getOperation();
  if (failed(checkXBeta(op, getX(), getBeta())))
    return failure();
  return checkRank(op, getAte(), 0, "ate");
}

//===-- Building-block ops ------------------------------------------------===//

LogicalResult dml::CateOp::verify() {
  Operation *op = getOperation();
  if (failed(checkXBeta(op, getX(), getBeta())))
    return failure();
  return checkPerRow(op, getTheta(), getX(), "theta");
}

LogicalResult dml::CateStderrOp::verify() {
  Operation *op = getOperation();
  if (failed(checkXCov(op, getX(), getCov())))
    return failure();
  return checkPerRow(op, getStderr(), getX(), "stderr");
}

LogicalResult dml::ScaleOp::verify() {
  Operation *op = getOperation();
  if (failed(checkRank(op, getX(), 1, "x")))
    return failure();
  return checkPerRow(op, getOut(), getX(), "out");
}

LogicalResult dml::AxpbyOp::verify() {
  Operation *op = getOperation();
  if (failed(checkRank(op, getX(), 1, "x")) ||
      failed(checkRank(op, getY(), 1, "y")) ||
      failed(checkDim(op, getX(), 0, getY(), 0, 0,
                      "x and y must have the same length")))
    return failure();
  return checkPerRow(op, getOut(), getX(), "out");
}

LogicalResult dml::MeanScaleOp::verify() {
  Operation *op = getOperation();
  if (failed(checkRank(op, getX(), 1, "x")))
    return failure();
  return checkRank(op, getOut(), 0, "out");
}

LogicalResult dml::ColMeanOp::verify() {
  Operation *op = getOperation();
  if (failed(checkRank(op, getX(), 2, "X")) ||
      failed(checkRank(op, getXbar(), 1, "xbar")))
    return failure();
  return checkDim(op, getXbar(), 0, getX(), 1, 0,
                  "xbar must have one entry per feature: "
                  "len(xbar) == X.shape[1]");
}

LogicalResult dml::AteFromMeanOp::verify() {
  Operation *op = getOperation();
  if (failed(checkRank(op, getXbar(), 1, "xbar")) ||
      failed(checkRank(op, getBeta(), 1, "beta")) ||
      failed(checkDim(op, getBeta(), 0, getXbar(), 0, 1,
                      "beta must have one entry per feature plus an "
                      "intercept: len(beta) == len(xbar) + 1")))
    return failure();
  return checkRank(op, getAte(), 0, "ate");
}
