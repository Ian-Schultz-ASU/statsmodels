// Passes for the dml dialect.
//
//   --dml-decompose             query ops -> building-block ops
//   --dml-decompose-no-pushdown same, without the average-effect rewrite
//   --dml-lower-to-linalg       building-block ops -> linalg/tensor/arith/math
//
// Between decompose and lower, MLIR's stock --cse pass removes duplicates.
//
// Every place where this file saves work is marked with a comment that starts
// with "EFFICIENCY". Search for that word to find them all:
//   1. Decompose to expose sharing   (DecomposePass, effect and interval)
//   2. Mean pushdown                 (DecomposePass, ate)
//   3. Intercept folding             (lowerCate)
//   4. No design-matrix copy         (lowerStderr)
//   5. One pass over X for all means (lowerColMean + CSE)
#include "DML/DML.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include <cmath>

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// --dml-decompose
//===----------------------------------------------------------------------===//

/// Rewrites each query op into building-block ops.
///
/// `Pushdown` selects how dml.ate is decomposed:
///   false: mean(theta(X)), one pass over X per model
///   true:  theta(mean(X)), one pass over X in total (see EFFICIENCY 2)
template <bool Pushdown>
struct DecomposePass
    : PassWrapper<DecomposePass<Pushdown>, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DecomposePass)
  StringRef getArgument() const final {
    return Pushdown ? "dml-decompose" : "dml-decompose-no-pushdown";
  }
  StringRef getDescription() const final {
    return "Split dml.effect / dml.effect_interval into cate, cate_stderr "
           "and elementwise ops so shared work becomes visible to CSE";
  }

  void runOnOperation() override {
    IRRewriter rewriter(&this->getContext());
    SmallVector<Operation *> queries;
    this->getOperation().walk([&](Operation *op) {
      if (isa<dml::EffectOp, dml::EffectIntervalOp, dml::AteOp>(op))
        queries.push_back(op);
    });
    for (Operation *op : queries) {
      rewriter.setInsertionPoint(op);
      Location loc = op->getLoc();
      // EFFICIENCY 1 (decompose to expose sharing): a query op hides the
      // expensive part, theta(X), inside it. Splitting it out as its own Pure
      // op lets CSE see that K scenarios all need the same theta.
      if (auto effect = dyn_cast<dml::EffectOp>(op)) {
        // effect = delta * theta(X)
        Type ty = effect.getEffect().getType();
        Value theta = rewriter.create<dml::CateOp>(loc, ty, effect.getX(),
                                                   effect.getBeta());
        Value out = rewriter.create<dml::ScaleOp>(loc, ty, theta,
                                                  effect.getDeltaAttr());
        rewriter.replaceOp(op, out);
        continue;
      }
      if (auto ate = dyn_cast<dml::AteOp>(op)) {
        Type ty = ate.getAte().getType();
        Value X = ate.getX();
        auto xTy = cast<RankedTensorType>(X.getType());
        Type f64 = rewriter.getF64Type();
        Value out;
        if (Pushdown) {
          // EFFICIENCY 2 (mean pushdown): theta is linear in X, so
          // mean(X @ b) == mean(X) @ b. col_mean depends on X only, so CSE
          // keeps one copy for every model and scenario on this population.
          auto barTy = RankedTensorType::get({xTy.getDimSize(1)}, f64);
          Value xbar = rewriter.create<dml::ColMeanOp>(loc, barTy, X);
          out = rewriter.create<dml::AteFromMeanOp>(loc, ty, xbar,
                                                    ate.getBeta(),
                                                    ate.getDeltaAttr());
        } else {
          // Baseline for the ablation: compute theta per row, then average
          auto thetaTy = RankedTensorType::get({xTy.getDimSize(0)}, f64);
          Value theta =
              rewriter.create<dml::CateOp>(loc, thetaTy, X, ate.getBeta());
          out = rewriter.create<dml::MeanScaleOp>(loc, ty, theta,
                                                  ate.getDeltaAttr());
        }
        rewriter.replaceOp(op, out);
        continue;
      }
      // effect_interval: lo/hi = delta * theta -/+ z * |delta| * stderr
      auto interval = cast<dml::EffectIntervalOp>(op);
      Type ty = interval.getLo().getType();
      double delta = interval.getDelta().convertToDouble();
      double half = interval.getZ().convertToDouble() * std::fabs(delta);

      // The two shared pieces: effect size and its standard error
      Value theta = rewriter.create<dml::CateOp>(loc, ty, interval.getX(),
                                                 interval.getBeta());
      Value se = rewriter.create<dml::CateStderrOp>(loc, ty, interval.getX(),
                                                    interval.getCov());

      // The per-scenario part is cheap elementwise arithmetic on those two
      auto f = [&](double v) { return rewriter.getF64FloatAttr(v); };
      Value lo =
          rewriter.create<dml::AxpbyOp>(loc, ty, theta, se, f(delta), f(-half));
      Value hi =
          rewriter.create<dml::AxpbyOp>(loc, ty, theta, se, f(delta), f(half));
      rewriter.replaceOp(op, {lo, hi});
    }
  }
};

//===----------------------------------------------------------------------===//
// --dml-lower-to-linalg
//===----------------------------------------------------------------------===//

struct LowerPass : PassWrapper<LowerPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerPass)
  StringRef getArgument() const final { return "dml-lower-to-linalg"; }
  StringRef getDescription() const final {
    return "Lower dml building-block ops to linalg, tensor, arith and math";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    math::MathDialect, tensor::TensorDialect>();
  }

  static Value constF64(OpBuilder &b, Location loc, double v) {
    return b.create<arith::ConstantOp>(loc, b.getF64Type(),
                                       b.getF64FloatAttr(v));
  }

  /// A tensor of `type` (rank 1 or 2, rows possibly dynamic, taken from X)
  /// filled with `value`.
  static Value filled(OpBuilder &b, Location loc, RankedTensorType type,
                      Value X, Value value) {
    SmallVector<Value> dyn;
    if (type.isDynamicDim(0))
      dyn.push_back(b.create<tensor::DimOp>(loc, X, 0));
    Value empty = b.create<tensor::EmptyOp>(loc, type, dyn);
    return b.create<linalg::FillOp>(loc, ValueRange{value}, ValueRange{empty})
        .getResult(0);
  }

  /// out[i] = body(ins[0][i], ins[1][i], ...) over rank-1 tensors.
  static Value elementwise(
      OpBuilder &b, Location loc, RankedTensorType type, ValueRange ins,
      function_ref<Value(OpBuilder &, Location, ValueRange)> body) {
    SmallVector<Value> dyn;
    if (type.isDynamicDim(0))
      dyn.push_back(b.create<tensor::DimOp>(loc, ins[0], 0));
    Value out = b.create<tensor::EmptyOp>(loc, type, dyn);
    AffineMap id = AffineMap::getMultiDimIdentityMap(1, b.getContext());
    SmallVector<AffineMap> maps(ins.size() + 1, id);
    SmallVector<utils::IteratorType> iters{utils::IteratorType::parallel};
    return b
        .create<linalg::GenericOp>(
            loc, TypeRange{type}, ins, ValueRange{out}, maps, iters,
            [&](OpBuilder &nb, Location nl, ValueRange args) {
              nb.create<linalg::YieldOp>(nl,
                                         body(nb, nl, args.drop_back(1)));
            })
        .getResult(0);
  }

  static int64_t numFeatures(Value X) {
    return cast<RankedTensorType>(X.getType()).getDimSize(1);
  }

  /// theta = beta[0] + X @ beta[1:]
  ///
  /// EFFICIENCY 3 (intercept folding): matvec computes out = init + X @ w.
  /// Starting `init` at beta[0] adds the intercept for free, so the n-by-(d+1)
  /// matrix [1, X] that EconML builds with a concatenate is never created.
  static Value lowerCate(dml::CateOp op, OpBuilder &b) {
    Location loc = op.getLoc();
    Value X = op.getX(), beta = op.getBeta();
    int64_t d = numFeatures(X);
    auto resultTy = cast<RankedTensorType>(op.getTheta().getType());

    // w = beta[1:], the d feature coefficients
    auto wTy = RankedTensorType::get({d}, b.getF64Type());
    Value w = b.create<tensor::ExtractSliceOp>(
        loc, wTy, beta, ArrayRef<OpFoldResult>{b.getIndexAttr(1)},
        ArrayRef<OpFoldResult>{b.getIndexAttr(d)},
        ArrayRef<OpFoldResult>{b.getIndexAttr(1)});

    // init = a length-n vector filled with the intercept beta[0]
    Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
    Value b0 = b.create<tensor::ExtractOp>(loc, beta, ValueRange{c0});
    Value init = filled(b, loc, resultTy, X, b0);

    // theta = init + X @ w
    return b
        .create<linalg::MatvecOp>(loc, TypeRange{resultTy}, ValueRange{X, w},
                                  ValueRange{init})
        .getResult(0);
  }

  /// With phi = [1, x] and cov = [[s00, c^T], [c, S]]:
  ///   var_i = s00 + sum_j x_ij * ((X S)_ij + 2 c_j);  stderr_i = sqrt(var_i)
  ///
  /// EFFICIENCY 4 (no design-matrix copy): expanding [1, x] cov [1, x]^T by
  /// hand turns the intercept row and column of cov into the scalar s00 and
  /// the vector c, so again [1, X] is never built. One matmul, one reduction.
  static Value lowerStderr(dml::CateStderrOp op, OpBuilder &b) {
    Location loc = op.getLoc();
    Value X = op.getX(), cov = op.getCov();
    int64_t d = numFeatures(X);
    auto xTy = cast<RankedTensorType>(X.getType());
    auto resultTy = cast<RankedTensorType>(op.getStderr().getType());
    Type f64 = b.getF64Type();
    auto idx = [&](int64_t v) -> OpFoldResult { return b.getIndexAttr(v); };


    // Split cov into s00 (scalar), c (d,) and S (d, d)
    Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
    Value s00 = b.create<tensor::ExtractOp>(loc, cov, ValueRange{c0, c0});
    Value c = b.create<tensor::ExtractSliceOp>(
        loc, RankedTensorType::get({d}, f64), cov,
        ArrayRef<OpFoldResult>{idx(0), idx(1)},
        ArrayRef<OpFoldResult>{idx(1), idx(d)},
        ArrayRef<OpFoldResult>{idx(1), idx(1)});
    Value S = b.create<tensor::ExtractSliceOp>(
        loc, RankedTensorType::get({d, d}, f64), cov,
        ArrayRef<OpFoldResult>{idx(1), idx(1)},
        ArrayRef<OpFoldResult>{idx(d), idx(d)},
        ArrayRef<OpFoldResult>{idx(1), idx(1)});

    // XS = X @ S, shape (n, d)
    Value zero = constF64(b, loc, 0.0);
    Value XS = b.create<linalg::MatmulOp>(
                    loc, TypeRange{xTy}, ValueRange{X, S},
                    ValueRange{filled(b, loc, xTy, X, zero)})
                   .getResult(0);

    // var[i] = s00 + sum over j of X[i,j] * (XS[i,j] + 2 * c[j])
    // The four maps say how each operand is indexed inside the (i, j) loop:
    // X[i,j], XS[i,j], c[j] and the output var[i].
    MLIRContext *ctx = b.getContext();
    AffineExpr i, j;
    bindDims(ctx, i, j);
    SmallVector<AffineMap> maps{
        AffineMap::get(2, 0, {i, j}, ctx), AffineMap::get(2, 0, {i, j}, ctx),
        AffineMap::get(2, 0, {j}, ctx), AffineMap::get(2, 0, {i}, ctx)};
    SmallVector<utils::IteratorType> iters{utils::IteratorType::parallel,
                                           utils::IteratorType::reduction};
    Value var =
        b.create<linalg::GenericOp>(
             loc, TypeRange{resultTy}, ValueRange{X, XS, c},
             ValueRange{filled(b, loc, resultTy, X, s00)}, maps, iters,
             [&](OpBuilder &nb, Location nl, ValueRange a) {
               Value two = constF64(nb, nl, 2.0);
               Value twoC = nb.create<arith::MulFOp>(nl, two, a[2]);
               Value inner = nb.create<arith::AddFOp>(nl, a[1], twoC);
               Value term = nb.create<arith::MulFOp>(nl, a[0], inner);
               Value acc = nb.create<arith::AddFOp>(nl, a[3], term);
               nb.create<linalg::YieldOp>(nl, acc);
             })
            .getResult(0);

    // stderr = sqrt(var)
    return elementwise(b, loc, resultTy, ValueRange{var},
                       [](OpBuilder &nb, Location nl, ValueRange a) -> Value {
                         return nb.create<math::SqrtOp>(nl, a[0]);
                       });
  }

  /// Row count of X as an f64 scalar.
  static Value rowsF64(OpBuilder &b, Location loc, Value X) {
    Value n = b.create<tensor::DimOp>(loc, X, 0);
    Value n64 = b.create<arith::IndexCastOp>(loc, b.getI64Type(), n);
    return b.create<arith::SIToFPOp>(loc, b.getF64Type(), n64);
  }

  static Value scalarTensor(OpBuilder &b, Location loc, Value v) {
    return b.create<tensor::FromElementsOp>(
        loc, RankedTensorType::get({}, b.getF64Type()), ValueRange{v});
  }

  /// out = a * mean(x): one reduction into a rank-0 tensor.
  static Value lowerMeanScale(dml::MeanScaleOp op, OpBuilder &b) {
    Location loc = op.getLoc();
    Value x = op.getX();
    MLIRContext *ctx = b.getContext();
    auto scalarTy = RankedTensorType::get({}, b.getF64Type());
    Value empty = b.create<tensor::EmptyOp>(loc, scalarTy, ValueRange{});
    Value init = b.create<linalg::FillOp>(loc,
                                          ValueRange{constF64(b, loc, 0.0)},
                                          ValueRange{empty})
                     .getResult(0);
    AffineExpr i;
    bindDims(ctx, i);
    SmallVector<AffineMap> maps{AffineMap::get(1, 0, {i}, ctx),
                                AffineMap::get(1, 0, {}, ctx)};
    SmallVector<utils::IteratorType> iters{utils::IteratorType::reduction};
    Value sum = b.create<linalg::GenericOp>(
                     loc, TypeRange{scalarTy}, ValueRange{x},
                     ValueRange{init}, maps, iters,
                     [&](OpBuilder &nb, Location nl, ValueRange a) {
                       nb.create<linalg::YieldOp>(
                           nl, nb.create<arith::AddFOp>(nl, a[1], a[0])
                                   .getResult());
                     })
                    .getResult(0);
    Value s = b.create<tensor::ExtractOp>(loc, sum, ValueRange{});
    Value mean = b.create<arith::DivFOp>(loc, s, rowsF64(b, loc, x));
    Value out = b.create<arith::MulFOp>(
        loc, mean, constF64(b, loc, op.getA().convertToDouble()));
    return scalarTensor(b, loc, out);
  }

  /// xbar[j] = sum_i X[i,j] / n
  ///
  /// EFFICIENCY 5 (one pass over X for all means): after pushdown and CSE this
  /// is the only loop over the n rows, however many models and scenarios the
  /// analyst asked about. Everything downstream works on d numbers.
  static Value lowerColMean(dml::ColMeanOp op, OpBuilder &b) {
    Location loc = op.getLoc();
    Value X = op.getX();
    MLIRContext *ctx = b.getContext();
    auto barTy = cast<RankedTensorType>(op.getXbar().getType());
    Value empty = b.create<tensor::EmptyOp>(loc, barTy, ValueRange{});
    Value init = b.create<linalg::FillOp>(loc,
                                          ValueRange{constF64(b, loc, 0.0)},
                                          ValueRange{empty})
                     .getResult(0);
    AffineExpr i, j;
    bindDims(ctx, i, j);
    SmallVector<AffineMap> maps{AffineMap::get(2, 0, {i, j}, ctx),
                                AffineMap::get(2, 0, {j}, ctx)};
    SmallVector<utils::IteratorType> iters{utils::IteratorType::reduction,
                                           utils::IteratorType::parallel};
    Value sums = b.create<linalg::GenericOp>(
                      loc, TypeRange{barTy}, ValueRange{X}, ValueRange{init},
                      maps, iters,
                      [&](OpBuilder &nb, Location nl, ValueRange a) {
                        nb.create<linalg::YieldOp>(
                            nl, nb.create<arith::AddFOp>(nl, a[1], a[0])
                                    .getResult());
                      })
                     .getResult(0);
    Value n = rowsF64(b, loc, X);
    return elementwise(b, loc, barTy, ValueRange{sums},
                       [&](OpBuilder &nb, Location nl, ValueRange a) -> Value {
                         return nb.create<arith::DivFOp>(nl, a[0], n);
                       });
  }

  /// ate = delta * (beta[0] + xbar . beta[1:])
  /// Costs d multiplications per (model, scenario). It never touches X.
  static Value lowerAteFromMean(dml::AteFromMeanOp op, OpBuilder &b) {
    Location loc = op.getLoc();
    Value xbar = op.getXbar(), beta = op.getBeta();
    int64_t d = cast<RankedTensorType>(xbar.getType()).getDimSize(0);
    auto scalarTy = RankedTensorType::get({}, b.getF64Type());
    Value w = b.create<tensor::ExtractSliceOp>(
        loc, RankedTensorType::get({d}, b.getF64Type()), beta,
        ArrayRef<OpFoldResult>{b.getIndexAttr(1)},
        ArrayRef<OpFoldResult>{b.getIndexAttr(d)},
        ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
    Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
    Value b0 = b.create<tensor::ExtractOp>(loc, beta, ValueRange{c0});
    Value empty = b.create<tensor::EmptyOp>(loc, scalarTy, ValueRange{});
    Value init =
        b.create<linalg::FillOp>(loc, ValueRange{b0}, ValueRange{empty})
            .getResult(0);
    Value dot = b.create<linalg::DotOp>(loc, TypeRange{scalarTy},
                                        ValueRange{xbar, w}, ValueRange{init})
                    .getResult(0);
    Value s = b.create<tensor::ExtractOp>(loc, dot, ValueRange{});
    Value out = b.create<arith::MulFOp>(
        loc, s, constF64(b, loc, op.getDelta().convertToDouble()));
    return scalarTensor(b, loc, out);
  }

  void runOnOperation() override {
    IRRewriter rewriter(&getContext());
    SmallVector<Operation *> ops;
    bool failed = false;
    getOperation().walk([&](Operation *op) {
      if (isa<dml::EffectOp, dml::EffectIntervalOp, dml::AteOp>(op)) {
        op->emitOpError("run --dml-decompose before --dml-lower-to-linalg");
        failed = true;
      }
      if (isa<dml::CateOp, dml::CateStderrOp, dml::ScaleOp, dml::AxpbyOp,
              dml::MeanScaleOp, dml::ColMeanOp, dml::AteFromMeanOp>(op))
        ops.push_back(op);
    });
    if (failed)
      return signalPassFailure();

    for (Operation *op : ops) {
      rewriter.setInsertionPoint(op);
      Location loc = op->getLoc();
      Value result;
      if (auto cate = dyn_cast<dml::CateOp>(op)) {
        if (numFeatures(cate.getX()) == ShapedType::kDynamic) {
          op->emitOpError("feature count must be static");
          return signalPassFailure();
        }
        result = lowerCate(cate, rewriter);
      } else if (auto se = dyn_cast<dml::CateStderrOp>(op)) {
        if (numFeatures(se.getX()) == ShapedType::kDynamic) {
          op->emitOpError("feature count must be static");
          return signalPassFailure();
        }
        result = lowerStderr(se, rewriter);
      } else if (auto ms = dyn_cast<dml::MeanScaleOp>(op)) {
        result = lowerMeanScale(ms, rewriter);
      } else if (auto cm = dyn_cast<dml::ColMeanOp>(op)) {
        result = lowerColMean(cm, rewriter);
      } else if (auto afm = dyn_cast<dml::AteFromMeanOp>(op)) {
        result = lowerAteFromMean(afm, rewriter);
      } else if (auto scale = dyn_cast<dml::ScaleOp>(op)) {
        double a = scale.getA().convertToDouble();
        result = elementwise(
            rewriter, loc, cast<RankedTensorType>(scale.getOut().getType()),
            ValueRange{scale.getX()},
            [&](OpBuilder &nb, Location nl, ValueRange v) -> Value {
              return nb.create<arith::MulFOp>(nl, v[0], constF64(nb, nl, a));
            });
      } else {
        auto axpby = cast<dml::AxpbyOp>(op);
        double a = axpby.getA().convertToDouble();
        double bb = axpby.getB().convertToDouble();
        result = elementwise(
            rewriter, loc, cast<RankedTensorType>(axpby.getOut().getType()),
            ValueRange{axpby.getX(), axpby.getY()},
            [&](OpBuilder &nb, Location nl, ValueRange v) -> Value {
              Value ax =
                  nb.create<arith::MulFOp>(nl, v[0], constF64(nb, nl, a));
              Value by =
                  nb.create<arith::MulFOp>(nl, v[1], constF64(nb, nl, bb));
              return nb.create<arith::AddFOp>(nl, ax, by);
            });
      }
      rewriter.replaceOp(op, result);
    }
  }
};

} // namespace

void dml::registerDMLPasses() {
  PassRegistration<DecomposePass<true>>();
  PassRegistration<DecomposePass<false>>();
  PassRegistration<LowerPass>();
}
