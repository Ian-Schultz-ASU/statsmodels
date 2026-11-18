#include "Causal/CausalOps.h"
#include "Causal/CausalPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace causal {
namespace {

class LowerTreatmentEffectPass
    : public mlir::PassWrapper<LowerTreatmentEffectPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerTreatmentEffectPass)

  llvm::StringRef getArgument() const final {
    return "ci-lower-treatment-effect";
  }

  llvm::StringRef getDescription() const final {
    return "Lower ci.treatment_effect to linalg.matvec and linalg.generic";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect, mlir::linalg::LinalgDialect,
                    mlir::tensor::TensorDialect>();
  }

  void runOnOperation() override {
    mlir::IRRewriter rewriter(&getContext());
    llvm::SmallVector<TreatmentEffectOp> effects;
    getOperation().walk(
        [&](TreatmentEffectOp effect) { effects.push_back(effect); });

    for (TreatmentEffectOp effect : effects) {
      lower(effect, rewriter);
    }
  }

private:
  static void lower(TreatmentEffectOp effect, mlir::IRRewriter &rewriter) {
    mlir::Location loc = effect.getLoc();
    mlir::Value data = effect.getData();
    mlir::Value coefficients = effect.getCoefficients();
    auto resultType =
        llvm::cast<mlir::RankedTensorType>(effect.getEffects().getType());

    rewriter.setInsertionPoint(effect);

    // 1. Gather the interaction coefficients into a contiguous 1-D tensor.
    auto interactionIndices = effect.getInteractionCoefficients();
    llvm::SmallVector<mlir::Value> weightValues;
    weightValues.reserve(interactionIndices.size());
    for (int64_t idx : interactionIndices) {
      mlir::Value cIdx = rewriter.create<mlir::arith::ConstantIndexOp>(loc, idx);
      weightValues.push_back(rewriter.create<mlir::tensor::ExtractOp>(
          loc, coefficients, mlir::ValueRange{cIdx}));
    }
    auto weightType = mlir::RankedTensorType::get(
        {static_cast<int64_t>(interactionIndices.size())},
        rewriter.getF64Type());
    mlir::Value weights = rewriter.create<mlir::tensor::FromElementsOp>(
        loc, weightType, weightValues);

    // 2. Compute data @ weights via linalg.matvec.
    llvm::SmallVector<mlir::Value> dynamicSizes;
    if (resultType.isDynamicDim(0)) {
      dynamicSizes.push_back(
          rewriter.create<mlir::tensor::DimOp>(loc, data, 0));
    }
    mlir::Value emptyInit = rewriter.create<mlir::tensor::EmptyOp>(
        loc, resultType, dynamicSizes);
    mlir::Value zero = rewriter.create<mlir::arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(0.0));
    mlir::Value zeroed =
        rewriter
            .create<mlir::linalg::FillOp>(
                loc, mlir::ValueRange{zero}, mlir::ValueRange{emptyInit})
            .getResult(0);
    mlir::Value dot =
        rewriter
            .create<mlir::linalg::MatvecOp>(
                loc, resultType, mlir::ValueRange{data, weights},
                mlir::ValueRange{zeroed})
            .getResult(0);

    // 3. Add main_effect and scale by (treated_value - control_value).
    mlir::Value cMainIdx = rewriter.create<mlir::arith::ConstantIndexOp>(
        loc, effect.getMainEffectIndex());
    mlir::Value mainEffect = rewriter.create<mlir::tensor::ExtractOp>(
        loc, coefficients, mlir::ValueRange{cMainIdx});
    double treatmentDelta = effect.getTreatedValue().convertToDouble()
        - effect.getControlValue().convertToDouble();
    mlir::Value delta = rewriter.create<mlir::arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(treatmentDelta));

    mlir::Value emptyOut = rewriter.create<mlir::tensor::EmptyOp>(
        loc, resultType, dynamicSizes);
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::AffineMap idMap = mlir::AffineMap::getMultiDimIdentityMap(1, ctx);
    llvm::SmallVector<mlir::AffineMap> indexingMaps{idMap, idMap};
    llvm::SmallVector<mlir::utils::IteratorType> iteratorTypes{
        mlir::utils::IteratorType::parallel};

    auto scaled = rewriter.create<mlir::linalg::GenericOp>(
        loc,
        mlir::TypeRange{resultType},
        mlir::ValueRange{dot},
        mlir::ValueRange{emptyOut},
        indexingMaps,
        iteratorTypes,
        [&](mlir::OpBuilder &builder,
            mlir::Location bodyLoc,
            mlir::ValueRange args) {
          mlir::Value sum =
              builder.create<mlir::arith::AddFOp>(bodyLoc, args[0], mainEffect);
          mlir::Value rowEffect =
              builder.create<mlir::arith::MulFOp>(bodyLoc, sum, delta);
          builder.create<mlir::linalg::YieldOp>(bodyLoc, rowEffect);
        });

    rewriter.replaceOp(effect, scaled.getResult(0));
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createLowerTreatmentEffectPass() {
  return std::make_unique<LowerTreatmentEffectPass>();
}

void registerLowerTreatmentEffectPass() {
  mlir::PassRegistration<LowerTreatmentEffectPass>();
}

} // namespace causal
