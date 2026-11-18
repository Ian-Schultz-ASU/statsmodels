#include "Causal/CausalOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseSet.h"
#include <cmath>

#define GET_OP_CLASSES
#include "Causal/CausalOps.cpp.inc"

mlir::LogicalResult causal::TreatmentEffectOp::verify() {
  auto dataType = llvm::dyn_cast<mlir::RankedTensorType>(getData().getType());
  auto coeffType =
      llvm::dyn_cast<mlir::RankedTensorType>(getCoefficients().getType());
  auto effectsType =
      llvm::dyn_cast<mlir::RankedTensorType>(getEffects().getType());

  if (!dataType || dataType.getRank() != 2) {
    return emitOpError("data operand must be a rank-2 tensor");
  }
  if (!coeffType || coeffType.getRank() != 1) {
    return emitOpError("coefficients operand must be a rank-1 tensor");
  }
  if (!effectsType || effectsType.getRank() != 1) {
    return emitOpError("result must be a rank-1 tensor");
  }
  if (!dataType.getElementType().isF64() || !coeffType.getElementType().isF64()
      || !effectsType.getElementType().isF64()) {
    return emitOpError("all tensor elements must be f64");
  }
  if (getTreatmentName().empty()) {
    return emitOpError("treatment_name must not be empty");
  }
  if (!std::isfinite(getControlValue().convertToDouble())
      || !std::isfinite(getTreatedValue().convertToDouble())
      || getControlValue() == getTreatedValue()) {
    return emitOpError(
        "control_value and treated_value must be distinct finite values");
  }
  if (getMainEffectIndex() < 0
      || (coeffType.hasStaticShape()
          && getMainEffectIndex() >= coeffType.getDimSize(0))) {
    return emitOpError("main_effect_index is outside the coefficient vector");
  }
  if (getInteractionColumns().size() != getInteractionCoefficients().size()) {
    return emitOpError(
        "interaction_columns and interaction_coefficients must have equal "
        "lengths");
  }
  if (getInteractionColumns().empty()) {
    return emitOpError("requires at least one treatment interaction");
  }
  if (!dataType.isDynamicDim(1)
      && dataType.getDimSize(1)
          != static_cast<int64_t>(getInteractionColumns().size())) {
    return emitOpError("data column count must match interaction_columns");
  }
  if (effectsType.hasStaticShape() && dataType.hasStaticShape()
      && effectsType.getDimSize(0) != dataType.getDimSize(0)) {
    return emitOpError("result row count must match data row count");
  }

  llvm::SmallDenseSet<int64_t, 8> seenIndices;
  for (int64_t index : getInteractionCoefficients()) {
    if (index < 0
        || (coeffType.hasStaticShape() && index >= coeffType.getDimSize(0))) {
      return emitOpError(
          "interaction coefficient index is outside the coefficient vector");
    }
    if (index == getMainEffectIndex() || !seenIndices.insert(index).second) {
      return emitOpError(
          "interaction coefficient indices must be distinct from each other "
          "and the main effect");
    }
  }

  llvm::SmallDenseSet<llvm::StringRef, 8> seenNames;
  for (mlir::Attribute column : getInteractionColumns()) {
    auto name = llvm::dyn_cast<mlir::StringAttr>(column);
    if (!name || name.getValue().empty()) {
      return emitOpError("interaction column names must be non-empty strings");
    }
    if (!seenNames.insert(name.getValue()).second) {
      return emitOpError("interaction column names must be unique");
    }
  }

  return mlir::success();
}
