#ifndef CAUSAL_PASSES_H
#define CAUSAL_PASSES_H

#include <memory>

namespace mlir {
class Pass;
}

namespace causal {

std::unique_ptr<mlir::Pass> createLowerTreatmentEffectPass();
void registerLowerTreatmentEffectPass();

} // namespace causal

#endif // CAUSAL_PASSES_H
