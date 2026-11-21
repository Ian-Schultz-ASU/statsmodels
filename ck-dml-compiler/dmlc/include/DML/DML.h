#ifndef DML_DML_H
#define DML_DML_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <memory>

#include "DML/DMLOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "DML/DMLOps.h.inc"

namespace mlir {
class Pass;
}
namespace dml {
void registerDMLPasses();
} // namespace dml

#endif // DML_DML_H
