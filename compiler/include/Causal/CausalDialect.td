include "mlir/IR/OpBase.td"

def Causal_Dialect : Dialect {
  let name = "ci";
  let cppNamespace = "::causal";
  let summary = "Dialect for causal inference workflows";
}
