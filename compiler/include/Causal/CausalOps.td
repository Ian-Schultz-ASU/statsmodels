include "Causal/CausalDialect.td"

class Causal_Op<string mnemonic, list<Trait> traits = []>
    : Op<Causal_Dialect, mnemonic, traits>;

def Causal_TreatmentEffectOp : Causal_Op<"treatment_effect"> {
  let summary = "Compute an OLS treatment contrast from interaction features";
  let description = [{
    For each row, computes (treated_value - control_value) multiplied by the
    fitted treatment main-effect coefficient plus the sum of each interaction
    feature times its fitted interaction coefficient.
  }];

  let arguments = (ins
    AnyRankedTensor:$data,
    AnyRankedTensor:$coefficients,
    StrAttr:$treatment_name,
    F64Attr:$control_value,
    F64Attr:$treated_value,
    I64Attr:$main_effect_index,
    ArrayAttr:$interaction_columns,
    DenseI64ArrayAttr:$interaction_coefficients
  );
  let results = (outs AnyRankedTensor:$effects);

  let hasVerifier = 1;
}
