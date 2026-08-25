/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Conversion/DxcgcToHip/Passes.h"
#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include <memory>

namespace mlir::hip {

#define GEN_PASS_DEF_CONVERTDXCGCTOHIPPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

static StringRef getDxcgcOpName(Operation *op) {
  StringRef name = op->getName().getStringRef();
  for (StringRef prefix : {"cgc.", "cgc_op.", "dxgml_op."})
    if (name.starts_with(prefix))
      return name.drop_front(prefix.size());
  return {};
}

static bool isUnaryOp(StringRef name) {
  return llvm::StringSwitch<bool>(name)
      .Cases("not", "abs", "neg", "cos", "sin", "ceil", "exp", "log",
             "sign", true)
      .Default(false);
}

static StringRef getHipOpName(StringRef name) {
  return llvm::StringSwitch<StringRef>(name)
      .Cases("add", "mul", "sub", "div", "min", "max", "equal", "less",
             name)
      .Cases("and", "or", "not", "abs", "neg", "cos", "sin", "ceil",
             name)
      .Cases("exp", "log", "sign", "matmul", "reshape", "transpose", "cast",
             name)
      .Default({});
}
struct ConvertDxcgcToHipPass
    : public impl::ConvertDxcgcToHipPassBase<ConvertDxcgcToHipPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    bool hadError = false;

    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      if (function.empty())
        continue;
      if (function.getNumArguments() == 0 ||
          !dyn_cast<ContextType>(function.getArgument(0).getType())) {
        function.emitError()
            << "convert-dxcgc-to-hip requires !hip.context as argument zero; "
               "run --hip-add-context-arg first";
        hadError = true;
        continue;
      }

      SmallVector<Operation *> sourceOps;
      function.walk([&](Operation *op) {
        if (!getDxcgcOpName(op).empty())
          sourceOps.push_back(op);
      });

      for (Operation *op : sourceOps) {
        StringRef sourceName = getDxcgcOpName(op);
        StringRef targetName = getHipOpName(sourceName);
        const unsigned expectedOperands = isUnaryOp(sourceName) ? 1 : 2;
        if (targetName.empty()) {
          op->emitError() << "unsupported DxCGC operation '" << sourceName << "'";
          hadError = true;
          continue;
        }
        if (op->getNumResults() != 1 ||
            op->getNumOperands() != expectedOperands || !op->getAttrs().empty()) {
          op->emitError() << "DxCGC operation '" << sourceName << "' must have "
                          << expectedOperands
                          << " operands, one result, and no attributes";
          hadError = true;
          continue;
        }

        auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
        if (!resultType || llvm::any_of(resultType.getShape(), ShapedType::isDynamic)) {
          op->emitError() << "DxCGC operation '" << sourceName
                          << "' must have a statically shaped ranked tensor result";
          hadError = true;
          continue;
        }

        OpBuilder builder(op);
        Value output = tensor::EmptyOp::create(
            builder, op->getLoc(), resultType.getShape(), resultType.getElementType());
        std::string hipOpName = ("hip." + targetName).str();
        OperationState state(op->getLoc(), hipOpName);
        SmallVector<Value> operands;
        operands.push_back(function.getArgument(0));
        operands.append(op->getOperands().begin(), op->getOperands().end());
        operands.push_back(output);
        state.addOperands(operands);
        state.addTypes(resultType);
        Operation *replacement = builder.create(state);
        op->getResult(0).replaceAllUsesWith(replacement->getResult(0));
        op->erase();
      }
    }

    if (hadError)
      signalPassFailure();
  }
};

} // namespace


} // namespace mlir::hip






