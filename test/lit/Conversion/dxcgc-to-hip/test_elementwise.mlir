// RUN: hip-mlir-opt %s -allow-unregistered-dialect --hip-add-context-arg --convert-dxcgc-to-hip | FileCheck %s

module {
  func.func @main(%lhs: tensor<1x32xf16>, %rhs: tensor<1x32xf16>) -> tensor<1x32xf16> {
    %result = "cgc.add"(%lhs, %rhs) : (tensor<1x32xf16>, tensor<1x32xf16>) -> tensor<1x32xf16>
    return %result : tensor<1x32xf16>
  }
}

// CHECK-LABEL: func.func @main(%[[CTX:.*]]: !hip.context, %[[LHS:.*]]: tensor<1x32xf16>, %[[RHS:.*]]: tensor<1x32xf16>)
// CHECK: %[[INIT:.*]] = tensor.empty() : tensor<1x32xf16>
// CHECK: %[[RESULT:.*]] = hip.add(%[[CTX]]) ins(%[[LHS]], %[[RHS]] : tensor<1x32xf16>, tensor<1x32xf16>) outs(%[[INIT]] : tensor<1x32xf16>)
// CHECK: return %[[RESULT]] : tensor<1x32xf16>

