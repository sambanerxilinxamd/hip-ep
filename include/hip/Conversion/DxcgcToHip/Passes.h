/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_CONVERSION_DXCGCTOHIP_PASSES_H
#define HIP_CONVERSION_DXCGCTOHIP_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir::hip {

/// Creates a pass that converts parsed cgc.* and cgc_op.* operations to HIP.
std::unique_ptr<Pass> createConvertDxcgcToHipPass();

} // namespace mlir::hip

#endif // HIP_CONVERSION_DXCGCTOHIP_PASSES_H
