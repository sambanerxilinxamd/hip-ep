##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_library(DxcgcToHip STATIC
  DxcgcToHip.cpp
)

add_dependencies(DxcgcToHip
  HipDialectIncGen
  HipTypesIncGen
  HipOpsIncGen
  HipPassesIncGen
)

target_link_libraries(DxcgcToHip PUBLIC
  HipDialectIR
  MLIRPass
  MLIRFuncDialect
  MLIRTensorDialect
  MLIRIR
  MLIRSupport
)
