import unittest

from dxcgc_to_hip import ConversionError, convert


class DxcgcToHipTest(unittest.TestCase):
    def test_elementwise_graph(self):
        source = """cgc.module {
  func.func @main(%x : !cgc.tensor<1x32x!cgc.float16>, %bias : !cgc.tensor<32x!cgc.float16>) -> !cgc.tensor<1x32x!cgc.float16> {
    %y = cgc_op.add(%x, %bias) : (!cgc.tensor<1x32x!cgc.float16>, !cgc.tensor<32x!cgc.float16>) -> !cgc.tensor<1x32x!cgc.float16>
    cgc.return %y : !cgc.tensor<1x32x!cgc.float16>
  }
}
"""
        result = convert(source)
        self.assertIn("func.func @main(%ctx: !hip.context", result)
        self.assertIn("tensor.empty() : tensor<1x32xf16>", result)
        self.assertIn("hip.add(%ctx) ins(%x, %bias : tensor<1x32xf16>, tensor<32xf16>)", result)
        self.assertNotIn("cgc_op", result)

    def test_unsupported_operation_is_rejected(self):
        source = """cgc.module {
  func.func @main(%x : !cgc.tensor<1x32x!cgc.float16>) -> !cgc.tensor<1x32x!cgc.float16> {
    %y = cgc_op.custom(%x) : (!cgc.tensor<1x32x!cgc.float16>) -> !cgc.tensor<1x32x!cgc.float16>
    cgc.return %y : !cgc.tensor<1x32x!cgc.float16>
  }
}
"""
        with self.assertRaisesRegex(ConversionError, "unsupported CGC operation"):
            convert(source)


if __name__ == "__main__":
    unittest.main()
