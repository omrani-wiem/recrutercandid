import unittest

from mcp.types import CallToolResult, TextContent

from graph import _mcp_result_payload


class McpResultPayloadTests(unittest.TestCase):
    def test_returns_structured_content(self):
        result = CallToolResult(
            content=[],
            structuredContent={"text": "CV transcrit", "confidence": 0.85},
        )

        self.assertEqual(
            _mcp_result_payload(result),
            {"text": "CV transcrit", "confidence": 0.85},
        )

    def test_parses_json_text_when_structured_content_is_missing(self):
        result = CallToolResult(
            content=[TextContent(type="text", text='{"text": "CV transcrit", "confidence": 0.85}')],
        )

        self.assertEqual(
            _mcp_result_payload(result),
            {"text": "CV transcrit", "confidence": 0.85},
        )

    def test_reports_mcp_tool_errors(self):
        result = CallToolResult(
            content=[TextContent(type="text", text="OCR indisponible")],
            isError=True,
        )

        with self.assertRaisesRegex(RuntimeError, "OCR indisponible"):
            _mcp_result_payload(result)


if __name__ == "__main__":
    unittest.main()