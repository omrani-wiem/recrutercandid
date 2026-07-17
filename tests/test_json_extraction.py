import unittest

from prompts.system_prompts import extract_json_payload


class ExtractJsonPayloadTests(unittest.TestCase):
    def test_extracts_json_from_markdown_fence(self):
        text = '```json\n{"title":"Senior Data Engineer","required_skills":["Python"]}\n```'
        self.assertEqual(
            extract_json_payload(text),
            '{"title":"Senior Data Engineer","required_skills":["Python"]}',
        )

    def test_extracts_json_from_plain_text_with_extra_words(self):
        text = 'Voici le résultat : {"title":"Data Engineer"} merci.'
        self.assertEqual(extract_json_payload(text), '{"title":"Data Engineer"}')


if __name__ == '__main__':
    unittest.main()
