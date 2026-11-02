import unittest

from langchain_core.documents import Document

from backend.chunk_documents import MODEL_MAX_TOKENS, split_resume_documents
from tests.tokenizer_fixture import make_tokenizer


class ChunkDocumentTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = make_tokenizer()

    def split(self, documents, **kwargs):
        return split_resume_documents(documents, tokenizer=self.tokenizer, **kwargs)

    def test_long_text_is_bounded_traceable_repeatable_and_keeps_the_end(self):
        text = "First sentence.\n" + (
            "The team built an API for the users.\n" * 100
        ) + "Last sentence."
        metadata = {"item_type": "project", "title": "API", "record_id": 3}
        document = Document(id="career_item:3", page_content=text, metadata=metadata)
        chunks = self.split([document])
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks, self.split([document]))
        self.assertEqual(len({chunk.id for chunk in chunks}), len(chunks))
        for index, chunk in enumerate(chunks):
            count = len(self.tokenizer.encode(chunk.page_content))
            self.assertLessEqual(count, MODEL_MAX_TOKENS)
            self.assertEqual(chunk.metadata["token_count"], count)
            self.assertEqual(chunk.metadata["parent_id"], document.id)
            self.assertEqual(chunk.metadata["chunk_index"], index)
            self.assertEqual(chunk.metadata["chunk_count"], len(chunks))
            self.assertEqual(chunk.metadata["record_id"], 3)
            self.assertIn(chunk.page_content, text)
        self.assertIn("First sentence.", chunks[0].page_content)
        self.assertIn("Last sentence.", chunks[-1].page_content)
        self.assertEqual(document.metadata, metadata)
        self.assertEqual(document.page_content, text)

    def test_short_documents_keep_their_sentences_and_separate_identities(self):
        documents = [
            Document(id="course:1", page_content="Course: Python", metadata={"name": "Python"}),
            Document(id="skill:1", page_content="Skill: C++ / Node.js — SQL",
                     metadata={"name": "C++ / Node.js — SQL"}),
        ]
        chunks = self.split(documents)
        self.assertEqual(len(chunks), 2)
        self.assertEqual([d.page_content for d in chunks], [d.page_content for d in documents])
        self.assertEqual([d.id for d in chunks], ["course:1:chunk:0", "skill:1:chunk:0"])

    def test_token_budget_overlap_and_source_identity_are_validated(self):
        document = Document(id="skill:1", page_content="Python")
        for settings in ({"chunk_size": 256}, {"chunk_overlap": 220},
                         {"chunk_overlap": -1}, {"chunk_size": 0}):
            with self.assertRaises(ValueError):
                self.split([document], **settings)
        with self.assertRaises(ValueError):
            self.split([Document(page_content="No source ID")])
        with self.assertRaises(ValueError):
            self.split([document, document])
        self.assertEqual(self.split([]), [])


if __name__ == "__main__":
    unittest.main()
