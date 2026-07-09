"""Library-import RAG embedding (regression for Library docs never retrieved).

Library PDF import used to write only to the SQL ``Document`` table, so the
document was never embedded into the vector store that normal-chat RAG
(``chat_processor``) queries — the model retrieved nothing unless the file was
attached per-message. ``_embed_library_document`` closes that gap. These tests
pin the behavior the fix depends on: the extracted text is chunked and written
to the vector store with ``owner`` in each chunk's metadata (so the
owner-filtered search can match it), and RAG failures never break the import.
"""

import unittest
from unittest.mock import patch

from routes.document_routes import _embed_library_document


class FakeRag:
    """Minimal stand-in for the VectorRAG returned by get_rag_manager()."""

    def __init__(self, accept=True):
        self._accept = accept
        self.added = []  # list of (text, metadata)

    def _split_into_chunks(self, text, chunk_size=1000, overlap=200):
        # Deterministic two-chunk split so we can assert chunk_id sequencing
        # without depending on the real sentence-aware chunker.
        mid = max(1, len(text) // 2)
        return [text[:mid], text[mid:]]

    def add_document(self, text, metadata):
        if not self._accept:
            return False
        self.added.append((text, metadata))
        return True


class TestLibraryDocumentRagEmbedding(unittest.TestCase):
    def test_embeds_chunks_with_owner_metadata(self):
        fake = FakeRag()
        with patch("src.rag_singleton.get_rag_manager", return_value=fake):
            written = _embed_library_document(
                "doc-123", "My Report", "some prose body text here", "alice"
            )

        self.assertEqual(written, 2)
        self.assertEqual(len(fake.added), 2)
        # Every chunk must carry owner + document_id so the owner-filtered
        # search (where={"owner": owner}) actually returns it. This is the
        # exact invariant whose absence caused the bug.
        for idx, (_text, meta) in enumerate(fake.added):
            self.assertEqual(meta["owner"], "alice")
            self.assertEqual(meta["document_id"], "doc-123")
            self.assertEqual(meta["chunk_id"], idx)
            self.assertEqual(meta["type"], "library")

    def test_empty_body_writes_nothing(self):
        fake = FakeRag()
        with patch("src.rag_singleton.get_rag_manager", return_value=fake):
            self.assertEqual(_embed_library_document("d", "t", "   ", "alice"), 0)
            self.assertEqual(_embed_library_document("d", "t", None, "alice"), 0)
        self.assertEqual(fake.added, [])

    def test_rag_unavailable_is_graceful(self):
        with patch("src.rag_singleton.get_rag_manager", return_value=None):
            # No exception, no writes — import must proceed even without a store.
            self.assertEqual(
                _embed_library_document("d", "t", "body text", "alice"), 0
            )

    def test_rag_error_never_raises(self):
        def boom():
            raise RuntimeError("chroma down")

        with patch("src.rag_singleton.get_rag_manager", side_effect=boom):
            self.assertEqual(
                _embed_library_document("d", "t", "body text", "alice"), 0
            )


if __name__ == "__main__":
    unittest.main()
