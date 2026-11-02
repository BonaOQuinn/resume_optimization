import unittest
from unittest.mock import Mock, patch

from deepagents.backends.protocol import FileUploadResponse
from deepagents.middleware.filesystem import FilesystemState
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

import agent as resume_agent


def build_retrieval_graph(query, repetitions=1):
    """Exercise the actual state backend and its read-after-write behavior."""
    def retrieve(state):
        outputs = [resume_agent.search_resume.invoke({"query": query})
                   for _ in range(repetitions)]
        return {"messages": [AIMessage(content="\n".join(outputs))]}

    def read_files(state):
        paths = [line for line in state["messages"][-1].content.splitlines()
                 if line.startswith("/retrieved/resume/")]
        responses = resume_agent.backend.download_files(paths)
        contents = []
        for response in responses:
            if response.error:
                raise AssertionError(f"Saved evidence could not be read: {response.path}")
            contents.append(response.content.decode("utf-8"))
        return {"messages": [AIMessage(content="\n\n".join(contents))]}

    graph = StateGraph(FilesystemState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("read_files", read_files)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "read_files")
    graph.add_edge("read_files", END)
    return graph.compile()


class ResumeRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.store = Mock()
        self.store.similarity_search.return_value = [
            Document(
                id="career_item:2:chunk:0",
                page_content="Built Python services for the team's users — with SQL.",
                metadata={
                    "source_file": "experience.md", "item_type": "experience",
                    "title": "Engineer", "organization": "Example",
                    "parent_id": "career_item:2", "record_id": 2, "chunk_index": 0,
                },
            ),
            Document(
                id="skill:1:chunk:0", page_content="Skill: Python",
                metadata={"source_file": "tech_skills.md", "item_type": "skill",
                          "name": "Python", "parent_id": "skill:1"},
            ),
        ]

    def test_search_saves_traceable_evidence_readable_in_the_same_graph(self):
        with patch.object(resume_agent, "get_resume_vector_store", return_value=self.store):
            result = build_retrieval_graph("Python and SQL", repetitions=2).invoke(
                {"messages": [], "files": {}}
            )
        self.assertEqual(len(result["files"]), 4)  # No overwrite on consecutive calls.
        self.assertEqual(self.store.similarity_search.call_count, 2)
        self.store.similarity_search.assert_called_with("Python and SQL", k=4)
        evidence = result["messages"][-1].content
        self.assertIn("Source file: experience.md", evidence)
        self.assertIn('"parent_id": "career_item:2"', evidence)
        self.assertIn("Chunk ID: career_item:2:chunk:0", evidence)
        self.assertIn("Built Python services for the team's users — with SQL.", evidence)
        self.assertIn("# Resume evidence: Python", evidence)

    def test_empty_collection_does_not_try_to_write_files(self):
        self.store.similarity_search.return_value = []
        with patch.object(resume_agent, "get_resume_vector_store", return_value=self.store), \
             patch.object(resume_agent.backend, "upload_files") as upload:
            message = resume_agent.search_resume.invoke({"query": "Python"})
        self.assertIn("Run python main.py", message)
        upload.assert_not_called()

    def test_blank_query_does_not_initialize_the_model(self):
        with patch.object(resume_agent, "get_resume_vector_store") as get_store:
            message = resume_agent.search_resume.invoke({"query": "   "})
        get_store.assert_not_called()
        self.assertIn("Enter a job requirement", message)

    def test_failed_uploads_are_not_reported_as_saved(self):
        self.store.similarity_search.return_value = self.store.similarity_search.return_value[:1]
        def fail_upload(uploads):
            return [FileUploadResponse(path=path, error="permission_denied")
                    for path, _ in uploads]
        with patch.object(resume_agent, "get_resume_vector_store", return_value=self.store), \
             patch.object(resume_agent.backend, "upload_files", side_effect=fail_upload):
            message = resume_agent.search_resume.invoke({"query": "Python"})
        self.assertIn("Saved 0 resume evidence chunks", message)
        self.assertIn("Failed to save:", message)
        self.assertIn("permission_denied", message)


if __name__ == "__main__":
    unittest.main()
