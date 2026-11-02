import unittest
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from pydantic import Field
from deepagents.backends.utils import file_data_to_string

from agent import create_resume_agent, prepare_resume_request
from prompts import (
    BASE_RESUME_PATH, JOB_PATH, DEFAULT_GUIDELINES_PATH, DEFAULT_TEMPLATE_PATH,
    build_resume_prompts,
)


class RecordingChatModel(FakeMessagesListChatModel):
    seen: list = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(messages)
        return super()._generate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )


class ResumeWorkflowTests(unittest.TestCase):
    def test_missing_guidelines_and_bad_modes_fail_before_agent_creation(self):
        with patch("agent.create_deep_agent") as create:
            for guidelines in ("", "   ", 42):
                with self.assertRaises(ValueError):
                    create_resume_agent(model="unused:test", latex_guidelines=guidelines)
            with self.assertRaises(ValueError):
                create_resume_agent(
                    model="unused:test", latex_guidelines="Preserve my template.",
                    output_mode="pdf",
                )
            create.assert_not_called()

    def test_latex_braces_and_rules_reach_orchestrator_and_reviewer_unchanged(self):
        guidelines = r"Keep \setlength{\parindent}{0pt} and use the template's section order."
        template = DEFAULT_TEMPLATE_PATH.read_text(encoding="utf-8").replace(
            r"\begin{document}", r"\newcommand{\entry}[2]{#1 & #2}" + "\n" + r"\begin{document}"
        )
        instructions, agents = build_resume_prompts(
            guidelines, latex_template=template, output_mode="latex_only"
        )
        self.assertIn(guidelines, instructions)
        self.assertIn(template, instructions)
        reviewer = next(a for a in agents if a["name"] == "latex-resume-reviewer")
        self.assertIn(guidelines, reviewer["system_prompt"])
        self.assertIn(template, reviewer["system_prompt"])
        self.assertEqual(len({a["name"] for a in agents}), 3)
        self.assertTrue(all(a["tools"] == [] for a in agents))
        # Each call returns a fresh configuration, so caller edits cannot leak.
        agents[0]["tools"].append("changed")
        _, another = build_resume_prompts(guidelines)
        self.assertEqual(another[0]["tools"], [])

    def test_input_sources_are_required_and_preserved_in_real_agent_state(self):
        for job, resume in (("", "Resume"), ("Job", ""), (None, "Resume")):
            with self.assertRaises(ValueError):
                prepare_resume_request(job, resume)
        model = RecordingChatModel(responses=[
            AIMessage(content="Please provide employment dates before I finish the resume.")
        ])
        guidelines = DEFAULT_GUIDELINES_PATH.read_text(encoding="utf-8")
        compiled = create_resume_agent(model=model)
        job = "Required: Python & SQL.\nPreferred: AWS."
        source = "Built Python services for the team's users — with SQL."
        result = compiled.invoke(prepare_resume_request(job, source))
        self.assertEqual(file_data_to_string(result["files"][JOB_PATH]), job)
        self.assertEqual(file_data_to_string(result["files"][BASE_RESUME_PATH]), source)
        self.assertEqual(result["messages"][-1].content,
                         "Please provide employment dates before I finish the resume.")
        self.assertTrue(any(
            guidelines in str(message.content)
            for messages in model.seen for message in messages
        ))


if __name__ == "__main__":
    unittest.main()
