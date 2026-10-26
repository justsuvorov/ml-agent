import json

from hamcrest import assert_that, contains_string, equal_to, raises

from ml_agent.dialog import OpenDialog, OpenDialogError
from ml_agent.reply import Reply
from ml_agent.tools import Tools
from tests.fakes import Echo, phrase


def test_reads_the_answer_wrapped_in_a_markdown_fence() -> None:
    answer = phrase()
    assert_that(
        Reply(
            f"```json\n{json.dumps({"answer": answer}, ensure_ascii=False)}\n```",
            Tools({}),
        ).dialog(OpenDialog(phrase(), phrase())).answer(),
        equal_to(answer),
        "Reply did not read the answer inside the markdown fence",
    )


def test_keeps_the_dialog_open_after_a_tool_call() -> None:
    assert_that(
        Reply(json.dumps({"tool": "echo", "arguments": {}}), Tools({"echo": Echo(phrase())}))
        .dialog(OpenDialog(phrase(), phrase()))
        .answer,
        raises(OpenDialogError),
        "Reply closed the dialog after a tool call",
    )


def test_records_the_reply_of_the_model_in_the_dialog() -> None:
    thought = phrase()
    assert_that(
        Reply(
            json.dumps({"thought": thought, "tool": "echo", "arguments": {}}, ensure_ascii=False),
            Tools({"echo": Echo(phrase())}),
        )
        .dialog(OpenDialog(phrase(), phrase()))
        .text(),
        contains_string(thought),
        "Reply did not record the reply of the model in the dialog",
    )
