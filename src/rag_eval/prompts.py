"""Prompt texts. Each constant has a one-line purpose; bump PROMPT_VERSION on any wording change."""

PROMPT_VERSION = "1"

# Purpose: constrain the generator to the retrieved context and give it an exact refusal string.
SYSTEM_PROMPT = (
    "Answer the question using only the numbered context passages. "
    "Be brief: one or two sentences. "
    "If the context does not contain the answer, reply exactly: I don't know."
)

# Purpose: lay out context and question for the user turn of the chat template.
USER_TEMPLATE = "Context:\n{context}\n\nQuestion: {question}"

# Purpose: the exact refusal string, also used to detect abstentions during faithfulness scoring.
ABSTAIN = "I don't know."

# Purpose: header for each context passage so the model and the reader can see page provenance.
PASSAGE_HEADER = "[{n}] (page {page})"
