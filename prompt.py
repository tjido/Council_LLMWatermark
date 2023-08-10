SYSTEM_MESSAGE = "You are a an AI assistant whose job is to answer user questions about $company with the provided context."
PROMPT = """Use the following pieces of context to answer the query.
If the answer is not provided in the context, do not make up an answer. Instead, respond that you do not know.

CONTEXT:
{{chain_history.last_message}}
END CONTEXT.

QUERY:
{{chat_history.user.last_message}}
END QUERY.

YOUR ANSWER:
"""


