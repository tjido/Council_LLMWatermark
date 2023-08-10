from typing import List
from transformers import AutoTokenizer
from tiktoken import Encoding
from llama_index.indices.vector_store import VectorIndexRetriever
from llama_index.schema import NodeWithScore

from council.skills import SkillBase
from council.runners.budget import Budget
from council.contexts import SkillContext, ChatMessage
from council.contexts import  ChatMessage

CONTEXT_TOKEN_LIMIT = 3000

## Document Retrieval
class ChunkingTokenizer:
    """Tokenizer for chunking document data for creation of embeddings"""

    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def __call__(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)


# Define utility class for document retrieval with LlamaIndex
class Retriever:
    def __init__(self, llm_tokenizer: Encoding, retriever: VectorIndexRetriever):
        """Class to retrieve text chunks from Llama Index and create context for LLM"""
        self.llm_tokenizer = llm_tokenizer
        self.retriever = retriever

    def retrieve_docs(self, query) -> str:
        """End-to-end functiont to retrieve most similar nodes and build the context"""
        nodes = self.retriever.retrieve(query)
        docs = self._extract_text(nodes)
        context = self._build_context(docs)

        return context

    @staticmethod
    def _extract_text(nodes: List[NodeWithScore]) -> List[str]:
        """Function to extract the text from the retrieved nodes"""
        return [node.node.text for node in nodes]

    def _build_context(self, docs: List[str]) -> str:
        """Function to build context for LLM by separating text chunks into paragraphs"""
        context = ""
        num_tokens = 0
        for doc in docs:
            doc += "\n\n"
            num_tokens += len(self.llm_tokenizer.encode(doc))
            if num_tokens <= CONTEXT_TOKEN_LIMIT:
                context += doc
            else:
                break

        return context


class DocRetrievalSkill(SkillBase):
    """Skill to retrieve documents and build context"""

    def __init__(self, retriever: Retriever):
        super().__init__(name="document_retrieval")
        self.retriever = retriever

    def execute(self, context: SkillContext, budget: Budget) -> ChatMessage:
        query = context.current.messages[-1].message
        context = self.retriever.retrieve_docs(query)

        return self.build_success_message(context)
    
