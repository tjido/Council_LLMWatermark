from typing import List
import logging
from string import Template
import pandas as pd
import tiktoken
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.langchain_helpers.text_splitter import TokenTextSplitter
from llama_index.node_parser import SimpleNodeParser
from council.skills import LLMSkill, PromptToMessages
from council.runners.budget import Budget
from council.contexts import SkillContext
from council.llm import OpenAILLM, LLMMessage
from council.chains import Chain
from council.agents import Agent
from council.contexts import AgentContext, ChatHistory
from council.prompt import PromptBuilder
from cachetools import cached, TTLCache
import keyboard
from document_retrieval import ChunkingTokenizer, DocRetrievalSkill, Retriever
from evaluator import Evaluator
from chatbot_controller import Controller
from watermake_ai import extract_watermark, embed_watermark, human_text
from prompt import PROMPT, SYSTEM_MESSAGE  # Make sure this is defined

# Create a disk-based cache
cache = TTLCache(maxsize=100, ttl=300)  # Configure cache for efficient data storage
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Load environment variables
import dotenv
dotenv.load_dotenv()

# Define constants
DOC_RETRIEVAL_LLM = 'gpt-3.5-turbo'
CONTROLLER_LLM = 'gpt-3.5-turbo'
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ENCODING_NAME = "cl100k_base"
MAX_CHUNK_SIZE = 256
CHUNK_OVERLAP = 20
NUM_RETRIEVED_DOCUMENTS = 50

@cached(cache)
def retrival_index(file):
    # Instantiate tokenizer for chunking
    chunking_tokenizer = ChunkingTokenizer(EMBEDDING_MODEL_NAME)

    # Instantiate tokenizer for OpenAI LLM
    text_splitter = TokenTextSplitter(
        chunk_size=MAX_CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        tokenizer=chunking_tokenizer,
        separator="\n\n",
        backup_separators=["\n", " "])

    # Instantiate node parser
    node_parser = SimpleNodeParser(text_splitter=text_splitter)

    # Specify the embedding model and node parser
    service_context = ServiceContext.from_defaults(
        embed_model=f"local:{EMBEDDING_MODEL_NAME}", node_parser=node_parser)

    # Extract the text from the pdf document
    documents = SimpleDirectoryReader(input_files=[file]).load_data()

    # Create the index by splitting text into nodes and calculating text embeddings
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)

    # Initialize index as retriever for top K most similar nodes
    index_retriever = index.as_retriever(similarity_top_k=NUM_RETRIEVED_DOCUMENTS)
    
    # Instantiate text splitter
    llm_tokenizer = tiktoken.get_encoding(ENCODING_NAME)
    # Document retrieval skill
    doc_retrieval_skill = DocRetrievalSkill(Retriever(llm_tokenizer, index_retriever))
    
    return doc_retrieval_skill

def build_context_messages(context: SkillContext) -> List[LLMMessage]:
    """Build context messages for LLMSkill"""
    context_message_prompt = PromptToMessages(prompt_builder=PromptBuilder(PROMPT))
    return context_message_prompt.to_user_message(context)

def load_agent(company_name, llm) -> Agent:
    # LLM Skill
    llm_skill_model = OpenAILLM.from_env(model=llm)
    llm_skill = LLMSkill(
        llm=llm_skill_model,
        system_prompt=Template(SYSTEM_MESSAGE).substitute(company=company_name),  # Make sure SYSTEM_MESSAGE is defined
        context_messages=build_context_messages,
    )
    # Initialize Chains
    doc_retrieval_chain = Chain(
        name="doc_retrieval_chain",
        description="Useful for answering questions about current events.",  # Reference Alistair's article
        runners=[doc_retrieval_skill, llm_skill],
    )
    # Initializing controller LLM
    controller_llm = OpenAILLM.from_env(model=CONTROLLER_LLM)
    controller = Controller(controller_llm, response_threshold=5)

    evaluator = Evaluator()
    
    agent = Agent(controller, [doc_retrieval_chain], evaluator)
    
    return agent

if __name__ == '__main__':
    PDF_FILE_NAME = 'msft-10K-2022.pdf'
    print('Extracting chunks from PDF File...')
    doc_retrieval_skill = retrival_index(PDF_FILE_NAME)
    print('Finish Extracting chunks!!!')
    company_name = input('Enter company name: ')
    print("You entered:", company_name)
    print('Running LLM Agents on chunks...')
    agent = load_agent(company_name=company_name, llm=DOC_RETRIEVAL_LLM)
    # Initialize a flag to keep track of whether it's the first iteration or not
    query = input('Query your document: ')
    chat_history = ChatHistory()
    chat_history.add_user_message(query)
    run_context = AgentContext(chat_history)
    result = agent.execute(run_context, Budget(600))
    if result:
        print('AI Generated Text: ')
        response = result.best_message.message
        watermarked_ai_text, embedded_watermark = embed_watermark(response, watermark_key="HIDDEN_WATERMARK")
        print('-------------------------------------------------------------')
        print(f"""Hidden Unicode WaterMark AI Text:\n   {watermarked_ai_text}""")
        print('-------------------------------------------------------------')
        extracted_watermark = extract_watermark(watermarked_ai_text, embedded_watermark)
        print(f"""AI Output:\n   {extracted_watermark}""")
        print('-------------------------------------------------------------')
        human_generated_text = human_text(response)
        print(f"""Human Output:\n {human_generated_text}""")
