"""
Generation Module for MTRAGEval.

This module provides Llama 3.1 8B Instruct based generation and grading:
- Query Rewriter: Rewrites context-dependent questions to standalone form
- Generator: Produces answers with strict I_DONT_KNOW fallback
- Relevance Grader (CRAG): Evaluates document relevance
- Hallucination Grader (Self-RAG): Checks if generation is supported by documents

Uses 4-bit quantization (NF4) via bitsandbytes for T4 GPU compatibility.
"""

import torch
from typing import Any
from dataclasses import dataclass
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFacePipeline


# --- PYDANTIC MODELS FOR JSON PARSING ---

class GradeDocuments(BaseModel):
    """Schema for document relevance grading output."""
    binary_score: str = Field(
        description="Is the document relevant? 'yes' or 'no'"
    )


class GradeHallucinations(BaseModel):
    """Schema for hallucination detection output."""
    binary_score: str = Field(
        description="Is the answer supported? 'yes' or 'no'"
    )


# --- GENERATION COMPONENTS CONTAINER ---

@dataclass
class GenerationComponents:
    """
    Container for all generation components.
    
    Encapsulates the LLM pipeline and all chain instances,
    avoiding global state and improving testability.
    
    Attributes:
        llm: The HuggingFacePipeline instance (Llama 3.1 or Qwen 2.5).
        query_rewriter: Chain for rewriting context-dependent queries.
        generator: Chain for RAG generation with I_DONT_KNOW fallback.
        retrieval_grader: Chain for CRAG document relevance grading.
        hallucination_grader: Chain for Self-RAG hallucination detection.
    """
    llm: Any
    query_rewriter: Any
    generator: Any
    retrieval_grader: Any
    hallucination_grader: Any


def get_llama_pipeline(model_id="meta-llama/Meta-Llama-3.1-8B-Instruct") -> HuggingFacePipeline:
    """
    Configure Llama 3.1 at 4-bit for T4 GPU.
    
    Sets up:
    - BitsAndBytesConfig for NF4 quantization
    - AutoTokenizer with padding fix
    - AutoModelForCausalLM with device_map="auto" for multi-GPU
    - HuggingFace text-generation pipeline
    
    Returns:
        HuggingFacePipeline wrapper for LangChain integration.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.3,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id
    )

    return HuggingFacePipeline(
        pipeline=pipe
    )

def _create_query_rewriter(llm) -> Any:
    """
    Creates the query rewriter chain.
    
    Analyzes chat history and rewrites context-dependent questions
    to standalone form.
    
    Returns:
        LangChain chain for query rewriting.
    """
    rewrite_system = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an AI assistant specialized in rewriting. 
Analyze the CHAT HISTORY and the LAST QUESTION.
Your task is to rewrite the last question so that it is understandable WITHOUT the history.

Examples:
Chat: "Who is Steve Jobs?" -> "Apple Founder."
User: "When was he born?"
Rewrite: "When was Steve Jobs born?"

If the question is already clear, return it unchanged.
Return ONLY the rewritten question, no preamble.<|eot_id|>"""

    rewrite_prompt = ChatPromptTemplate.from_messages([
        ('system', rewrite_system), # system message
        ('placeholder', '{messages}'), # {messages} will be replaced with chat history
        ('human', '{question}') # {question} will be replaced with last question
    ])

    return rewrite_prompt | llm | StrOutputParser()

def _create_generator(llm) -> Any:
    """
    Creates the RAG generator chain.
    
    Generates answers using provided context with strict I_DONT_KNOW fallback
    when information is not available.
    
    Returns:
        LangChain chain for generation.
    """
    gen_system = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a strict assistant for the mtRAG task.
Answer the user's question using EXCLUSIVELY the context provided below.

GOLDEN RULE:
If the context does not contain the necessary information to answer, YOU MUST answer with: "I_DONT_KNOW".
Do not try to guess. Do not use your internal knowledge.

CONTEXT:
{context}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Question: {question}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""

    gen_prompt = ChatPromptTemplate.from_template(gen_system)

    return gen_prompt | llm | StrOutputParser()

def _create_retrieval_grader(llm) -> Any:
    """
    Creates the document relevance grader (CRAG component).
    
    Evaluates if a document is relevant to the question.
    Returns JSON with binary_score: 'yes' or 'no'.
    
    Returns:
        LangChain chain for document grading.
    """
    parser = JsonOutputParser(
        pydantic_object=GradeDocuments
    )

    prompt = ChatPromptTemplate.from_template(
        """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a grader assessing relevance of a retrieved document to a user question.
If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
Respond EXCLUSIVELY with a JSON object with key "binary_score". For example: {"binary_score": "yes"} or {"binary_score": "no"}.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Question: {question}
Document: {document}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""
    )

    return prompt | llm | parser

def _create_hallucination_grader(llm) -> Any:
    """
    Creates the hallucination grader (Self-RAG component).
    
    Compares generation against source documents to detect hallucinations.
    Returns JSON with binary_score: 'yes' (supported) or 'no' (hallucinated).
    
    Returns:
        LangChain chain for hallucination detection.
    """
    parser = JsonOutputParser(
        pydantic_object=GradeHallucinations
    )

    prompt = ChatPromptTemplate.from_template(
        """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
Give a binary score 'yes' or 'no'. 'yes' means that the answer is fully supported by the set of facts.
'no' means that the answer contains information that is not found in the documents (hallucination).
Respond EXCLUSIVELY with a JSON object with key "binary_score". For example: {"binary_score": "yes"} or {"binary_score": "no"}.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Documents: {documents}
Answer: {generation}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""
    )

    return prompt | llm | parser


def create_generation_components() -> GenerationComponents:
    """
    Factory function to create and configure all generation components.
    
    Creates the quantized LLM pipeline and initializes all chains
    for query rewriting, generation, and grading.
    
    Returns:
        GenerationComponents: Dataclass containing initialized LLM and chains.
    
    Example:
        >>> components = create_generation_components()
        >>> result = components.generator.invoke({"context": "...", "question": "..."})
    """
    print("Creating Generation Components...")
    llm = get_llama_pipeline()
    
    components = GenerationComponents(
        llm=llm,
        query_rewriter=_create_query_rewriter(llm),
        generator=_create_generator(llm),
        retrieval_grader=_create_retrieval_grader(llm),
        hallucination_grader=_create_hallucination_grader(llm)
    )
    
    print("Generation Components Ready.")
    return components
