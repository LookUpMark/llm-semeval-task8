"""Generation Module: Llama 3.1 with 4-bit quantization for RAG + grading chains."""

import torch
from typing import Any, List, Tuple
from dataclasses import dataclass
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFacePipeline


# Pydantic schemas for JSON parsing
class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Is the document relevant? 'yes' or 'no'")

class GradeHallucinations(BaseModel):
    binary_score: str = Field(description="Is the answer supported? 'yes' or 'no'")


@dataclass
class GenerationComponents:
    """Container for LLM pipeline and all chain instances."""
    llm: Any
    query_rewriter: Any
    generator: Any
    retrieval_grader: Any
    hallucination_grader: Any


def get_llama_pipeline(model_id="meta-llama/Llama-3.1-8B-Instruct") -> HuggingFacePipeline:
    """Configure Llama 3.1 with 4-bit NF4 quantization for T4 GPU."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
    )
    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer,
        max_new_tokens=1024, do_sample=True, temperature=0.01,
        repetition_penalty=1.1, pad_token_id=tokenizer.eos_token_id, return_full_text=False
    )
    return HuggingFacePipeline(pipeline=pipe)


def _format_chat_history(messages: List[Tuple[str, str]]) -> str:
    return "\n".join(f"{'User' if r.lower() in ['user','human'] else 'Assistant'}: {c}" for r, c in messages)


def _create_query_rewriter(llm) -> Any:
    """Creates the query rewriter chain."""
    template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an AI assistant specialized in rewriting to resolve coreferences.
Analyze the CHAT HISTORY and the LAST QUESTION.
Your task is to rewrite the LAST QUESTION so that it is understandable WITHOUT the history.

### EXAMPLES
Chat History:
User: Who is the CEO of Apple?
Assistant: Tim Cook.
Last Question: How old is he?
Rewrite: How old is Tim Cook?

Chat History:
User: Tell me about Paris.
Assistant: It's the capital of France.
Last Question: What is the population of New York?
Rewrite: What is the population of New York?

Chat History:
User: Using the given context, explain the main idea.
Assistant: The context discusses quantum mechanics.
Last Question: Summarize it.
Rewrite: Summarize the main idea of quantum mechanics.
### END EXAMPLES

If the question is already clear, return it unchanged.
Return ONLY the rewritten question, no preamble.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Chat History:
{messages}

Last Question: {question}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""
    
    prompt = PromptTemplate(template=template, input_variables=["messages", "question"])
    return ({"messages": lambda x: _format_chat_history(x["messages"]), "question": lambda x: x["question"]} 
            | prompt | llm | StrOutputParser())


def _create_generator(llm) -> Any:
    """Creates the RAG generator chain with strict I_DONT_KNOW fallback."""
    template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a strict assistant for a RAG task.
Answer the user's question using EXCLUSIVELY the context provided below.

### EXAMPLES
Context: The sky is blue. The grass is green.
Question: What color is the sky?
Answer: The sky is blue.

Context: The capital of Italy is Rome. The Colosseum is in Rome.
Question: Who is the President of France?
Answer: I_DONT_KNOW

Context: Photosynthesis is the process used by plants to convert light into energy.
Question: How do plants get energy?
Answer: Plants get energy through a process called photosynthesis, which converts light into energy.
### END EXAMPLES

GOLDEN RULE:
If the context does not contain the necessary information to answer, YOU MUST answer with: "I_DONT_KNOW".
Do not try to guess. Do not use your internal knowledge.

CONTEXT:
{context}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Question: {question}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""
    
    return PromptTemplate(template=template, input_variables=["context", "question"]) | llm | StrOutputParser()


def _create_retrieval_grader(llm) -> Any:
    """Creates the document relevance grader (CRAG component)."""
    template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a grader assessing relevance of a retrieved document to a user question.
If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
It does not need to be a stringent test. The goal is to filter out erroneous retrievals.

### EXAMPLES
Question: What is the capital of France?
Document: Paris is the capital city of France.
Output: {{"binary_score": "yes"}}

Question: How do I make pasta?
Document: The history of pasta starts in Italy.
Output: {{"binary_score": "yes"}}

Question: Who won the 2018 World Cup?
Document: The 2018 World Cup was hosted in Russia.
Output: {{"binary_score": "no"}}

Question: What is the weather in London?
Document: Apple Inc. released a new iPhone.
Output: {{"binary_score": "no"}}
### END EXAMPLES

Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
Respond EXCLUSIVELY with a JSON object with key "binary_score".<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Question: {question}
Document: {document}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""
    
    return (PromptTemplate(template=template, input_variables=["question", "document"]) 
            | llm | JsonOutputParser(pydantic_object=GradeDocuments))


def _create_hallucination_grader(llm) -> Any:
    """Creates the hallucination grader (Self-RAG component)."""
    template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
Give a binary score 'yes' or 'no'. 'yes' means that the answer is fully supported by the set of facts.
'no' means that the answer contains information that is not found in the documents (hallucination).

### EXAMPLES
Documents: Apple was founded in 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne.
Answer: Apple was founded in 1976.
Output: {{"binary_score": "yes"}}

Documents: Apple was founded in 1976.
Answer: Apple was founded in 1976 by Tim Cook.
Output: {{"binary_score": "no"}}

Documents: The sky is blue due to Rayleigh scattering.
Answer: The sky is blue.
Output: {{"binary_score": "yes"}}
### END EXAMPLES

Respond EXCLUSIVELY with a JSON object with key "binary_score".<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Documents: {documents}
Answer: {generation}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""
    
    return (PromptTemplate(template=template, input_variables=["documents", "generation"]) 
            | llm | JsonOutputParser(pydantic_object=GradeHallucinations))


def create_generation_components(model_id: str = "meta-llama/Llama-3.1-8B-Instruct") -> GenerationComponents:
    """Factory: creates quantized LLM and all chains (rewriter, generator, graders)."""
    print(f"Creating Generation Components with model: {model_id}...")
    llm = get_llama_pipeline(model_id=model_id)
    
    components = GenerationComponents(
        llm=llm,
        query_rewriter=_create_query_rewriter(llm),
        generator=_create_generator(llm),
        retrieval_grader=_create_retrieval_grader(llm),
        hallucination_grader=_create_hallucination_grader(llm)
    )
    print("Generation Components Ready.")
    return components
