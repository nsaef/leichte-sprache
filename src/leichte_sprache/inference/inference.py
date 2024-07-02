from argparse import ArgumentParser
import os

from vllm import LLM, SamplingParams

from leichte_sprache.utils.db_utils import get_pinecone_index
from leichte_sprache.utils.model_utils import generate_vllm, create_embeddings


def run_inference_vllm(
    args: ArgumentParser,
    prompts: list[str],
    sampling_params: SamplingParams = None,
    max_model_len: int = 2048,
) -> list[str]:
    """Run inference using vLLM. Generates five examples per prompt.

    :param args: arguments including modelname
    :param prompts: list of prompts in the chat format
    :return: list of generated texts (five per prompt)
    """
    llm = LLM(
        model=args.model_name,
        max_model_len=max_model_len,
    )
    if not sampling_params:
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            skip_special_tokens=True,
            top_k=50,
            n=5,
        )
    results = generate_vllm(
        messages=prompts,
        llm=llm,
        sampling_params=sampling_params,
    )
    texts = [output.text for res in results for output in res.outputs]
    return texts


def run_rag():
    index = get_pinecone_index()
    input_text = "das hier ist ein beispieltext #todo: einen richtigen holen"

    input_embeddings = create_embeddings(input_texts=[input_text])
    response = index.query(
        vector=input_embeddings,
        # filter={
        #     "genre": {"$eq": "documentary"},
        #     "year": 2019
        # },
        top_k=1,
        include_metadata=True,
    )
    # integrate answer in prompt
    # generate
