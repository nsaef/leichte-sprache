from argparse import ArgumentParser
import os

from vllm import LLM, SamplingParams

from leichte_sprache.constants import DLF_DICT, MDR_DICT, SRC_COLUMN
from leichte_sprache.utils.db_utils import get_pinecone_index
from leichte_sprache.utils.model_utils import (
    generate_vllm,
    create_embeddings,
    create_prompts_sg_to_ls,
)


def run_inference_vllm(
    model_name: str,
    prompts: list[str],
    sampling_params: SamplingParams = None,
    max_model_len: int = 2048,
) -> list[str]:
    """Run inference using vLLM. Generates five examples per prompt.

    :param args: arguments including modelname #todo update
    :param prompts: list of prompts in the chat format
    :return: list of generated texts (five per prompt)
    """
    llm = LLM(
        model=model_name,
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
    input_texts = [
        "Sie kam hinein, aber nicht mehr heraus: Die Feuerwehr musste in Mittelfranken eine Jugendliche auf einem Spielplatz aus einer Kleinkindschaukel schneiden. Die Jugendliche hatte sich nach Angaben der Polizei am Montagabend derart in der Schaukel verkeilt, dass sie sich nicht mehr befreien konnte. Das Mädchen blieb der Polizei zufolge unverletzt - den Schaden an der Schaukel werde sie aber wohl bezahlen müssen, hieß es."
    ]
    input_embeddings = create_embeddings(input_texts=input_texts)
    response = index.query(
        vector=input_embeddings[0],
        filter={
            SRC_COLUMN: {"$in": [MDR_DICT, DLF_DICT]},
        },
        top_k=1,
        include_metadata=True,
    )
    match = response.matches[0]

    # integrate answer in prompt
    prompt_suffix = "\nText in Leichter Sprache:"
    rag_prompt_addition = f" Bezieh dabei die foglenden Informationen und Definitionen ein, sofern sie sinnvoll den Text ergänzen: {match['metadata']['text']}"
    prompts = create_prompts_sg_to_ls(texts=input_texts)
    for prompt in prompts:
        prompt_text = prompt[1].get("content").replace(prompt_suffix, "")
        prompt_text += rag_prompt_addition + prompt_suffix
        prompt[1]["content"] = prompt_text

    # generate
    generated_texts = run_inference_vllm(
        model_name="/home/nasrin/data/trainings/discolm-ls-03/awq",
        prompts=prompts,
        max_model_len=1024,
    )
    print(generated_texts)
    return


if __name__ == "__main__":
    run_rag()
