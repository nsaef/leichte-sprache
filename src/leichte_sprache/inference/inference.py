from leichte_sprache.utils.model_utils import generate_vllm


from vllm import LLM, SamplingParams


from argparse import ArgumentParser


def run_inference_vllm(args: ArgumentParser, prompts: list[str]) -> list[str]:
    """Run inference using vLLM. Generates five examples per prompt.

    :param args: arguments including modelname
    :param prompts: list of prompts in the chat format
    :return: list of generated texts (five per prompt)
    """
    llm = LLM(
        model=args.model_name,
        max_model_len=2048,
    )
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
