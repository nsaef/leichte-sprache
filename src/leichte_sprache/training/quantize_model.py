from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer


def run_quantization(model_path: str, quant_path: str):
    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",
    }

    # Load model
    model = AutoAWQForCausalLM.from_pretrained(
        model_path, **{"low_cpu_mem_usage": True}
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Quantize
    model.quantize(tokenizer, quant_config=quant_config)

    # Save quantized model
    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)


if __name__ == "__main__":
    model_path = "/home/nasrin/data/trainings/discolm-ls-01/merged"
    quant_path = "/home/nasrin/data/trainings/discolm-ls-01/awq"
    run_quantization(model_path=model_path, quant_path=quant_path)
