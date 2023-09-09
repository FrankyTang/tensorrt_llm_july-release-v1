'''
Convert pytorch model.
'''
import argparse
import configparser
import dataclasses
import os
from pathlib import Path

import torch
import torch.multiprocessing as multiprocessing
from tqdm import tqdm
from utils.convert import split_and_save_weight

from tensorrt_llm._utils import str_dtype_to_torch, torch_to_numpy


@dataclasses.dataclass(frozen=True)
class ProgArgs:
    out_dir: str
    in_file: str
    tensor_parallelism: int = 1
    processes: int = 4
    calibrate_kv_cache: bool = False
    smoothquant: float = None
    model: str = "uniad"
    storage_type: str = "fp32"
    dataset_cache_dir: str = None

    @staticmethod
    def parse(args=None) -> 'ProgArgs':
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument(
            '--out-dir',
            '-o',
            type=str,
            help='file name of output directory',
            required=True)
        parser.add_argument(
            '--in-file',
            '-i',
            type=str,
            help='file name of input checkpoint file',
            required=True)
        parser.add_argument(
            '--tensor-parallelism',
            '-tp',
            type=int,
            help='Requested tensor parallelism for inference',
            default=1)
        parser.add_argument(
            "--processes",
            "-p",
            type=int,
            help="How many processes to spawn for conversion (default: 4). Set it to a lower value to reduce RAM usage.",
            default=4)
        parser.add_argument(
            "--calibrate-kv-cache",
            "-kv",
            action="store_true",
            help="Generate scaling factors for KV cache. Used for storing KV cache in int8."
        )
        parser.add_argument(
            "--smoothquant",
            "-sq",
            type=float,
            default=None,
            help="Set the α parameter (see https://arxiv.org/pdf/2211.10438.pdf)"
            " to Smoothquant the model, and output int8 weights."
            " A good first try is 0.5. Must be in [0, 1]")
        parser.add_argument(
            "--model",
            default="uniad",
            type=str,
            help="Specify uniad variants to convert checkpoints correctly")
        parser.add_argument("--storage-type",
                            "-t",
                            type=str,
                            default="float32",
                            choices=["float32", "float16", "bfloat16"])
        parser.add_argument("--dataset-cache-dir",
                            type=str,
                            default=None,
                            help="cache dir to load the hugging face dataset")
        return ProgArgs(**vars(parser.parse_args(args)))


@torch.no_grad()
def smooth_gpt_model(model, scales, alpha):
    # Smooth the activation and weights with smoother = $\diag{s}$
    for name, module in model.named_modules():
        if not isinstance(module, GPT2Block):
            continue

        # qkv_proj
        layer_name = name + ".attn.c_attn"
        smoother = smooth_gemm(module.attn.c_attn.weight.T,
                               scales[layer_name]["x"], module.ln_1.weight,
                               module.ln_1.bias, alpha)
        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
        scales[layer_name]["w"] = module.attn.c_attn.weight.abs().max(dim=0)[0]

        # fc1
        layer_name = name + ".mlp.c_fc"
        smoother = smooth_gemm(module.mlp.c_fc.weight.T,
                               scales[layer_name]["x"], module.ln_2.weight,
                               module.ln_2.bias, alpha)
        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
        scales[layer_name]["w"] = module.mlp.c_fc.weight.abs().max(dim=0)[0]


# StarCoder uses nn.Linear for these following ops whose weight matrix is transposed compared to transformer.Conv1D
def transpose_weights(hf_name, param):
    weight_to_transpose = ["c_attn", "c_proj", "c_fc"]
    if any([k in hf_name for k in weight_to_transpose]):
        if len(param.shape) == 2:
            param = param.transpose(0, 1)
    return param


@torch.no_grad()
def pytorch_converter(args: ProgArgs):
    infer_tp = args.tensor_parallelism
    saved_dir = Path(args.out_dir) / f"{infer_tp}-gpu"
    saved_dir.mkdir(parents=True, exist_ok=True)

    model = torch.load(args.in_file)["state_dict"]
    act_range = {}

    config = configparser.ConfigParser()
    config["uniad"] = {}
    for key in vars(args):
        config["uniad"][key] = f"{vars(args)[key]}"
    # for k, v in vars(model.config).items():
    #     config["uniad"][k] = f"{v}"
    config["uniad"]["storage_dtype"] = args.storage_type
    config["uniad"]["multi_query_mode"] = str(False)
    with open(saved_dir / "config.ini", 'w') as configfile:
        config.write(configfile)

    storage_type = str_dtype_to_torch(args.storage_type)
    global_ft_weights = []
    int8_outputs = None
    if args.calibrate_kv_cache:
        int8_outputs = "kv_cache_only"
    if args.smoothquant is not None:
        int8_outputs = "all"

    starmap_args = []
    for name in model.keys():
        # backbone is pursely cnn, skip
        if "img_backbone" in name or "img_neck" in name:
            continue

        param = model[name]
        if "weight" not in name and "bias" not in name:
            continue

        torch_to_numpy(param.to(storage_type).cpu()).tofile(
            saved_dir / f"{name}.bin")

    starmap_args = tqdm(starmap_args, desc="saving weights")
    if args.processes > 1:
        with multiprocessing.Pool(args.processes) as pool:
            pool.starmap(split_and_save_weight, starmap_args)
    else:
        # simpler for debug situations
        for starmap_arg in starmap_args:
            split_and_save_weight(*starmap_arg)


def run_conversion(args: ProgArgs):
    print("\n=============== Arguments ===============")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("========================================")
    pytorch_converter(args)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    run_conversion(ProgArgs.parse())
