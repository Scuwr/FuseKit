from fusekit import Modeling, Datasets
from fusekit.Common import env, Memory

import random
import numpy as np
import torch

random.seed(0)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)

import warnings, os, shutil
warnings.filterwarnings("ignore", category=UserWarning)

def composition_test(device=None, memory_limit=None):
    adapter_path = env.adapters / "composition_test"

    if not adapter_path.exists():
        model = Modeling.Llama3_8b(device=device, memory_limit=memory_limit)
        model = model.init_lora()
        train_dataset = Datasets.CommonsenseQA(model.tokenizer, split='train', data_limit=10, num_shots=1)
        val_dataset = Datasets.CommonsenseQA(model.tokenizer, split='val', data_limit=10, num_shots=1)
        model.finetune(2e-4, 0.002, 1, train_dataset, val_dataset, batch_size=2, 
                    save_path=adapter_path)

    model = Modeling.Llama3_8b(device=device, memory_limit=memory_limit)
    dataset = Datasets.CommonsenseQA(model.tokenizer, split='val', data_limit=10, num_shots=1)

    # Here we compose the same adapter to itself. 
    # Ideal composition requires that composing the same adapter 
    # must not destructively interfere with the output.

    print('\n\n=== Baseline ===')
    model.load_adapters([adapter_path]) # Baseline
    metrics, _ = model.evaluate(dataset)

    print('\n\n=== Destructive Methods ===')
    model.load_adapters([adapter_path, adapter_path]) # Default PEFT composition
    metrics, _ = model.evaluate(dataset)

    model.load_adapters([adapter_path, adapter_path], Modeling.PEMAddition())
    metrics, _ = model.evaluate(dataset)

    print('\n\n=== Safe Compositional Methods ===')
    model.load_adapters([adapter_path, adapter_path], Modeling.LoraHub())
    metrics, _ = model.evaluate(dataset)

    model.load_adapters([adapter_path, adapter_path], Modeling.AdapterSoup())
    metrics, _ = model.evaluate(dataset)

    model.load_adapters([adapter_path, adapter_path], Modeling.AverageOfDeltas())
    metrics, _ = model.evaluate(dataset)

    print('\n\n=== SecureLLM ===')
    model.load_adapters([adapter_path, adapter_path], Modeling.LogitSum())
    metrics, _ = model.evaluate(dataset)

    model.load_adapters([adapter_path, adapter_path], Modeling.LogitMax())
    metrics, _ = model.evaluate(dataset)

    model.load_adapters([adapter_path, adapter_path], Modeling.LogitMean())
    metrics, _ = model.evaluate(dataset)

    shutil.rmtree(adapter_path)

print("Multi-GPU Test")
composition_test()
Memory.clear_cuda()
device, memory_limit = Memory.get_available_gpus()

print("Single GPU Test")
composition_test(device=device[0])