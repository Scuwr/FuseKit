from fusekit import Modeling, Datasets
import fusekit.Common.env as env
import fusekit.Common.Memory as Memory

import warnings, os, shutil
warnings.filterwarnings("ignore", category=UserWarning)


def similarity_test(device=None, memory_limit=None):
    model = Modeling.Llama3_8b(device=device, memory_limit=memory_limit).init_lora(rank=8, alpha=32, dropout=0.1)
    train_dataset = Datasets.BBHWordSorting(model.tokenizer, data_limit=100)
    adapter_path = env.adapters / "similarity_test"

    model.finetune(lr=0.0002, weight_decay=0.002, epochs=1, batch_size=1,
                train_dataset=train_dataset, val_datasets=None,
                save_path=adapter_path, overwrite=True)

    model = Modeling.Llama3_8b(device=device, memory_limit=memory_limit).load_adapters([adapter_path])
    val_dataset = Datasets.BBHWordSorting(model.tokenizer, data_limit=100)
    model.evaluate(val_dataset)

    shutil.rmtree(adapter_path)

Memory.clear_cuda()
print("Similarity Multi-GPU Test")
similarity_test()
Memory.clear_cuda()
device, memory_limit = Memory.get_available_gpus()

print("Similarity Single GPU Test")
similarity_test(device=device[0])


