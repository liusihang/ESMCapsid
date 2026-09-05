import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
from pathlib import Path
import torch
import torch.distributed as dist
from transformers import (
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
    AutoTokenizer,
)

from datasets import load_dataset, load_from_disk
import numpy as np

# =================================================================

# =================================================================
MODEL_NAME = "<EXTERNAL_SOURCE_DIR>/esmc_600m_MLMfinetuned_shuffle" 
TRAIN_FILE = "<EXTERNAL_SOURCE_PATH>/filtered_BadResults.csv"
SEQ_COLUMN_NAME = "seq"
OUTPUT_DIR = "./esmc_600m_MLMfinetunedBased_shuffle"
MAX_SEQ_LENGTH = 1480 
USE_BF16 = True


TEST_SIZE = 0.1
VAL_SIZE = 0.05
SEED = 42


CACHE_DIR = "./tokenized_cache"
DEEPSPEED_CONFIG = Path(__file__).resolve().with_name("ds_config.json")

# =================================================================

# =================================================================
def prepare_features(examples, tokenizer, max_seq_length, col_name):
    """Tokenize sequences without padding"""
    sequences = examples[col_name]
    cleaned_sequences = []
    for seq in sequences:
        if isinstance(seq, str) and len(seq.strip()) > 0:
            cleaned_sequences.append(seq.strip())
        else:
            cleaned_sequences.append("")
    
    tokenized_inputs = tokenizer(
        cleaned_sequences,
        truncation=True,
        max_length=max_seq_length,
        padding=False,
        return_special_tokens_mask=True
    )
    return tokenized_inputs


def filter_empty_sequences(example):
    """Remove empty sequences"""
    return len(example['input_ids']) > 2


# =================================================================

# =================================================================
class MLMDataCollatorNoPadLoss(DataCollatorForLanguageModeling):
    def torch_call(self, examples):
        batch = super().torch_call(examples)
        if "attention_mask" in batch and "labels" in batch:
            batch["labels"] = batch["labels"].masked_fill(
                batch["attention_mask"] == 0, -100
            )
        return batch


# =================================================================

# =================================================================
def main():
    
    
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    
    
    if world_size > 1:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank) 

    set_seed(SEED)
    
    is_main_process = (local_rank == -1) or (local_rank == 0)
    
    if is_main_process:
        print(f" World size: {world_size}, Local rank: {local_rank}")
        print(f" Loading ESMC model from {MODEL_NAME}...")
    
    # -------------------------------------------------------------
    
    # -------------------------------------------------------------
    model = AutoModelForMaskedLM.from_pretrained(
        MODEL_NAME, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16 if USE_BF16 else torch.float16
    )
    
    if is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        print(f" Total parameters: {total_params:,}")

    # -------------------------------------------------------------
    
    # -------------------------------------------------------------
    try:
        if hasattr(model, "tokenizer") and model.tokenizer is not None:
            tokenizer = model.tokenizer
            if is_main_process:
                print(" Using model.tokenizer")
        else:
            raise AttributeError("Model has no tokenizer attribute")
    except Exception:
        if is_main_process:
            print(f" Falling back to facebook/esm2_t6_8M_UR50D tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")

    if tokenizer.pad_token_id is None or tokenizer.pad_token_id != 1:
        tokenizer.pad_token = "<pad>"
        tokenizer.pad_token_id = 1

    # -------------------------------------------------------------
    
    # -------------------------------------------------------------
    cache_file_train = os.path.join(CACHE_DIR, "train_tokenized")
    cache_file_val = os.path.join(CACHE_DIR, "val_tokenized")
    cache_file_test = os.path.join(CACHE_DIR, "test_tokenized")
    
    
    if is_main_process:
        os.makedirs(CACHE_DIR, exist_ok=True)
        print(f"\n Processing Dataset...")
        
        
        raw_dataset = load_dataset("csv", data_files={"train": TRAIN_FILE})
        
        
        split_dataset = raw_dataset["train"].train_test_split(test_size=TEST_SIZE, seed=SEED, shuffle=True)
        train_val_split = split_dataset["train"].train_test_split(test_size=VAL_SIZE / (1 - TEST_SIZE), seed=SEED, shuffle=True)
        
        train_dataset = train_val_split["train"]
        val_dataset = train_val_split["test"]
        test_dataset = split_dataset["test"]
        
        print(f" Dataset split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
        
        # 3. Tokenize
        def tokenize_function(examples):
            return prepare_features(examples, tokenizer, MAX_SEQ_LENGTH, SEQ_COLUMN_NAME)
        
        
        NUM_PROC = 8 
        
        print(" Tokenizing Train...")
        tokenized_train = train_dataset.map(tokenize_function, batched=True, num_proc=NUM_PROC, remove_columns=train_dataset.column_names, load_from_cache_file=True)
        
        print(" Tokenizing Val...")
        tokenized_val = val_dataset.map(tokenize_function, batched=True, num_proc=NUM_PROC, remove_columns=val_dataset.column_names, load_from_cache_file=True)
        
        print(" Tokenizing Test...")
        tokenized_test = test_dataset.map(tokenize_function, batched=True, num_proc=NUM_PROC, remove_columns=test_dataset.column_names, load_from_cache_file=True)
        
        # 4. Filter
        print(" Filtering empty sequences...")
        tokenized_train = tokenized_train.filter(filter_empty_sequences, num_proc=NUM_PROC)
        tokenized_val = tokenized_val.filter(filter_empty_sequences, num_proc=NUM_PROC)
        tokenized_test = tokenized_test.filter(filter_empty_sequences, num_proc=NUM_PROC)
        
        # 5. Save
        print(" Saving datasets to disk...")
        tokenized_train.save_to_disk(cache_file_train)
        tokenized_val.save_to_disk(cache_file_val)
        tokenized_test.save_to_disk(cache_file_test)
        print(" Data processing complete.")

    
    
    if world_size > 1:
        dist.barrier()
        print(f" Rank {local_rank} passed barrier.")

    
    if is_main_process:
        print("\n Loading tokenized datasets from disk (all processes)...")
    
    tokenized_train = load_from_disk(cache_file_train)
    tokenized_val = load_from_disk(cache_file_val)
    tokenized_test = load_from_disk(cache_file_test)
    
    if is_main_process:
        train_lengths = [len(x) for x in tokenized_train['input_ids'][:100]]
        print(f"\n Sequence length stats - Min: {min(train_lengths)}, Max: {max(train_lengths)}, Mean: {np.mean(train_lengths):.1f}")

    # -------------------------------------------------------------
    # D. Data Collator
    # -------------------------------------------------------------
    data_collator = MLMDataCollatorNoPadLoss(
        tokenizer=tokenizer, 
        mlm=True, 
        mlm_probability=0.15,
        pad_to_multiple_of=8
    )

    # -------------------------------------------------------------
    
    # -------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=False,
        
        deepspeed=str(DEEPSPEED_CONFIG),
        
        # Synchronized with the manuscript: 100,000 optimization steps.
        max_steps=100_000,
        per_device_train_batch_size=28,
        per_device_eval_batch_size=36,
        gradient_accumulation_steps=4,
        
        learning_rate=1e-5,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        
        bf16=True,      
        fp16=False,
        
        logging_steps=50,
        logging_first_step=True,
        
        eval_strategy="steps",
        eval_steps=500,
        
        save_strategy="steps",
        save_steps=500,
        save_total_limit=4,
        
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        report_to="none",
        dataloader_num_workers=4,  
        dataloader_pin_memory=True,
        
        group_by_length=True,
        max_grad_norm=1.0,
        
        seed=SEED,
        data_seed=SEED,
        local_rank=local_rank, 
    )

    # -------------------------------------------------------------
    # F. Trainer
    # -------------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # -------------------------------------------------------------
    
    # -------------------------------------------------------------
    if is_main_process:
        print("\n" + "="*60)
        print(" Pre-training Checks")
        print("="*60)
        
        dl = trainer.get_train_dataloader()
        batch = next(iter(dl))
        print(f" Batch shape: {batch['input_ids'].shape}")
        
        model.eval()
        with torch.no_grad():
            device = next(model.parameters()).device
            batch_on_device = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**batch_on_device)
        print(f" Initial loss: {outputs.loss.item():.4f}")

    # -------------------------------------------------------------
    
    # -------------------------------------------------------------
    if is_main_process:
        print("\n" + "="*60)
        print(" Starting Training")
        print("="*60)
    
    model.train()
    train_result = trainer.train(resume_from_checkpoint=False)
    
    
    trainer.save_model()
    trainer.save_state()
    
    if is_main_process:
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

    # -------------------------------------------------------------
    
    # -------------------------------------------------------------
    if is_main_process:
        print("\n" + "="*60)
        print(" Evaluating on Test Set")
        print("="*60)
    
    test_results = trainer.evaluate(eval_dataset=tokenized_test)
    
    if is_main_process:
        print(f"\n Test Results:")
        print(f"   Loss: {test_results['eval_loss']:.4f}")
        print(f"   Perplexity: {np.exp(test_results['eval_loss']):.2f}")
        trainer.save_metrics("test", test_results)
        print("\n Training completed!")
        print(f" Model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
