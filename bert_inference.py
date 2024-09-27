import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, BertForSequenceClassification, BertTokenizer
from datasets import load_dataset, load_from_disk
import torch.nn as nn
import logging
import argparse
import os
import sys
import torch.distributed as dist
import csv 
from torch.utils.data import DistributedSampler
import json

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stderr))
logger.setLevel(logging.INFO)



def cleanup():
    dist.destroy_process_group()


# Initialize the distributed environment
def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    else:
        logger.warning('Not using distributed mode')
        args.rank = -1
        args.world_size = 1
        args.gpu = 0
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=args.rank)
    logger.info(f'Process rank is : {args.rank}')
    torch.cuda.set_device(args.gpu)

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)

# Define encoding function
def encode(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length')

def main(args):
    init_distributed_mode(args)

    # Only the main process should log
    if args.rank == 0:
        logger.info(f'Using {args.world_size} nodes and {torch.cuda.device_count()} GPUs per node.')

    # Load dataset
    # dataset = load_dataset(args.input_files, streaming=True)
    dataset = load_dataset('parquet', data_files=args.input_file, streaming=True)

    # Tokenize and remove unnecessary columns
    ## url and id 
    tokenized_dataset = dataset.map(encode, batched=True, remove_columns=["dump", "file_path", "date", "language", "language_score", "token_count"])
    # sampler = DistributedSampler(tokenized_dataset['train'], num_replicas=world_size, rank=rank)
    # dataloader = DataLoader(tokenized_dataset['train'], batch_size=16, sampler=sampler, collate_fn=lambda x: x)

    # Create model and move it to GPU with id rank
    # device = torch.device(f'cuda:{rank}')

    # Load model
    model = BertForSequenceClassification.from_pretrained(args.checkpoint, local_files_only=True)
    model = model.to(f'cuda:{args.gpu}')  # Move model to the specific GPU

    # Wrap model for distributed data parallel
    # model = DDP(model, device_ids=[args.gpu])

    # Create DataLoader
    def collate_fn(batch):
        input_ids = torch.tensor([example['input_ids'] for example in batch])
        attention_mask = torch.tensor([example['attention_mask'] for example in batch])
        ids = [example['id'] for example in batch]
        urls = [example['url'] for example in batch]
        texts = [example['text'] for example in batch]
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'id': ids, 'url': urls, 'text': texts}

    dataloader = DataLoader(tokenized_dataset['train'], batch_size=64, collate_fn=collate_fn)

    # Run inference
    model.eval()

    output_file = os.path.join(args.output_file, f'predictions_{args.shard_num}_rank_{args.rank}_{args.app_id}.jsonl')
    with open(output_file, mode='w', newline='') as file:
        # writer.writerow(['id', 'url', 'text', 'prediction', 'prbability', 'threshold'])

        for batch in dataloader:
            input_ids = batch['input_ids'].to(f'cuda:{args.gpu}')
            attention_mask = batch['attention_mask'].to(f'cuda:{args.gpu}')
            
            logger.info(f"Input tensors are on device: {input_ids.device}")

            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)

            # Process outputs as needed
            probabilities = torch.softmax(outputs.logits, dim=-1)
            threshold = -100
            predictions = (probabilities[:, 1] >= threshold).int()

            document_scores = [{"id": batch['id'][i], "url": batch['url'][i], "probabilities": probabilities[i, 1].item(), "threshold": threshold} for i in range(len(predictions))]

            # document_scores = [{"id": batch['id'][i], "url": batch['url'][i], "probabilities": probabilities[i, 1].item(), "threshold": threshold} for i in range(len(predictions)) if int(predictions[i].item()) == 1]

            for i, entry in enumerate(document_scores):
                json.dump(entry, file)
                file.write('\n')

                logger.info(f"batch[id][i] is {batch['id'][i]} and batch['url'][i] is {batch['url'][i]} and batch['text'][i] is {batch['text'][i]} and predictions[i].item() is {predictions[i].item()}")
                logger.info(f"batch['text'][i] is {batch['text'][i]} and predictions[i].item() is {predictions[i].item()}")
                logger.info(f"probabilities[i, 1].item() is {probabilities[i, 1].item()}")
            
            logger.info(f"outputs.logits is {outputs.logits}")
            logger.info(f"predictions is {predictions}")
            logger.info(f"probabilities is {probabilities}")
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_files', type=str, required=True, help="Path to the input files")
    parser.add_argument('--output_file', type=str, required=True, help="file to save the output results")
    parser.add_argument('--app_id', type=str, required=True, help="app_id")
    parser.add_argument('--shard_num', type=str, required=True, help="shard_num")
    parser.add_argument('--checkpoint', type=str, required=True, help="checkpoint")
    args = parser.parse_args()
    main(args)
