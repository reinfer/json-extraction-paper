from argparse import ArgumentParser, Namespace
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from transformers import (
    Adafactor,
    T5ForConditionalGeneration,
    T5TokenizerFast,
)

from trainers.args import (
    MainArgs,
    LoggingArgs,
    TrainingArgs,
)
from trainers.trainers import Trainer


def jsonl_to_data(
    dataset_name: str,
) -> Tuple[
    Dict[str, Any],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    path = f"data/{dataset_name}"
    with open(f"{path}/schemas.json", "r") as f:
        schemas = json.loads(f.read())
    
    comments_train = []
    with open(f"{path}/comments_train.jsonl", "r") as f:
        for line in f:
            comments_train.append(json.loads(line))
    
    comments_val = []
    if "comments_val.jsonl" in os.listdir(path):
        with open(f"{path}/comments_val.jsonl", "r") as f:
            for line in f:
                comments_val.append(json.loads(line))
    
    comments_test = []
    if "comments_test.jsonl" in os.listdir(path):
        with open(f"{path}/comments_test.jsonl", "r") as f:
            for line in f:
                comments_test.append(json.loads(line))

    return schemas, comments_train, comments_val, comments_test

def download_data(
    args: Namespace,
) -> Tuple[
    np.random.Generator,
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, Dict[str, Any]],
]:
    schemas: Dict[str, Dict[str, Any]] = {}
    comments_train: List[Dict[str, Any]] = []
    comments_val: List[Dict[str, Any]] = []
    comments_test: List[Dict[str, Any]] = []
    
    for dataset_name in args.datasets.split(","):
        dataset_name = dataset_name.strip()
        
        (
            schemas_d,
            comments_train_d,
            comments_val_d,
            comments_test_d,
        ) = jsonl_to_data(
            dataset_name=dataset_name,
        )
        
        schemas[dataset_name] = schemas_d
        comments_train += comments_train_d
        comments_val += comments_val_d
        comments_test += comments_test_d
    
    rng = np.random.default_rng()
    
    return (
        rng,
        comments_train,
        comments_val,
        comments_test,
        schemas,
    )


def configure_parser(parser: ArgumentParser) -> None:
    MainArgs.add_args(parser)
    TrainingArgs.add_args(parser)
    LoggingArgs.add_args(parser)


def run(args: Namespace) -> None:
    logging_args = LoggingArgs.from_args(args)
    training_args = TrainingArgs.from_args(args)
    
    (
        rng,
        comments_train,
        comments_val,
        comments_test,
        schemas,
    ) = download_data(
        args=args
    )
    
    load_from = (
        args.model_name if args.resume_from is None else args.resume_from
    )
    
    model = T5ForConditionalGeneration.from_pretrained(
        load_from,
    )
    tokenizer = T5TokenizerFast.from_pretrained(
        load_from,
    )
    
    tokenizer.add_tokens(["{", "}"])
    model.resize_token_embeddings(len(tokenizer))
    
    optimizer = Adafactor(
        params=model.parameters(),
        lr=training_args.learning_rate,
        scale_parameter=False,
        relative_step=False,
    )
    
    if args.resume_from is not None:
        optimizer.load_state_dict(
            torch.load(os.path.join(args.resume_from, "optimizer.pt")),
        )
    
    device = (
        torch.device("cuda:0")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    trainer = Trainer(
        comments_train=comments_train,
        comments_val=comments_val,
        comments_test=comments_test,
        schemas=schemas,
        training_args=training_args,
        logging_args=logging_args,
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        rng=rng,
        device=device,
    )

    if not args.no_train:
        trainer.train()
    
    if args.test:
        trainer.test()


if __name__ == "__main__":
    parser = ArgumentParser()
    configure_parser(parser)
    args = parser.parse_args()
    
    run(args)
