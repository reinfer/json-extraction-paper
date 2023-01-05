import json
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from torch.utils.data import Dataset
from transformers import BatchEncoding
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


PROMPT_PREFIX_JSON_SCHEMAS = "The valid JSON schemas are:"
PROMPT_PREFIX_TEXT = "[[Text:]]"
PROMPT_PREFIX_SUBJECT = "Subject: "
PROMPT_PREFIX_BODY = "Message: "
PROMPT_PREFIX_JSON = "[[JSON:]]"

PROMPT_INSTRUCTION = "Write a JSON summarizing the text.\n\n"
PROMPT_SEPARATOR = "\n\n"


def _augment_schema(
    schema: Dict[str, str],
    base_property: str,
    rng: np.random.Generator,
) -> Dict[str, str]:
    # Augmentation modes are:
    # - delete keys
    # - reorder keys
    # - both
    delete_keys = rng.choice(2)
    reorder_keys = rng.choice(2)
    
    if delete_keys:
        deletable_keys = [k for k in schema if k != base_property]
        to_delete = rng.choice(2, len(deletable_keys))
        retained_keys = [base_property] + [
            deletable_keys[i] for i in range(len(deletable_keys))
            if not to_delete[i]
        ]
    else:
        retained_keys = list(schema)
    
    if reorder_keys:
        ordered_keys = rng.permutation(retained_keys).tolist()
    else:
        ordered_keys = retained_keys
    
    schema_augmented = {k: schema[k] for k in ordered_keys}
    
    return schema_augmented

def comment_to_prompt_and_target(
    comment: Dict[str, Any],
    schemas: Dict[str, Dict[str, Any]],
    is_email: bool,
    augmentation: bool,
    augmentation_prob: float,
    rng: np.random.Generator,
) -> Tuple[str, str, List[Dict[str, str]]]:
    dataset = comment["dataset"]
    base_property = comment["base_property"]
    
    # variable is called `intents`` but these could be e.g. article types for 
    # DBpedia
    intents = [i.strip() for i in comment[f"{base_property}s"].split(",")]
    translated_from_json = json.loads(comment["json"])
    schemas_dataset = schemas[dataset]
    
    augmented_schemas: Dict[str, Dict[str, str]] = {}
    
    prompt_lines = [f'{PROMPT_INSTRUCTION}{PROMPT_PREFIX_JSON_SCHEMAS}']

    for intent in intents:
        schema = schemas_dataset[intent]["schema"]
        
        if augmentation:
            to_augment = rng.choice(
                2, p=[1 - augmentation_prob, augmentation_prob]
            )
        else:
            to_augment = False
        
        if to_augment:
            schema_augmented = _augment_schema(
                schema=schema,
                base_property=base_property,
                rng=rng,
            )
            augmented_schemas[intent] = schema_augmented
            prompt_lines.append(json.dumps(schema_augmented))
        else:
            prompt_lines.append(json.dumps(schema))
    
    prompt_lines.append(f'\n{PROMPT_PREFIX_TEXT}')
    
    if is_email:
        subject_str = comment['subject']
        body_str = comment['body']
        
        prompt_lines.append(f"{PROMPT_PREFIX_SUBJECT}{subject_str}")
        prompt_lines.append(f"{PROMPT_PREFIX_BODY}{body_str}")
    else:
        text = comment['body']
        prompt_lines.append(text)
    
    prompt_lines.append(f'\n{PROMPT_PREFIX_JSON}')
    prompt_lines.append("[{")
    
    prompt = "\n".join(prompt_lines)
    
    target_json: List[Dict[str, str]] = []
    
    for t in translated_from_json:
        t_out: Dict[str, str] = {}
        
        if t[base_property] in augmented_schemas:
            for key in augmented_schemas[t[base_property]]:
                if key == base_property:
                    t_out[key] = t[key]
            
        else:
            for key, value in t.items():
                t_out[key] = value
        
        target_json.append(t_out)
    
    target = json.dumps(target_json)[2:]
    
    return prompt, target, target_json


class JSONExtractionDataset(Dataset[Any]):
    def __init__(
        self,
        comments: List[Dict[str, Any]],
        schemas: Dict[str, Dict[str, Any]],
        tokenizer: PreTrainedTokenizerFast,
        tokenize_max_length: int,
        n_examples: int,
        rng: np.random.Generator,
        augmentation: bool = False,
        augmentation_prob: float = 0.5,
        generation: bool = False,
    ) -> None:
        super().__init__()
        
        self._comments = comments
        self._schemas = schemas
        self._tokenizer = tokenizer
        self._tokenize_max_length = tokenize_max_length
        
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._n_examples = n_examples
        self._augmentation = augmentation
        self._augmentation_prob = augmentation_prob
        self._generation = generation
        
        self._rng = rng
        
        if self._n_examples > 0:
            self._indices_by_intent: Dict[str, Set[int]] = {}
            
            for i in range(len(self._comments)):
                comment = self._comments[i]
                base_property = comment["base_property"]
                intents = {
                    intent.strip() for intent in
                    comment[f"{base_property}s"].split(",")
                }
                
                for intent in intents:
                    if intent in self._indices_by_intent:
                        self._indices_by_intent[intent].add(i)
                    else:
                        self._indices_by_intent[intent] = set([i])

    def __len__(
        self,
    ) -> int:
        return len(self._comments)
    
    def _get_prompt_examples(
        self,
        comment: Dict[str, Any],
        index: int,
    ) -> List[Dict[str, Any]]:
        example_inds: Set[int] = set([index])
        examples: List[Dict[str, Any]] = []
        
        base_property = comment["base_property"]
        
        intents = [
            intent.strip() for intent in
            comment[f"{base_property}s"].split(",")
        ]
        
        while len(examples) < self._n_examples:
            intent_example = intents[self._rng.choice(len(intents))]
            
            available_inds_set = self._indices_by_intent[
                intent_example
            ].difference(example_inds)
            
            if len(available_inds_set) < 1:
                available_inds_set = set(
                    range(len(self._comments))
                ).difference(example_inds)
            
            available_inds = list(available_inds_set)
            
            example_ind = available_inds[self._rng.choice(len(available_inds))]
            example_inds.add(example_ind)
            examples.append(self._comments[example_ind])
        
        return examples

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        comment = self._comments[index]
        
        dataset_type = comment["dataset_type"]
        is_email = dataset_type == "email"
        
        prompt, target, target_json = comment_to_prompt_and_target(
            comment=comment,
            schemas=self._schemas,
            is_email=is_email,
            augmentation=self._augmentation,
            augmentation_prob=self._augmentation_prob,
            rng=self._rng,
        )
        
        comment["prompt"] = prompt
        comment["target"] = target
        comment["target_json"] = target_json
        
        if self._n_examples > 0:
            examples = self._get_prompt_examples(comment=comment, index=index)
            
            for comment_example in examples:
                prompt, target, target_json = comment_to_prompt_and_target(
                    comment=comment_example,
                    schemas=self._schemas,
                    is_email=is_email,
                    augmentation=self._augmentation,
                    augmentation_prob=self._augmentation_prob,
                    rng=self._rng,
                )
                
                comment_example["prompt"] = prompt
                comment_example["target"] = target
                comment_example["target_json"] = target_json
            
        else:
            examples = []
        
        return examples, comment

    def _collate_fn_train(
        self,
        batch: List[Tuple[List[Dict[str, Any]], Dict[str, Any]]],
    ) -> Tuple[BatchEncoding, BatchEncoding]:
        if self._n_examples > 0:
            prompts = []
            targets = []
            
            for i in range(len(batch)):
                examples_i = batch[i][0]
                comment_i = batch[i][1]
                
                prompt_i = PROMPT_SEPARATOR.join(
                    [f'{e["prompt"]}{e["target"]}' for e in examples_i] +
                    [comment_i["prompt"]]
                )
                
                target_i = comment_i["target"]
                
                prompts.append(prompt_i)
                targets.append(target_i)
        else:
            prompts = [b[1]["prompt"] for b in batch]
            targets = [b[1]["target"] for b in batch]
        
        prompts_tokenized = self._tokenizer(
            text=prompts,
            padding=True,
            truncation=True,
            max_length=self._tokenize_max_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        
        targets_tokenized = self._tokenizer(
            text=targets,
            padding=True,
            truncation=True,
            max_length=self._tokenize_max_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        
        return prompts_tokenized, targets_tokenized
    
    def _collate_fn_generation(
        self,
        batch: List[Tuple[List[Dict[str, Any]], Dict[str, Any]]],
    ) -> Tuple[BatchEncoding, List[str], List[Dict[str, Any]]]:
        if self._n_examples > 0:
            prompts = []
            
            for i in range(len(batch)):
                examples_i = batch[i][0]
                comment_i = batch[i][1]
                
                prompt_i = PROMPT_SEPARATOR.join(
                    [f'{e["prompt"]}{e["target"]}' for e in examples_i] +
                    [comment_i["prompt"]]
                )
                
                prompts.append(prompt_i)
        else:
            prompts = [b[1]["prompt"] for b in batch]
        
        comments = [b[1] for b in batch]
        
        prompts_tokenized = self._tokenizer(
            text=prompts,
            padding=True,
            truncation=True,
            max_length=self._tokenize_max_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        
        return prompts_tokenized, prompts, comments
    
    def collate_fn(
        self,
        batch: List[Tuple[List[Dict[str, Any]], Dict[str, Any]]],
    ) -> Any:
        if self._generation:
            return self._collate_fn_generation(batch)
        else:
            return self._collate_fn_train(batch)
