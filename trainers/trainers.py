import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BatchEncoding
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from .args import LoggingArgs, TrainingArgs
from .data import JSONExtractionDataset
from .generation import Generator
from .metrics import (
    positives_negatives,
    precision_recall_f1,
)


class Trainer:
    def __init__(
        self,
        comments_train: List[Dict[str, Any]],
        comments_val: List[Dict[str, Any]],
        comments_test: List[Dict[str, Any]],
        schemas: Dict[str, Dict[str, Any]],
        training_args: TrainingArgs,
        logging_args: LoggingArgs,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerFast,
        optimizer: torch.optim.Optimizer,
        rng: np.random.Generator,
        device: torch.device
    ) -> None:
        self._comments_train = comments_train
        self._comments_val = comments_val
        self._comments_test = comments_test
        self._schemas = schemas
        
        self._training_args = training_args
        self._logging_args = logging_args
        
        self._rng = rng
        
        self._model = model
        self._tokenizer = tokenizer
        self._optimizer = optimizer
        
        self._device = device
        
        self._model = self._model.to(self._device)
        
        self._generator = Generator(
            model=self._model,
            tokenizer=self._tokenizer,
            device=self._device,
            training_args=self._training_args,
            logging_args=self._logging_args,
        )
        
        self._best_loss = float("inf")

    def _save_checkpoint(
        self,
        name: str,
    ) -> None:
        """Save a model checkpoint.
        
        Parameters
        ----------
        name : str
            The filename.
        """
        checkpoint_dir = self._logging_args.output_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self._tokenizer.save_pretrained(os.path.join(checkpoint_dir, name))
        
        self._model.save_pretrained(os.path.join(checkpoint_dir, name))
        
        torch.save(
            self._optimizer.state_dict(),
            os.path.join(checkpoint_dir, name, "optimizer.pt")
        )
    
    def _validate_metrics(
        self,
        data_loader: DataLoader[Any],
        data_name: str,
    ) -> None:
        self._model.eval()
        
        (
            comments,
            prompts,
            input_ids,
            generations_str,
            generations_json,
        ) = self._generator.do_generation(
            data_loader=data_loader,
            num_beams=self._training_args.n_beams,
        )
        
        self._model.train()
        
        pos_neg = positives_negatives(
            true=[comment["target_json"] for comment in comments],
            pred=generations_json,
            base_properties=[comment["base_property"] for comment in comments],
        )
        
        metrics = precision_recall_f1(pos_neg=pos_neg)
        
        print(f"{data_name} metrics = {metrics}")
    
    def _validate_loss(
        self,
        data_loader: DataLoader[Any],
        data_name: str,
        early_stopping: bool = False,
    ) -> None:
        self._model.eval()
                    
        with torch.no_grad():
            total_loss = 0.
            n_data = 0.

            for batch in data_loader:
                total_loss_batch, n_data_batch = self._compute_loss(
                    batch,
                    mutliply_batch_size=True,
                )
                
                total_loss += total_loss_batch.item()
                n_data += n_data_batch.item()
            
            loss = total_loss / n_data
            
            if early_stopping:
                if loss < self._best_loss:
                    self._save_checkpoint(name="best")
                    self._best_loss = loss

            print(f"{data_name} loss = {loss}")
        
        self._model.train()
    
    def _train(
        self,
        train_data_loader: DataLoader[Any],
        val_loss_data_loader: Optional[DataLoader[Any]] = None,
        val_metrics_data_loader: Optional[DataLoader[Any]] = None,
    ) -> None:
        global_step = 1

        while global_step <= self._training_args.max_iterations:
            for batch in train_data_loader:
                loss, _ = self._compute_loss(batch)
                loss.backward()
                self._optimizer.step()
                self._optimizer.zero_grad()

                print(
                    f"Iteration {global_step}: loss = {loss.item()}"
                )

                if global_step % self._logging_args.val_frequency == 0:
                    if val_loss_data_loader is not None:
                        self._validate_loss(
                            data_loader=val_loss_data_loader,
                            data_name="val",
                            early_stopping=True,
                        )

                    if val_metrics_data_loader is not None:
                        self._validate_metrics(
                            data_loader=val_metrics_data_loader,
                            data_name="val",
                        )
            
                if global_step % self._logging_args.checkpoint_frequency == 0:
                    self._save_checkpoint(name="latest")

                global_step += 1

                if global_step > self._training_args.max_iterations:
                    break
        
    def _compute_loss(
        self,
        batch_tokenized: Tuple[BatchEncoding, BatchEncoding],
        mutliply_batch_size: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prompts_input_ids = batch_tokenized[0]["input_ids"].to(self._device)
        prompts_attention_mask = batch_tokenized[0]["attention_mask"].to(
            self._device,
        )
        
        targets_input_ids = batch_tokenized[1]["input_ids"].to(self._device)

        loss: torch.Tensor = self._model(
            input_ids=prompts_input_ids,
            attention_mask=prompts_attention_mask,
            labels=targets_input_ids,
        ).loss
        
        if mutliply_batch_size:
            loss = loss * prompts_input_ids.shape[0]
        
        n_data = torch.tensor(prompts_input_ids.shape[0], device=self._device)
        
        return loss, n_data

    def _prepare_data_loader(
        self,
        comments: List[Dict[str, Any]],
        schemas: Dict[str, Dict[str, Any]],
        batch_size: int,
        shuffle: bool = True,
        generation: bool = False,
        augmentation: bool = False,
        augmentation_prob: float = 0.5,
    ) -> DataLoader[Any]:
        dataset = JSONExtractionDataset(
            comments=comments,
            schemas=schemas,
            tokenizer=self._tokenizer,
            tokenize_max_length=self._training_args.tokenize_max_length,
            n_examples=self._training_args.n_examples_prompt,
            rng=self._rng,
            augmentation=augmentation,
            augmentation_prob=augmentation_prob,
            generation=generation,
        )
        
        data_loader = DataLoader(
            dataset=dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
        )
        
        return data_loader
    
    def train(
        self,
    ) -> None:
        train_data_loader = self._prepare_data_loader(
            comments=self._comments_train,
            schemas=self._schemas,
            batch_size=self._training_args.train_batch_size,
            augmentation=self._training_args.augmentation,
            augmentation_prob=self._training_args.augmentation_prob,
        )

        if len(self._comments_val) > 0:
            val_loss_data_loader = self._prepare_data_loader(
                comments=self._comments_val,
                schemas=self._schemas,
                batch_size=self._training_args.val_batch_size,
            )
        
            val_metrics_data_loader = self._prepare_data_loader(
                comments=self._comments_val,
                schemas=self._schemas,
                batch_size=self._training_args.gen_batch_size,
                generation=True,
            )

            self._train(
                train_data_loader=train_data_loader,
                val_loss_data_loader=val_loss_data_loader,
                val_metrics_data_loader=val_metrics_data_loader,
            )
        
        else:
            self._train(
                train_data_loader=train_data_loader,
            )
    
    def test(
        self,
    ) -> None:
        test_data_loader = self._prepare_data_loader(
            comments=self._comments_test,
            schemas=self._schemas,
            batch_size=self._training_args.gen_batch_size,
            generation=True,
        )
        
        self._validate_metrics(
            data_loader=test_data_loader,
            data_name="test",
        )
