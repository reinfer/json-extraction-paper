import json
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from transformers.generation_utils import BeamSearchEncoderDecoderOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from .args import LoggingArgs, TrainingArgs


class Generator:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerFast,
        device: torch.device,
        training_args: TrainingArgs,
        logging_args: LoggingArgs,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        
        self._device = device
        
        self._training_args = training_args
        self._logging_args = logging_args
    
    def _do_generation_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_beams: int,
    ) -> Tuple[
        BeamSearchEncoderDecoderOutput,
        List[str],
        List[Optional[List[Dict[str, Any]]]],
    ]:
        with torch.no_grad():
            generations = self._model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self._training_args.generate_max_length,
                num_beams=num_beams,
                return_dict_in_generate=True,
            )
        
        assert isinstance(generations, BeamSearchEncoderDecoderOutput)
        
        generations_str = self._tokenizer.batch_decode(
            generations.sequences,
            skip_special_tokens=True,
        )
        
        generations_json = []
        
        for i in range(len(generations_str)):
            try:
                generation_json = json.loads("[{" + generations_str[i])
            except json.JSONDecodeError:
                generation_json = None
            
            generations_json.append(generation_json)
        
        return generations, generations_str, generations_json
    
    def do_generation(
        self,
        data_loader: DataLoader[Any],
        num_beams: int,
        num_generations: Optional[int] = None,
    ) -> Tuple[
        List[Dict[str, Any]],
        List[str],
        List[List[int]],
        List[str],
        List[Optional[List[Dict[str, str]]]],
    ]:
        self._model = self._model.bfloat16()
        
        comments_all = []
        prompts_all = []
        input_ids_all = []
        generations_str_all = []
        generations_json_all = []
        
        for batch in data_loader:
            batch_tokenized, prompts, comments = batch
            
            input_ids = batch_tokenized["input_ids"].to(self._device)
            attention_mask = batch_tokenized["attention_mask"].to(self._device)
            
            (
                generations,
                generations_str,
                generations_json,
            ) = self._do_generation_batch(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=num_beams,
            )
            
            comments_all += comments
            prompts_all += prompts
            input_ids_all += input_ids.cpu().tolist()
            generations_str_all += generations_str
            generations_json_all += generations_json
            
            if (
                num_generations is not None
                and len(comments_all) >= num_generations
            ):
                break
        
        self._model = self._model.float()
        
        return (
            comments_all,
            prompts_all,
            input_ids_all,
            generations_str_all,
            generations_json_all,
        )
