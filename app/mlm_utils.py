from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
from transformers import DataCollatorForLanguageModeling
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


@dataclass
class DataCollatorForEntityLanguageModeling(DataCollatorForLanguageModeling):
    """Implements Entity masking so that only entites within the 'entities.txt' file will
    be masked.
    """

    tokenizer: PreTrainedTokenizerBase
    mask_probability: float = 0.8
    mlm: bool = True
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    tf_experimental_compile: bool = False
    return_tensors: str = "pt"
    entities_file_path: str = "entities.txt"

    def __post_init__(self):
        super().__post_init__()

    def exclude_non_entities(self, inputs, entities_tokens_ids, special_tokens_mask):
        """Exclude non-entity tokens from token masking."""
        all_entities_positions = []

        # get exception ids
        for instance_ids in inputs.tolist():
            sentence_entities_positions = []

            for entity_id in entities_tokens_ids:
                for i in range(len(instance_ids)):
                    if instance_ids[i : i + len(entity_id)] == entity_id:
                        sequence_ids = list(range(i, i + len(entity_id)))
                        sentence_entities_positions.extend(sequence_ids)

            all_entities_positions.append(sentence_entities_positions)

        new_special_tokens_mask = []

        # manipulate special tokens mask list by excluding
        for stm_idx, stm in enumerate(special_tokens_mask):
            new_stm = []
            for idx, _ in enumerate(stm):
                if idx in all_entities_positions[stm_idx]:
                    new_stm.append(False)
                else:
                    new_stm.append(True)
            new_special_tokens_mask.append(new_stm)

        return torch.tensor(new_special_tokens_mask)

    def torch_mask_tokens(
        self, inputs: Any, special_tokens_mask: Optional[Any] = None
    ) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random,
        10% original.
        """

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training
        # (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True
                )
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        # Exclude non entity tokens from Token Masking
        with open(self.entities_file_path, "r") as f:
            entities = f.read().split("\n")

        entities_tokens = [self.tokenizer.tokenize(entity) for entity in entities]
        entities_tokens_ids = [
            self.tokenizer.convert_tokens_to_ids(tokenized_entity)
            for tokenized_entity in entities_tokens
        ]

        special_tokens_mask = self.exclude_non_entities(
            inputs, entities_tokens_ids, special_tokens_mask
        )

        # CLS and other tokens are excluded from the token masking
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, self.mask_probability)).bool()
            & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
