"""
Dataset for sequence-to-structure prediction training.

Formats protein data for causal language modeling:
- 3Di format: <AA>SEQUENCE<SEP><3Di>STRUCTURE
- Kanzi format: <AA>SEQUENCE<SEP><KANZI>TOKEN_IDS
"""

import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from .foldseek_db import PairedFoldseekDB


# Special tokens for sequence-to-structure format
AA_START = "<AA>"
SS_START = "<3Di>"
KANZI_START = "<KANZI>"
CONTACTS_START = "<CONTACTS>"
SEP_TOKEN = "<SEP>"

# Kanzi token prefix for vocabulary
KANZI_TOKEN_PREFIX = "<K"  # Tokens are <K0>, <K1>, ..., <K999>


def extract_contacts(
    ca_coords: np.ndarray,
    distance_threshold: float = 8.0,
    min_seq_separation: int = 6,
    max_contacts: int = 50,
) -> list[tuple[int, int]]:
    """Extract non-local contacts from C-alpha coordinates.

    Args:
        ca_coords: C-alpha coordinates, shape (L, 3) in Angstroms.
        distance_threshold: Maximum distance for a contact (default 8Å).
        min_seq_separation: Minimum sequence separation |i-j| for non-local contacts.
        max_contacts: Maximum number of contacts to return.

    Returns:
        List of (i, j) tuples representing contacts, sorted by sequence separation.
    """
    n = len(ca_coords)
    contacts = []

    for i in range(n):
        for j in range(i + min_seq_separation, n):
            dist = np.linalg.norm(ca_coords[i] - ca_coords[j])
            if dist < distance_threshold:
                contacts.append((i, j, j - i))  # (i, j, seq_separation)

    # Sort by sequence separation (longer-range contacts first, more informative)
    contacts.sort(key=lambda x: -x[2])

    # Return top contacts as (i, j) tuples (1-indexed for natural language)
    return [(c[0] + 1, c[1] + 1) for c in contacts[:max_contacts]]


class StructurePredictionDataset(Dataset):
    """Map-style dataset for structure prediction training."""

    def __init__(
        self,
        db_path: str | Path,
        tokenizer: Any,
        max_length: int = 1024,
        split: str = "train",
        val_fraction: float = 0.001,
        seed: int = 42,
    ):
        """Initialize dataset.

        Args:
            db_path: Path to Foldseek database (e.g., 'data/foldseek/afdb50/afdb50')
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length for tokenization
            split: 'train' or 'val'
            val_fraction: Fraction of data for validation
            seed: Random seed for train/val split
        """
        self.db_path = Path(db_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.eos_token = tokenizer.eos_token or ""

        # Load index to get dataset size
        self.paired_db = PairedFoldseekDB(db_path)
        self.total_size = len(self.paired_db)

        # Create train/val split indices
        rng = random.Random(seed)
        all_indices = list(range(self.total_size))
        rng.shuffle(all_indices)

        val_size = int(self.total_size * val_fraction)
        if split == "val":
            self.indices = all_indices[:val_size]
        else:
            self.indices = all_indices[val_size:]

        # Open database handles
        self.paired_db.__enter__()

    def __len__(self) -> int:
        return len(self.indices)

    def __del__(self):
        try:
            self.paired_db.__exit__(None, None, None)
        except Exception:
            pass

    def format_example(self, aa_seq: str, ss_seq: str) -> str:
        """Format a sequence pair for training.

        Space-separates characters to ensure 1:1 token alignment.
        """
        aa_spaced = " ".join(aa_seq)
        ss_spaced = " ".join(ss_seq)
        return f"{AA_START} {aa_spaced} {SEP_TOKEN} {SS_START} {ss_spaced} {self.eos_token}"

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        real_idx = self.indices[idx]
        aa_seq, ss_seq = self.paired_db.get_pair(real_idx)

        # Format the text
        text = self.format_example(aa_seq, ss_seq)

        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # Create labels: mask loss on input sequence (before <SEP>)
        labels = input_ids.clone()

        # Find <SEP> token position to mask input
        sep_token_id = self.tokenizer.convert_tokens_to_ids(SEP_TOKEN)
        if sep_token_id is not None and sep_token_id != self.tokenizer.unk_token_id:
            sep_positions = (input_ids == sep_token_id).nonzero(as_tuple=True)[0]
            if len(sep_positions) > 0:
                # Mask everything before and including <SEP>
                sep_pos = sep_positions[0].item()
                labels[:sep_pos + 1] = -100
        else:
            # Fallback: find position of <3Di> token pattern
            ss_start_id = self.tokenizer.convert_tokens_to_ids(SS_START)
            if ss_start_id is not None and ss_start_id != self.tokenizer.unk_token_id:
                ss_positions = (input_ids == ss_start_id).nonzero(as_tuple=True)[0]
                if len(ss_positions) > 0:
                    ss_pos = ss_positions[0].item()
                    labels[:ss_pos] = -100

        # Mask padding
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class StreamingStructureDataset(IterableDataset):
    """Memory-efficient streaming dataset for large protein databases."""

    def __init__(
        self,
        db_path: str | Path,
        tokenizer: Any,
        max_length: int = 1024,
        world_size: int = 1,
        rank: int = 0,
        shuffle_buffer_size: int = 10000,
        seed: int = 42,
    ):
        """Initialize streaming dataset.

        Args:
            db_path: Path to Foldseek database
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
            world_size: Total number of processes
            rank: Current process rank
            shuffle_buffer_size: Size of shuffle buffer
            seed: Random seed
        """
        self.db_path = Path(db_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.world_size = world_size
        self.rank = rank
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.eos_token = tokenizer.eos_token or ""

    def format_example(self, aa_seq: str, ss_seq: str) -> str:
        """Format a sequence pair for training.

        Space-separates characters to ensure 1:1 token alignment.
        """
        aa_spaced = " ".join(aa_seq)
        ss_spaced = " ".join(ss_seq)
        return f"{AA_START} {aa_spaced} {SEP_TOKEN} {SS_START} {ss_spaced} {self.eos_token}"

    def process_example(self, aa_seq: str, ss_seq: str) -> dict[str, torch.Tensor] | None:
        """Process a single example."""
        # Skip if sequences are too long or mismatched
        if len(aa_seq) != len(ss_seq):
            return None
        if len(aa_seq) * 2 + 20 > self.max_length:  # Rough estimate with special tokens
            return None

        text = self.format_example(aa_seq, ss_seq)

        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        labels = input_ids.clone()

        # Mask input sequence (before structure)
        sep_token_id = self.tokenizer.convert_tokens_to_ids(SEP_TOKEN)
        if sep_token_id is not None and sep_token_id != self.tokenizer.unk_token_id:
            sep_positions = (input_ids == sep_token_id).nonzero(as_tuple=True)[0]
            if len(sep_positions) > 0:
                sep_pos = sep_positions[0].item()
                labels[:sep_pos + 1] = -100

        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __iter__(self):
        """Iterate over dataset with sharding and shuffling."""
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        # Combine rank and worker for unique sharding
        effective_rank = self.rank * num_workers + worker_id
        effective_world_size = self.world_size * num_workers

        rng = random.Random(self.seed + effective_rank)
        buffer = []

        with PairedFoldseekDB(self.db_path) as db:
            for idx, aa_seq, ss_seq in db:
                # Shard by index
                if idx % effective_world_size != effective_rank:
                    continue

                example = self.process_example(aa_seq, ss_seq)
                if example is None:
                    continue

                buffer.append(example)

                if len(buffer) >= self.shuffle_buffer_size:
                    rng.shuffle(buffer)
                    for item in buffer:
                        yield item
                    buffer = []

            # Yield remaining items
            if buffer:
                rng.shuffle(buffer)
                for item in buffer:
                    yield item


class KanziStructureDataset(Dataset):
    """Dataset for structure prediction using Kanzi tokens.

    Instead of predicting 3Di tokens (20 vocab), this dataset predicts
    Kanzi tokens (1000 vocab) which can be decoded to 3D coordinates.
    """

    def __init__(
        self,
        db_path: str | Path,
        tokenizer: Any,
        kanzi_tokenizer: Any,
        max_length: int = 1024,
        split: str = "train",
        val_fraction: float = 0.001,
        seed: int = 42,
        max_protein_length: int = 400,
        min_protein_length: int = 100,
        use_contacts: bool = False,
        max_contacts: int = 50,
    ):
        """Initialize Kanzi dataset.

        Args:
            db_path: Path to Foldseek database with C-alpha coordinates.
            tokenizer: Hugging Face tokenizer (with Kanzi tokens added).
            kanzi_tokenizer: KanziTokenizer instance for encoding coordinates.
            max_length: Maximum sequence length for tokenization.
            split: 'train' or 'val'.
            val_fraction: Fraction of data for validation.
            seed: Random seed for train/val split.
            max_protein_length: Maximum protein length to include.
            min_protein_length: Minimum protein length to include.
            use_contacts: Whether to include contact hints in the input.
            max_contacts: Maximum number of contacts to include.
        """
        self.db_path = Path(db_path)
        self.tokenizer = tokenizer
        self.kanzi_tokenizer = kanzi_tokenizer
        self.max_length = max_length
        self.split = split
        self.max_protein_length = max_protein_length
        self.min_protein_length = min_protein_length
        self.use_contacts = use_contacts
        self.max_contacts = max_contacts

        # Load database with C-alpha coordinates
        self.paired_db = PairedFoldseekDB(db_path, include_ca=True)
        self.total_size = len(self.paired_db)

        # Create train/val split indices
        rng = random.Random(seed)
        all_indices = list(range(self.total_size))
        rng.shuffle(all_indices)

        val_size = int(self.total_size * val_fraction)
        if split == "val":
            self.indices = all_indices[:val_size]
        else:
            self.indices = all_indices[val_size:]

        # Open database handles
        self.paired_db.__enter__()

    def __len__(self) -> int:
        return len(self.indices)

    def __del__(self):
        try:
            self.paired_db.__exit__(None, None, None)
        except Exception:
            pass

    def format_example(
        self,
        aa_seq: str,
        kanzi_tokens: list[int],
        contacts: list[tuple[int, int]] | None = None,
    ) -> str:
        """Format a sequence with Kanzi tokens for training.

        Args:
            aa_seq: Amino acid sequence.
            kanzi_tokens: List of Kanzi token indices (0-999).
            contacts: Optional list of (i, j) contact pairs (1-indexed).

        Returns:
            Formatted string like "<AA> M K T ... <SEP> <KANZI> <K599> <K358> ... </s>"
            Or with contacts: "<AA> M K T ... <CONTACTS> 5-20 7-35 ... <SEP> <KANZI> ..."
        """
        aa_spaced = " ".join(aa_seq)
        # Convert Kanzi token indices to special tokens
        kanzi_str = " ".join(f"{KANZI_TOKEN_PREFIX}{t}>" for t in kanzi_tokens)
        eos = self.tokenizer.eos_token or ""

        if contacts:
            # Format contacts as "i-j" pairs (uses standard number tokens)
            contacts_str = " ".join(f"{i}-{j}" for i, j in contacts)
            return f"{AA_START} {aa_spaced} {CONTACTS_START} {contacts_str} {SEP_TOKEN} {KANZI_START} {kanzi_str} {eos}"
        else:
            return f"{AA_START} {aa_spaced} {SEP_TOKEN} {KANZI_START} {kanzi_str} {eos}"

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        real_idx = self.indices[idx]

        try:
            # Get amino acid sequence and C-alpha coordinates
            aa_seq, _, ca_coords = self.paired_db.get_triplet(real_idx)

            # Skip proteins that are too long or too short
            if len(aa_seq) > self.max_protein_length or len(aa_seq) < self.min_protein_length:
                return self.__getitem__((idx + 1) % len(self))

            # Verify C-alpha coordinates match sequence length
            if len(ca_coords) != len(aa_seq):
                # Truncate to shorter length
                min_len = min(len(ca_coords), len(aa_seq))
                aa_seq = aa_seq[:min_len]
                ca_coords = ca_coords[:min_len]

            # Skip if too short after truncation
            if len(aa_seq) < self.min_protein_length:
                return self.__getitem__((idx + 1) % len(self))

            # Encode C-alpha coordinates to Kanzi tokens
            kanzi_tokens = self.kanzi_tokenizer.encode(ca_coords)

            # Verify length match
            if len(kanzi_tokens) != len(aa_seq):
                # Truncate to shorter length
                min_len = min(len(kanzi_tokens), len(aa_seq))
                aa_seq = aa_seq[:min_len]
                kanzi_tokens = kanzi_tokens[:min_len]

        except Exception:
            # Skip problematic entries
            return self.__getitem__((idx + 1) % len(self))

        # Extract contacts if enabled
        contacts = None
        if self.use_contacts:
            contacts = extract_contacts(ca_coords, max_contacts=self.max_contacts)

        # Format the text
        text = self.format_example(aa_seq, kanzi_tokens, contacts=contacts)

        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # Create labels: mask loss on input sequence (before <SEP>)
        labels = input_ids.clone()

        sep_token_id = self.tokenizer.convert_tokens_to_ids(SEP_TOKEN)
        if sep_token_id is not None and sep_token_id != self.tokenizer.unk_token_id:
            sep_positions = (input_ids == sep_token_id).nonzero(as_tuple=True)[0]
            if len(sep_positions) > 0:
                sep_pos = sep_positions[0].item()
                labels[: sep_pos + 1] = -100
        else:
            # Fallback: find <KANZI> token
            kanzi_start_id = self.tokenizer.convert_tokens_to_ids(KANZI_START)
            if kanzi_start_id is not None and kanzi_start_id != self.tokenizer.unk_token_id:
                kanzi_positions = (input_ids == kanzi_start_id).nonzero(as_tuple=True)[0]
                if len(kanzi_positions) > 0:
                    kanzi_pos = kanzi_positions[0].item()
                    labels[:kanzi_pos] = -100

        # Mask padding
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def add_kanzi_tokens(tokenizer) -> int:
    """Add Kanzi tokens to a tokenizer.

    Args:
        tokenizer: Hugging Face tokenizer to modify.

    Returns:
        Number of tokens added.
    """
    # Add special tokens
    special_tokens = {
        "additional_special_tokens": [
            AA_START,
            KANZI_START,
            SEP_TOKEN,
        ]
        + [f"{KANZI_TOKEN_PREFIX}{i}>" for i in range(1000)]  # <K0> to <K999>
    }
    num_added = tokenizer.add_special_tokens(special_tokens)
    return num_added
