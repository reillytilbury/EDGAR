"""
genetic_helpers_v2.py

Refactored genetic algorithm helpers using proper class abstractions.
Provides Program dataclass, Island class, and ProgramRegistry for efficient operations.
"""

from dataclasses import dataclass, field, asdict
from typing import Callable, Optional, Tuple, List, Dict, Iterator, Any
from collections import defaultdict
from pathlib import Path
import hashlib
import logging

import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml


# Default values for EvolutionConfig
DEFAULT_CAPACITY_PER_ISLAND = 12
DEFAULT_N_MIGRANTS = 2
DEFAULT_OVERLAP_THRESHOLD = 6
DEFAULT_LOSS_TOLERANCE = 0.01
DEFAULT_COSINE_TOLERANCE = 0.99
DEFAULT_MIN_WISE_PROGRAMS = 0
DEFAULT_LARGE_LM_NAME = ""
DEFAULT_LOSS_TYPE = 'train_loss'


@dataclass
class EvolutionConfig:
    """Configuration for evolutionary algorithm parameters."""
    
    # Island capacity
    capacity_per_island: int = DEFAULT_CAPACITY_PER_ISLAND
    
    # Migration
    n_migrants: int = DEFAULT_N_MIGRANTS
    
    # Deduplication
    overlap_threshold: int = DEFAULT_OVERLAP_THRESHOLD
    loss_tolerance: float = DEFAULT_LOSS_TOLERANCE
    cosine_tolerance: float = DEFAULT_COSINE_TOLERANCE
    
    # Wise program preservation
    min_wise_programs: int = DEFAULT_MIN_WISE_PROGRAMS
    large_lm_name: str = DEFAULT_LARGE_LM_NAME
    
    # Loss type for comparisons
    loss_type: str = DEFAULT_LOSS_TYPE
    
    @classmethod
    def from_yaml(cls, yaml_path: Optional[Path] = None) -> 'EvolutionConfig':
        """
        Load configuration from experiment.yaml.
        
        Args:
            yaml_path: Path to experiment.yaml. If None, uses default config location.
            
        Returns:
            EvolutionConfig with values from YAML (falls back to defaults for missing keys).
        """
        if yaml_path is None:
            # Default to config/experiment.yaml relative to this file's parent
            yaml_path = Path(__file__).parent.parent / 'config' / 'experiment.yaml'
        
        try:
            with open(yaml_path, 'r') as f:
                config_data = yaml.safe_load(f)
        except FileNotFoundError:
            logging.warning(f"Config file not found at {yaml_path}, using defaults")
            return cls()
        
        evolution_config = config_data.get('evolution', {})
        return cls.from_dict(evolution_config)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EvolutionConfig':
        """Create config from a dictionary (e.g., already-loaded YAML section)."""
        return cls(
            capacity_per_island=config_dict.get('capacity_per_island', DEFAULT_CAPACITY_PER_ISLAND),
            n_migrants=config_dict.get('n_migrants', DEFAULT_N_MIGRANTS),
            overlap_threshold=config_dict.get('overlap_threshold', DEFAULT_OVERLAP_THRESHOLD),
            loss_tolerance=config_dict.get('loss_tolerance', DEFAULT_LOSS_TOLERANCE),
            cosine_tolerance=config_dict.get('cosine_tolerance', DEFAULT_COSINE_TOLERANCE),
            min_wise_programs=config_dict.get('min_wise_programs', DEFAULT_MIN_WISE_PROGRAMS),
            large_lm_name=config_dict.get('large_lm_name', DEFAULT_LARGE_LM_NAME),
            loss_type=config_dict.get('loss_type', DEFAULT_LOSS_TYPE),
        )


# Type aliases for event hooks
ProgramEventHook = Callable[['Program', 'Island'], None]
MigrationEventHook = Callable[['Program', 'Island', 'Island'], None]  # program, source, dest
PruneEventHook = Callable[[List['Program'], 'Island'], None]  # removed programs, island
DeduplicateEventHook = Callable[[List['Program'], 'Island'], None]  # removed programs, island


@dataclass
class Program:
    """Typed representation of a single evolved program."""
    
    # Code representations
    code_string: str
    program: Callable  # JAX-compatible neuron model
    param_estimator_code_string: str
    param_estimator: Callable
    
    # Identity
    iteration_number: int
    birth_island: int
    batch_index: int
    
    # Performance metrics
    train_loss: float
    test_loss: Optional[float] = None
    initial_loss: Optional[float] = None
    
    # Parameters
    params: Optional[jnp.ndarray] = None
    initial_params: Optional[jnp.ndarray] = None
    
    # Metadata
    llm_name: Optional[str] = None
    parent1_id: Optional[Tuple[int, int, int]] = None
    parent2_id: Optional[Tuple[int, int, int]] = None
    
    # Evaluation cache
    evaluation_matrix: Optional[jnp.ndarray] = None
    
    @property
    def uid(self) -> Tuple[int, int, int]:
        """Unique identifier: (iteration, birth_island, batch_index)."""
        return (self.iteration_number, self.birth_island, self.batch_index)
    
    @property
    def n_params(self) -> int:
        """Number of free parameters."""
        if self.params is None:
            return 0
        return self.params.shape[1] if self.params.ndim == 2 else len(self.params)
    
    @property
    def code_hash(self) -> str:
        """Hash of the program code string for fast exact-match detection."""
        return hashlib.md5(self.code_string.encode()).hexdigest()
    
    @property
    def behavior_hash(self) -> Optional[str]:
        """Hash of evaluation matrix for behavioral similarity detection."""
        if self.evaluation_matrix is None:
            return None
        # Quantize to reduce sensitivity to floating point differences
        quantized = np.round(np.array(self.evaluation_matrix) * 100).astype(np.int32)
        return hashlib.md5(quantized.tobytes()).hexdigest()
    
    def is_similar_to(self, other: 'Program', loss_tol: float = 0.02, cosine_tol: float = 0.95) -> bool:
        """
        Check if this program is similar to another.
        
        Args:
            other: Program to compare against.
            loss_tol: Tolerance for loss difference.
            cosine_tol: Minimum cosine similarity of evaluation matrices.
            
        Returns:
            True if programs are considered equivalent.
        """
        # Different parameter shapes = different programs
        if self.params is None or other.params is None:
            return False
        if self.params.shape != other.params.shape:
            return False
        
        # Exact match by identity
        if self.uid == other.uid:
            return True
        
        # Exact match by code
        if self.code_string == other.code_string:
            return True
        
        # Check loss difference
        if abs(self.train_loss - other.train_loss) >= loss_tol:
            return False
        
        # Check behavioral similarity via evaluation matrix
        if self.evaluation_matrix is None or other.evaluation_matrix is None:
            return False
        
        y_a = self.evaluation_matrix
        y_b = other.evaluation_matrix
        y_a_normed = y_a / jnp.linalg.norm(y_a, axis=1, keepdims=True)
        y_b_normed = y_b / jnp.linalg.norm(y_b, axis=1, keepdims=True)
        cosine_similarity = jnp.sum(y_a_normed * y_b_normed, axis=1)
        
        return float(jnp.mean(cosine_similarity)) >= cosine_tol
    
    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame creation."""
        return {
            'program_code_string': self.code_string,
            'program': self.program,
            'parameter_estimator_code_string': self.param_estimator_code_string,
            'parameter_estimator': self.param_estimator,
            'iteration_number': self.iteration_number,
            'birth_island': self.birth_island,
            'batch_index': self.batch_index,
            'train_loss': self.train_loss,
            'test_loss': self.test_loss,
            'params': self.params,
            'initial_loss': self.initial_loss,
            'initial_params': self.initial_params,
            'llm_name': self.llm_name,
            'parent1_id': self.parent1_id,
            'parent2_id': self.parent2_id,
            'evaluation_matrix': self.evaluation_matrix,
        }
    
    @classmethod
    def from_series(cls, series: pd.Series) -> 'Program':
        """Create Program from a pandas Series (DataFrame row)."""
        return cls(
            code_string=series['program_code_string'],
            program=series['program'],
            param_estimator_code_string=series['parameter_estimator_code_string'],
            param_estimator=series['parameter_estimator'],
            iteration_number=series['iteration_number'],
            birth_island=series['birth_island'],
            batch_index=series['batch_index'],
            train_loss=series['train_loss'],
            test_loss=series.get('test_loss'),
            params=series.get('params'),
            initial_loss=series.get('initial_loss'),
            initial_params=series.get('initial_params'),
            llm_name=series.get('llm_name'),
            parent1_id=series.get('parent1_id'),
            parent2_id=series.get('parent2_id'),
            evaluation_matrix=series.get('evaluation_matrix'),
        )


class ProgramRegistry:
    """
    Global registry for O(1) duplicate detection.
    
    Maintains indexes by:
    - UID (iteration, birth_island, batch_index)
    - Code hash (for exact code matches)
    - Behavior hash (for functionally equivalent programs)
    """
    
    def __init__(self):
        self._by_uid: Dict[Tuple[int, int, int], Program] = {}
        self._by_code_hash: Dict[str, List[Program]] = defaultdict(list)
        self._by_behavior_hash: Dict[str, List[Program]] = defaultdict(list)
    
    def register(self, program: Program) -> bool:
        """
        Register a program in the registry.
        
        Args:
            program: Program to register.
            
        Returns:
            True if program was registered (not a duplicate).
            False if an equivalent program already exists.
        """
        # Check UID collision
        if program.uid in self._by_uid:
            return False
        
        # Check code hash collision
        code_hash = program.code_hash
        if code_hash in self._by_code_hash:
            for existing in self._by_code_hash[code_hash]:
                if existing.code_string == program.code_string:
                    return False
        
        # Register
        self._by_uid[program.uid] = program
        self._by_code_hash[code_hash].append(program)
        
        if program.behavior_hash:
            self._by_behavior_hash[program.behavior_hash].append(program)
        
        return True
    
    def unregister(self, program: Program) -> bool:
        """Remove a program from the registry."""
        if program.uid not in self._by_uid:
            return False
        
        del self._by_uid[program.uid]
        
        code_hash = program.code_hash
        if code_hash in self._by_code_hash:
            self._by_code_hash[code_hash] = [
                p for p in self._by_code_hash[code_hash] if p.uid != program.uid
            ]
        
        if program.behavior_hash and program.behavior_hash in self._by_behavior_hash:
            self._by_behavior_hash[program.behavior_hash] = [
                p for p in self._by_behavior_hash[program.behavior_hash] if p.uid != program.uid
            ]
        
        return True
    
    def find_similar(self, program: Program, loss_tol: float = 0.02, cosine_tol: float = 0.95) -> List[Program]:
        """
        Find programs similar to the given one.
        
        Uses behavior hash for O(1) candidate lookup, then verifies similarity.
        """
        candidates = []
        
        # Check exact code matches first
        code_hash = program.code_hash
        if code_hash in self._by_code_hash:
            candidates.extend(self._by_code_hash[code_hash])
        
        # Check behavior hash matches
        if program.behavior_hash and program.behavior_hash in self._by_behavior_hash:
            for p in self._by_behavior_hash[program.behavior_hash]:
                if p.uid not in [c.uid for c in candidates]:
                    candidates.append(p)
        
        # Verify similarity
        return [p for p in candidates if p.uid != program.uid and program.is_similar_to(p, loss_tol, cosine_tol)]
    
    def has_duplicate(self, program: Program, loss_tol: float = 0.02, cosine_tol: float = 0.95) -> bool:
        """Check if an equivalent program already exists."""
        return len(self.find_similar(program, loss_tol, cosine_tol)) > 0
    
    def clear(self) -> None:
        """Clear all registered programs."""
        self._by_uid.clear()
        self._by_code_hash.clear()
        self._by_behavior_hash.clear()
    
    def __len__(self) -> int:
        return len(self._by_uid)
    
    def __contains__(self, program: Program) -> bool:
        return program.uid in self._by_uid


class Island:
    """
    Manages a population of programs with built-in genetic operations.
    
    Encapsulates:
    - Adding/removing programs
    - Deduplication
    - Pruning to capacity
    - Sampling for crossover
    - Migration (sending/receiving)
    
    Event Hooks:
    - on_program_added: Called when a program is successfully added
    - on_program_removed: Called when a program is removed
    - on_duplicates_removed: Called after deduplication with list of removed programs
    - on_pruned: Called after pruning with list of removed programs
    """
    
    def __init__(self, island_id: int, config: Optional[EvolutionConfig] = None):
        """
        Initialize an island.
        
        Args:
            island_id: Unique identifier for this island.
            config: Evolution configuration. Uses defaults if not provided.
        """
        self.island_id = island_id
        self.config = config or EvolutionConfig()
        self._programs: List[Program] = []
        self._registry = ProgramRegistry()
        
        # Event hooks
        self.on_program_added: List[ProgramEventHook] = []
        self.on_program_removed: List[ProgramEventHook] = []
        self.on_duplicates_removed: List[DeduplicateEventHook] = []
        self.on_pruned: List[PruneEventHook] = []
    
    @property
    def capacity(self) -> int: 
        """Get capacity from config. CR dkwon : update this!"""
        return self.config.capacity_per_island
    
    def add(self, program: Program, check_duplicate: bool = True) -> bool:
        """
        Add a program to the island.
        
        Args:
            program: Program to add.
            check_duplicate: If True, reject duplicates.
            
        Returns:
            True if program was added successfully.
        """
        if check_duplicate and not self._registry.register(program):
            logging.info(f"Rejected duplicate program {program.uid} on island {self.island_id}")
            return False
        
        self._programs.append(program)
        
        # Fire event hooks
        for hook in self.on_program_added:
            hook(program, self)
        
        return True
    
    def remove(self, program: Program) -> bool:
        """Remove a specific program from the island."""
        if program not in self._programs:
            return False
        
        self._programs.remove(program)
        self._registry.unregister(program)
        
        # Fire event hooks
        for hook in self.on_program_removed:
            hook(program, self)
        
        return True
    
    def remove_at(self, index: int) -> Optional[Program]:
        """Remove program at given index."""
        if index < 0 or index >= len(self._programs):
            return None
        
        program = self._programs.pop(index)
        self._registry.unregister(program)
        
        # Fire event hooks
        for hook in self.on_program_removed:
            hook(program, self)
        
        return program
    
    def remove_duplicates(self, loss_tol: Optional[float] = None, cosine_tol: Optional[float] = None, 
                          loss_type: Optional[str] = None) -> int:
        """
        Remove duplicate programs, keeping the one with lower loss.
        
        Args:
            loss_tol: Tolerance for loss comparison. Uses config default if None.
            cosine_tol: Tolerance for behavioral similarity. Uses config default if None.
            loss_type: Which loss to use ('train_loss' or 'test_loss'). Uses config default if None.
            
        Returns:
            Number of programs removed.
        """
        # Use config defaults if not specified
        loss_tol = loss_tol if loss_tol is not None else self.config.loss_tolerance
        cosine_tol = cosine_tol if cosine_tol is not None else self.config.cosine_tolerance
        loss_type = loss_type if loss_type is not None else self.config.loss_type
        
        to_remove = set()
        n = len(self._programs)
        
        for i in range(n):
            if i in to_remove:
                continue
            p_i = self._programs[i]
            
            for j in range(i + 1, n):
                if j in to_remove:
                    continue
                p_j = self._programs[j]
                
                if not p_i.is_similar_to(p_j, loss_tol, cosine_tol):
                    continue
                
                # Mark the worse one for removal
                loss_i = p_i.train_loss if loss_type == 'train_loss' else (p_i.test_loss or float('inf'))
                loss_j = p_j.train_loss if loss_type == 'train_loss' else (p_j.test_loss or float('inf'))
                
                if loss_i <= loss_j:
                    to_remove.add(j)
                else:
                    to_remove.add(i)
                    break  # p_i is marked, move to next i
        
        # Collect removed programs for event hook
        removed_programs = [self._programs[idx] for idx in sorted(to_remove, reverse=True)]
        
        # Remove in reverse order to maintain indices
        for idx in sorted(to_remove, reverse=True):
            program = self._programs.pop(idx)
            self._registry.unregister(program)
        
        # Fire event hooks
        if removed_programs:
            for hook in self.on_duplicates_removed:
                hook(removed_programs, self)
        
        return len(to_remove)
    
    def prune(self, keep_n: Optional[int] = None, large_lm_name: Optional[str] = None, 
              min_wise: Optional[int] = None) -> int:
        """
        Prune population to capacity, keeping best programs.
        
        Args:
            keep_n: Number of programs to keep. Uses config capacity if None.
            large_lm_name: Name of large LLM for wise program preservation. Uses config if None.
            min_wise: Minimum number of "wise" programs to keep. Uses config if None.
            
        Returns:
            Number of programs removed.
        """
        # Use config defaults if not specified
        keep_n = keep_n if keep_n is not None else self.config.capacity_per_island
        large_lm_name = large_lm_name if large_lm_name is not None else self.config.large_lm_name
        min_wise = min_wise if min_wise is not None else self.config.min_wise_programs
        
        if len(self._programs) <= keep_n:
            return 0
        
        # Sort by train_loss
        sorted_programs = sorted(self._programs, key=lambda p: p.train_loss)
        
        # Separate wise and regular programs
        if large_lm_name and min_wise > 0:
            wise = [p for p in sorted_programs if p.llm_name == large_lm_name]
            regular = [p for p in sorted_programs if p.llm_name != large_lm_name]
            
            # Keep top min_wise wise programs
            kept_wise = wise[:min_wise]
            remaining_slots = keep_n - len(kept_wise)
            
            # Fill remaining with best overall (excluding kept wise)
            remaining_pool = [p for p in sorted_programs if p not in kept_wise]
            kept_regular = remaining_pool[:remaining_slots]
            
            to_keep = set(p.uid for p in kept_wise + kept_regular)
        else:
            to_keep = set(p.uid for p in sorted_programs[:keep_n])
        
        # Collect removed programs for event hook
        removed_programs = [p for p in self._programs if p.uid not in to_keep]
        
        # Remove programs not in keep set
        original_count = len(self._programs)
        self._programs = [p for p in self._programs if p.uid in to_keep]
        
        # Rebuild registry
        self._registry.clear()
        for p in self._programs:
            self._registry.register(p)
        
        # Fire event hooks
        if removed_programs:
            for hook in self.on_pruned:
                hook(removed_programs, self)
        
        return original_count - len(self._programs)
    
    def sort_by_loss(self, ascending: bool = True) -> None:
        """Sort programs by train_loss."""
        self._programs.sort(key=lambda p: p.train_loss, reverse=not ascending)
    
    def sample(self, k: int, replace: bool = False) -> List[Program]:
        """
        Sample k programs from the island.
        
        Args:
            k: Number of programs to sample.
            replace: Whether to sample with replacement.
            
        Returns:
            List of sampled programs.
        """
        k = min(k, len(self._programs))
        if k == 0:
            return []
        
        indices = np.random.choice(len(self._programs), size=k, replace=replace)
        return [self._programs[i] for i in indices]
    
    def sample_by_loss(self, k: int, temperature: float = 1.0) -> List[Program]:
        """
        Sample programs with probability weighted by loss (lower loss = higher probability).
        
        Args:
            k: Number of programs to sample.
            temperature: Controls randomness. Lower = more deterministic toward best.
            
        Returns:
            List of sampled programs.
        """
        if len(self._programs) == 0:
            return []
        
        k = min(k, len(self._programs))
        
        # Compute probabilities
        losses = np.array([p.train_loss for p in self._programs])
        relative_losses = losses - losses.min()
        std = np.std(relative_losses) + 1e-6
        normalized = relative_losses / std
        
        temp = max(temperature, 1e-3)
        probs = np.exp(-normalized / temp)
        probs = probs / probs.sum()
        
        # Handle edge case where some probs might be 0
        n_nonzero = np.sum(probs > 0)
        k = min(k, n_nonzero)
        
        indices = np.random.choice(len(self._programs), size=k, replace=False, p=probs)
        return [self._programs[i] for i in indices]
    
    def get_migrants(self, n: int, temperature: float = 1.0) -> List[Program]:
        """Get programs to migrate to another island."""
        return self.sample_by_loss(n, temperature)
    
    def receive_migrants(self, migrants: List[Program], check_duplicate: bool = False) -> int:
        """
        Receive migrating programs from another island.
        
        Args:
            migrants: Programs to receive.
            check_duplicate: Whether to reject duplicates.
            
        Returns:
            Number of programs successfully added.
        """
        added = 0
        for program in migrants:
            if self.add(program, check_duplicate=check_duplicate):
                added += 1
        return added
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert island to pandas DataFrame for compatibility."""
        if not self._programs:
            return pd.DataFrame(columns=[
                'program_code_string', 'program', 'parameter_estimator_code_string', 
                'parameter_estimator', 'iteration_number', 'birth_island', 'batch_index',
                'train_loss', 'test_loss', 'params', 'initial_loss', 'initial_params',
                'llm_name', 'parent1_id', 'parent2_id', 'evaluation_matrix'
            ])
        
        return pd.DataFrame([p.to_dict() for p in self._programs])
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, island_id: int, 
                       config: Optional[EvolutionConfig] = None) -> 'Island':
        """Create Island from a pandas DataFrame."""
        island = cls(island_id=island_id, config=config)
        for _, row in df.iterrows():
            program = Program.from_series(row)
            island.add(program, check_duplicate=False)
        return island
    
    @property
    def best_program(self) -> Optional[Program]:
        """Get the program with lowest train_loss."""
        if not self._programs:
            return None
        return min(self._programs, key=lambda p: p.train_loss)
    
    @property
    def programs(self) -> List[Program]:
        """Get all programs (read-only view)."""
        return list(self._programs)
    
    def __len__(self) -> int:
        return len(self._programs)
    
    def __iter__(self) -> Iterator[Program]:
        return iter(self._programs)
    
    def __getitem__(self, index: int) -> Program:
        return self._programs[index]
    
    def __repr__(self) -> str:
        return f"Island(id={self.island_id}, population={len(self)}, capacity={self.capacity})"


class MigrationTopology:
    """Defines migration patterns between islands."""
    
    def __init__(self, explore_map: List[int], exploit_map: List[int]):
        """
        Initialize migration topology.
        
        Args:
            explore_map: Destination island for each source during exploration.
            exploit_map: Destination island for each source during exploitation.
        """
        self.explore_map = explore_map
        self.exploit_map = exploit_map
    
    def get_destination(self, source_island: int, mode: str) -> int:
        """Get destination island for migration."""
        topology = self.explore_map if mode == 'explore' else self.exploit_map
        return topology[source_island % len(topology)]
    
    @classmethod
    def ring(cls, n_islands: int) -> 'MigrationTopology':
        """Create a ring topology where each island sends to the next."""
        ring_map = [(i + 1) % n_islands for i in range(n_islands)]
        return cls(explore_map=ring_map, exploit_map=ring_map)
    
    @classmethod
    def star(cls, n_islands: int, hub: int = 0) -> 'MigrationTopology':
        """Create a star topology where all islands send to a central hub."""
        star_map = [hub] * n_islands
        return cls(explore_map=star_map, exploit_map=star_map)


class Archipelago:
    """
    Orchestrates multiple islands and their interactions.
    
    Provides high-level operations for the entire population system.
    
    Event Hooks:
    - on_migration: Called after each migration (program, source_island, dest_island)
    - on_global_prune: Called after pruning all islands
    - on_global_deduplicate: Called after deduplicating all islands
    """
    
    def __init__(self, n_islands: int, config: Optional[EvolutionConfig] = None, 
                 topology: Optional[MigrationTopology] = None):
        """
        Initialize archipelago.
        
        Args:
            n_islands: Number of islands to create.
            config: Evolution configuration. Uses defaults if not provided.
            topology: Migration topology. Defaults to ring.
        """
        self.config = config or EvolutionConfig()
        self.islands = [Island(i, self.config) for i in range(n_islands)]
        self.topology = topology or MigrationTopology.ring(n_islands)
        self.n_islands = n_islands
        
        # Event hooks
        self.on_migration: List[MigrationEventHook] = []
        self.on_global_prune: List[Callable[['Archipelago'], None]] = []
        self.on_global_deduplicate: List[Callable[['Archipelago'], None]] = []
    
    def add_program(self, island_id: int, program: Program, check_duplicate: bool = True) -> bool:
        """Add a program to a specific island."""
        if island_id < 0 or island_id >= self.n_islands:
            raise ValueError(f"Invalid island_id: {island_id}")
        return self.islands[island_id].add(program, check_duplicate)
    
    def seed_all(self, programs: List[Program]) -> None:
        """Add seed programs to all islands."""
        for island in self.islands:
            for program in programs:
                island.add(program, check_duplicate=False)
    
    def perform_migration(self, n_migrants: Optional[int] = None, mode: str = 'explore', 
                          temperature: float = 1.0) -> Dict[int, int]:
        """
        Perform migration between all islands according to topology.
        
        Args:
            n_migrants: Number of programs to migrate from each island. Uses config if None.
            mode: 'explore' or 'exploit' - determines topology.
            temperature: Controls selection randomness.
            
        Returns:
            Dict mapping destination island to number of programs received.
        """
        n_migrants = n_migrants if n_migrants is not None else self.config.n_migrants
        
        # Collect migrants from each island
        migrants_by_source = {}
        for island in self.islands:
            migrants_by_source[island.island_id] = island.get_migrants(n_migrants, temperature)
        
        # Send to destinations
        received_counts = defaultdict(int)
        for source_id, migrants in migrants_by_source.items():
            dest_id = self.topology.get_destination(source_id, mode)
            dest_id = dest_id % self.n_islands  # Ensure valid index
            
            source_island = self.islands[source_id]
            dest_island = self.islands[dest_id]
            
            for program in migrants:
                if dest_island.add(program, check_duplicate=False):
                    received_counts[dest_id] += 1
                    # Fire event hooks
                    for hook in self.on_migration:
                        hook(program, source_island, dest_island)
        
        return dict(received_counts)
    
    def deduplicate_within_islands(self, loss_tol: Optional[float] = None, 
                                   cosine_tol: Optional[float] = None) -> int:
        """Remove duplicates within each island."""
        loss_tol = loss_tol if loss_tol is not None else self.config.loss_tolerance
        cosine_tol = cosine_tol if cosine_tol is not None else self.config.cosine_tolerance
        
        total_removed = 0
        for island in self.islands:
            total_removed += island.remove_duplicates(loss_tol, cosine_tol)
        return total_removed
    
    def deduplicate_between_islands(self, overlap_threshold: Optional[int] = None, 
                                    loss_tol: Optional[float] = None, 
                                    cosine_tol: Optional[float] = None) -> int:
        """
        Remove programs that are duplicated across islands.
        
        When two islands share too many similar programs, remove duplicates from
        the island with the higher index.
        """
        overlap_threshold = overlap_threshold if overlap_threshold is not None else self.config.overlap_threshold
        loss_tol = loss_tol if loss_tol is not None else self.config.loss_tolerance
        cosine_tol = cosine_tol if cosine_tol is not None else self.config.cosine_tolerance
        
        total_removed = 0
        
        for i in range(self.n_islands):
            for j in range(i + 1, self.n_islands):
                # Find duplicates in island j that exist in island i
                duplicates_in_j = []
                for idx_j, p_j in enumerate(self.islands[j]):
                    for p_i in self.islands[i]:
                        if p_j.is_similar_to(p_i, loss_tol, cosine_tol):
                            duplicates_in_j.append(idx_j)
                            break
                
                if len(duplicates_in_j) < overlap_threshold:
                    continue
                
                # Keep at least 2 programs
                n_to_remove = min(len(duplicates_in_j), len(self.islands[j]) - 2)
                indices_to_remove = duplicates_in_j[:n_to_remove]
                
                # Remove in reverse order
                for idx in sorted(indices_to_remove, reverse=True):
                    self.islands[j].remove_at(idx)
                    total_removed += 1
                
                logging.info(f"Removed {len(indices_to_remove)} duplicates from island {j} (overlap with island {i})")
        
        return total_removed
    
    def deduplicate_all(self, overlap_threshold: Optional[int] = None, 
                        loss_tol: Optional[float] = None, 
                        cosine_tol: Optional[float] = None) -> int:
        """Perform both within-island and between-island deduplication."""
        within = self.deduplicate_within_islands(loss_tol, cosine_tol)
        between = self.deduplicate_between_islands(overlap_threshold, loss_tol, cosine_tol)
        
        # Fire event hooks
        for hook in self.on_global_deduplicate:
            hook(self)
        
        return within + between
    
    def prune_all(self, capacity: Optional[int] = None, large_lm_name: Optional[str] = None, 
                  min_wise: Optional[int] = None) -> int:
        """Prune all islands to capacity."""
        total_removed = 0
        for island in self.islands:
            total_removed += island.prune(capacity, large_lm_name, min_wise)
        
        # Fire event hooks
        for hook in self.on_global_prune:
            hook(self)
        
        return total_removed
    
    def sort_all_by_loss(self) -> None:
        """Sort all islands by train_loss."""
        for island in self.islands:
            island.sort_by_loss()
    
    def get_best_programs(self, n: int = 3) -> List[Program]:
        """Get the n best programs across all islands."""
        all_programs = []
        for island in self.islands:
            all_programs.extend(island.programs)
        
        all_programs.sort(key=lambda p: p.train_loss)
        return all_programs[:n]
    
    def get_all_programs(self) -> List[Program]:
        """Get all programs from all islands."""
        all_programs = []
        for island in self.islands:
            all_programs.extend(island.programs)
        return all_programs
    
    def to_dataframes(self) -> List[pd.DataFrame]:
        """Convert all islands to list of DataFrames for compatibility."""
        return [island.to_dataframe() for island in self.islands]
    
    def combined_dataframe(self) -> pd.DataFrame:
        """Get all programs as a single DataFrame."""
        dfs = self.to_dataframes()
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)
    
    @classmethod
    def from_dataframes(cls, dfs: List[pd.DataFrame], config: Optional[EvolutionConfig] = None, 
                        topology: Optional[MigrationTopology] = None) -> 'Archipelago':
        """Create Archipelago from list of DataFrames."""
        n_islands = len(dfs)
        archipelago = cls(n_islands=n_islands, config=config, topology=topology)
        
        for i, df in enumerate(dfs):
            archipelago.islands[i] = Island.from_dataframe(df, island_id=i, config=config)
        
        return archipelago
    
    def __len__(self) -> int:
        """Total number of programs across all islands."""
        return sum(len(island) for island in self.islands)
    
    def __getitem__(self, island_id: int) -> Island:
        return self.islands[island_id]
    
    def __iter__(self) -> Iterator[Island]:
        return iter(self.islands)
    
    def __repr__(self) -> str:
        populations = [len(island) for island in self.islands]
        return f"Archipelago(n_islands={self.n_islands}, populations={populations})"
