"""
L-system genotype mutation utilities for modular robots.

Author: omn
Date: 2025-10-07
"""

import random
from typing import Tuple, Dict
import re

def crossover_lsystem(axiom1: str, rules1: Dict[str, str],axiom2: str, rules2: Dict[str, str], crossover_rate: float = 0.1):
    offspring1_axiom = axiom1
    offspring2_axiom = axiom2
    offspring1_rules = rules1.copy()
    offspring2_rules = rules2.copy()
    return offspring1_axiom, offspring1_rules, offspring2_axiom, offspring2_rules
    