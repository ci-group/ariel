"""
L-system genotype random initialization utility for modular robots.

Author: omn 
Date: 2025-10-07
"""

import random
from typing import Tuple, Dict
import re

def random_lsystem():
    rules = {}
    axiom = ''
    allowed_numbers = [0, 90, 180, 270]
    for face in ['FRONT', 'LEFT', 'RIGHT', 'BACK', 'TOP', 'BOTTOM']:
        letter = random.choice(['B', 'H','N'])
        number = random.choice(allowed_numbers)
        axiom+=f"{letter}({number},{face})"
    return axiom, rules

