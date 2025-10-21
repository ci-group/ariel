"""
L-system genotype mutation utilities for modular robots.

Author: omn
Date: 2025-10-07
"""

import random
from typing import Tuple, Dict
import re

def crossover_one_point_lsystem(axiom1: str, rules1: Dict[str, str],axiom2: str, rules2: Dict[str, str], crossover_rate: float = 0.3):
    crossover_len = int(1+(6*crossover_rate/2))
    crossover_point = random.randint(0,6)
    if crossover_len+crossover_point>=6:
        crossover_point = 6-crossover_len-1
    gene_pattern = re.compile(r"([CBHN]\((0|90|180|270),(FRONT|LEFT|RIGHT|BACK|TOP|BOTTOM)\))||\[|\]|C")
    axiom_tokens_tmp = [m.group(0) for m in gene_pattern.finditer(axiom1)]
    axiom_tokens1 = [token for i, token in enumerate(axiom_tokens_tmp) if token not in ['[',']','C','']]
    axiom_tokens_tmp = [m.group(0) for m in gene_pattern.finditer(axiom2)]
    axiom_tokens2 = [token for i, token in enumerate(axiom_tokens_tmp) if token not in ['[',']','C','']]
    print("crossover position:",crossover_point)
    print("crossover length:",crossover_len)
    offspring1_axiom = 'C'
    offspring2_axiom = 'C'
    rules1_keys = list(rules1.keys())
    rules2_keys = list(rules2.keys())
    rules1_values = list(rules1.values())
    rules2_values = list(rules2.values())
    offspring1_rules = {}
    offspring2_rules = {}
    offspring1_axiom_token = []
    offspring2_axiom_token = []
    for i in range(0,crossover_point):
        offspring1_axiom_token.append(axiom_tokens1[i])
        offspring2_axiom_token.append(axiom_tokens2[i])
        for j in range(0,len(rules1_keys)):
            if axiom_tokens1[i]==rules1_keys[j]:
                offspring1_rules[rules1_keys[j]]=rules1_values[j]
        for j in range(0,len(rules2_keys)):
            if axiom_tokens2[i]==rules2_keys[j]:
                offspring2_rules[rules2_keys[j]]=rules2_values[j]
    for i in range(crossover_point,crossover_point+crossover_len):
        offspring1_axiom_token.append(axiom_tokens2[i])
        offspring2_axiom_token.append(axiom_tokens1[i])
        for j in range(0,len(rules1_keys)):
            if axiom_tokens1[i]==rules1_keys[j]:
                offspring2_rules[rules1_keys[j]]=rules1_values[j]
        for j in range(0,len(rules2_keys)):
            if axiom_tokens2[i]==rules2_keys[j]:
                offspring1_rules[rules2_keys[j]]=rules2_values[j]
    for i in range(crossover_point+crossover_len,len(axiom_tokens1)):
        offspring1_axiom_token.append(axiom_tokens1[i])
        offspring2_axiom_token.append(axiom_tokens2[i])
        for j in range(0,len(rules1_keys)):
            if axiom_tokens1[i]==rules1_keys[j]:
                offspring1_rules[rules1_keys[j]]=rules1_values[j]
        for j in range(0,len(rules2_keys)):
            if axiom_tokens1[i]==rules2_keys[j]:
                offspring2_rules[rules2_keys[j]]=rules2_values[j]

    for i in range(0,len(offspring1_axiom_token)):
        offspring1_axiom+="["+offspring1_axiom_token[i]+"]"
    for i in range(0,len(offspring2_axiom_token)):
        offspring2_axiom+="["+offspring2_axiom_token[i]+"]"

    rules1_not_assigned = [rule for rule in rules1_keys if rule not in offspring1_rules.keys()]
    rules2_not_assigned = [rule for rule in rules2_keys if rule not in offspring2_rules.keys()]
    for i in range(0,len(rules1_not_assigned)):
        if random.random()<0.5:
            offspring1_rules[rules1_not_assigned[i]]=rules1[rules1_not_assigned[i]]
        else:
            offspring2_rules[rules1_not_assigned[i]]=rules1[rules1_not_assigned[i]]
    for i in range(0,len(rules2_not_assigned)):
        if random.random()<0.5:
            offspring2_rules[rules2_not_assigned[i]]=rules2[rules2_not_assigned[i]]
        else:
            offspring1_rules[rules2_not_assigned[i]]=rules2[rules2_not_assigned[i]]
    return offspring1_axiom, offspring1_rules, offspring2_axiom, offspring2_rules
