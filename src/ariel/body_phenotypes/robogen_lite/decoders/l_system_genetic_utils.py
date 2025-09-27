"""
L-system genotype mutation and crossover utilities for modular robots.

Author: omn (with help from GitHub Copilot)
Date: 2025-09-27
"""

import random
from typing import Tuple, Dict
import re

def mutate_lsystem(axiom: str, rules: Dict[str, str], mutation_rate: float = 0.1) -> Tuple[str, Dict[str, str]]:
    """
    Mutate the axiom and/or rules of an L-system genotype.
    - Randomly changes, adds, or removes symbols in the axiom.
    - Randomly mutates rule replacements.
    Returns mutated (axiom, rules).
    """
    # Tokenize axiom using regex to preserve gene and bracket structure
    gene_pattern = re.compile(r"([CBH]\((0|90|180|270),(FRONT|LEFT|RIGHT|BACK|TOP|BOTTOM)\))|N|\[|\]")
    axiom_tokens = [m.group(0) for m in gene_pattern.finditer(axiom)]
    faces = ['FRONT', 'LEFT', 'RIGHT', 'BACK', 'TOP', 'BOTTOM']
    letters = ['C', 'B', 'H', 'N']
    allowed_numbers = [0, 90, 180, 270]
    def random_gene():
        # Only generate valid gene tokens (never bare letters)
        letter = random.choice(['B', 'H'])
        number = random.choice(allowed_numbers)
        face = random.choice(faces)
        return f"{letter}({number},{face})"

    def random_branch():
        # Always at least one gene inside brackets
        num_genes = random.randint(1, 3)
        genes = [random_gene() for _ in range(num_genes)]
        return '[' + ''.join(genes) + ']'
    for i in range(len(axiom_tokens)):
        token = axiom_tokens[i]
        # Only mutate if token is a gene (not a bracket)
        if re.fullmatch(r"([CBH]\((0|90|180|270),(FRONT|LEFT|RIGHT|BACK|TOP|BOTTOM)\))|N", token):
            if random.random() < mutation_rate:
                op = random.choice(['replace', 'delete', 'insert'])
                if op == 'replace':
                    axiom_tokens[i] = random_gene()
                elif op == 'delete':
                    axiom_tokens[i] = ''
                elif op == 'insert':
                    axiom_tokens[i] += random_gene()
    # Remove any C genes from axiom tokens
    axiom_tokens = [t for t in axiom_tokens if not re.fullmatch(r"C\((0|90|180|270),(FRONT|LEFT|RIGHT|BACK|TOP|BOTTOM)\)", t)]
    mutated_axiom = ''.join(axiom_tokens)
    # Ensure axiom contains exactly one C (no parameters) at the start
    if not mutated_axiom.startswith('C'):
        mutated_axiom = 'C' + mutated_axiom
    # Remove any additional C occurrences
    mutated_axiom = 'C' + mutated_axiom.replace('C', '', 1)

    # Mutate rules: operate on rule replacement strings as token sequences
    mutated_rules = rules.copy()
    def random_rule_key():
        # Only B or H allowed in rule keys
        letter = random.choice(['B', 'H'])
        number = random.choice(allowed_numbers)
        face = random.choice(faces)
        return f"{letter}({number},{face})"
    for k in list(mutated_rules.keys()):
        # If key is not a valid gene, replace it
        if not re.fullmatch(r"([BH]\((0|90|180|270),(FRONT|LEFT|RIGHT|BACK|TOP|BOTTOM)\))|N", k):
            new_k = random_rule_key() if k != 'N' else 'N'
            mutated_rules[new_k] = mutated_rules.pop(k)
            k = new_k
        if random.random() < mutation_rate:
            op = random.choice(['replace', 'delete', 'add'])
            if op == 'replace':
                # Replace with random sequence of gene tokens and branches
                new_tokens = []
                for _ in range(random.randint(1, 5)):
                    if random.random() < 0.4:
                        new_tokens.append(random_branch())
                    else:
                        new_tokens.append(random_gene())
                rule_val = ''.join(new_tokens)
                # Post-process to ensure only valid gene tokens and branches
                valid_pattern = re.compile(r"([BH]\((0|90|180|270),(FRONT|LEFT|RIGHT|BACK|TOP|BOTTOM)\))|N|\[[^\[\]]+\]")
                valid_tokens = [m.group(0) for m in valid_pattern.finditer(rule_val)]
                # Filter out any gene tokens with invalid numbers or faces
                filtered_tokens = []
                for token in valid_tokens:
                    m = re.fullmatch(r"([BH])\((\d+),(FRONT|LEFT|RIGHT|BACK|TOP|BOTTOM)\)", token)
                    if m:
                        if int(m.group(2)) in allowed_numbers:
                            filtered_tokens.append(token)
                    elif token == 'N' or (token.startswith('[') and token.endswith(']')):
                        filtered_tokens.append(token)
                mutated_rules[k] = ''.join(filtered_tokens)
            elif op == 'delete':
                del mutated_rules[k]
            elif op == 'add':
                # Only add a branch or a gene, never concatenate gene tokens directly
                if random.random() < 0.4:
                    add_val = random_branch()
                else:
                    add_val = random_gene()
                # Post-process addition
                valid_pattern = re.compile(r"([BH]\((0|90|180|270),(FRONT|LEFT|RIGHT|BACK|TOP|BOTTOM)\))|N|\[[^\[\]]+\]")
                valid_tokens = [m.group(0) for m in valid_pattern.finditer(add_val)]
                filtered_tokens = []
                for token in valid_tokens:
                    m = re.fullmatch(r"([BH])\((\d+),(FRONT|LEFT|RIGHT|BACK|TOP|BOTTOM)\)", token)
                    if m:
                        if int(m.group(2)) in allowed_numbers:
                            filtered_tokens.append(token)
                    elif token == 'N' or (token.startswith('[') and token.endswith(']')):
                        filtered_tokens.append(token)
                mutated_rules[k] += ''.join(filtered_tokens)
    # Possibly add a new rule
    if random.random() < mutation_rate:
        new_key = random_rule_key() if random.random() < 0.75 else 'N'
        new_val = ''
        for _ in range(random.randint(1, 4)):
            if random.random() < 0.4:
                new_val += random_branch()
            else:
                new_val += random_gene()
        # Post-process to ensure only valid gene tokens and branches
        valid_pattern = re.compile(r"([BH]\((0|90|180|270),(FRONT|LEFT|RIGHT|BACK|TOP|BOTTOM)\))|N|\[[^\[\]]+\]")
        valid_tokens = [m.group(0) for m in valid_pattern.finditer(new_val)]
        filtered_tokens = []
        for token in valid_tokens:
            m = re.fullmatch(r"([BH])\((\d+),(FRONT|LEFT|RIGHT|BACK|TOP|BOTTOM)\)", token)
            if m:
                if int(m.group(2)) in allowed_numbers:
                    filtered_tokens.append(token)
            elif token == 'N' or (token.startswith('[') and token.endswith(']')):
                filtered_tokens.append(token)
        mutated_rules[new_key] = ''.join(filtered_tokens)
    return mutated_axiom, mutated_rules

def crossover_lsystem(
    axiom1: str, rules1: Dict[str, str],
    axiom2: str, rules2: Dict[str, str]
) -> Tuple[Tuple[str, Dict[str, str]], Tuple[str, Dict[str, str]]]:
    """
    Perform crossover between two L-system genotypes (axiom and rules).
    - Single-point crossover for axiom strings.
    - Rule crossover: randomly swap rule replacements between parents.
    Returns two offspring (axiom, rules) tuples.
    """
    # Tokenize axiom strings
    gene_pattern = re.compile(r"([CBH]\((0|90|180|270),(FRONT|LEFT|RIGHT|BACK|TOP|BOTTOM)\))|N|\[|\]")
    tokens1 = [m.group(0) for m in gene_pattern.finditer(axiom1)]
    tokens2 = [m.group(0) for m in gene_pattern.finditer(axiom2)]
    min_len = min(len(tokens1), len(tokens2))
    if min_len > 1:
        point = random.randint(1, min_len - 1)
    else:
        point = 1
    def ensure_single_c(axiom_tokens):
        # Remove any C genes with parameters
        axiom_tokens = [t for t in axiom_tokens if not re.fullmatch(r"C\((0|90|180|270),(FRONT|LEFT|RIGHT|BACK|TOP|BOTTOM)\)", t)]
        axiom_str = ''.join(axiom_tokens)
        # Ensure axiom contains exactly one C (no parameters) at the start
        if not axiom_str.startswith('C'):
            axiom_str = 'C' + axiom_str
        # Remove any additional C occurrences
        axiom_str = 'C' + axiom_str.replace('C', '', 1)
        return axiom_str

    child1_tokens = tokens1[:point] + tokens2[point:]
    child2_tokens = tokens2[:point] + tokens1[point:]
    child1_axiom = ensure_single_c(child1_tokens)
    child2_axiom = ensure_single_c(child2_tokens)

    # Rule crossover (unchanged)
    keys1 = set(rules1.keys())
    keys2 = set(rules2.keys())
    all_keys = list(keys1 | keys2)
    child1_rules = {}
    child2_rules = {}
    for k in all_keys:
        if k in rules1 and k in rules2:
            if random.random() < 0.5:
                child1_rules[k] = rules1[k]
                child2_rules[k] = rules2[k]
            else:
                child1_rules[k] = rules2[k]
                child2_rules[k] = rules1[k]
        elif k in rules1:
            child1_rules[k] = rules1[k]
        elif k in rules2:
            child2_rules[k] = rules2[k]
    return (child1_axiom, child1_rules), (child2_axiom, child2_rules)
