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
    gene_pattern = re.compile(r"([CBH]\((0|90|180|270),(FRONT|LEFT|RIGHT|BACK|TOP|BOTTOM)\))|N|\[|\]|C")
    axiom_tokens = [m.group(0) for m in gene_pattern.finditer(axiom)]
    faces = ['FRONT', 'LEFT', 'RIGHT', 'BACK', 'TOP', 'BOTTOM']
    letters = ['C', 'B', 'H', 'N']
    allowed_numbers = [0, 90, 180, 270]
    def random_gene():
        letter = random.choice(['B', 'H'])
        number = random.choice(allowed_numbers)
        face = random.choice(faces)
        return f"{letter}({number},{face})"
    def random_branch():
        num_genes = random.randint(1, 3)
        genes = [random_gene() for _ in range(num_genes)]
        return '[' + ''.join(genes) + ']'
    # Only mutate one gene/branch in the axiom per call, and only if random < mutation_rate
    gene_indices = [i for i, token in enumerate(axiom_tokens) if re.fullmatch(r"([BH]\((0|90|180|270),(FRONT|LEFT|RIGHT|BACK|TOP|BOTTOM)\))|N", token)]
    modified_gene = None
    new_gene = None
    if gene_indices and random.random() < mutation_rate:
        i = random.choice(gene_indices)
        token = axiom_tokens[i]
        op = random.choice(['add_gene', 'remove_gene', 'create_branch', 'remove_branch', 'modify_gene'])
        if op == 'add_gene':
            # Only add one gene
            axiom_tokens.insert(i+1, random_gene())
        elif op == 'remove_gene':
            # Only remove one gene
            axiom_tokens[i] = ''
        elif op == 'create_branch':
            axiom_tokens[i] = '[' + axiom_tokens[i] + ']'
        elif op == 'remove_branch':
            if i > 0 and axiom_tokens[i-1] == '[' and i+1 < len(axiom_tokens) and axiom_tokens[i+1] == ']':
                axiom_tokens[i-1] = ''
                axiom_tokens[i+1] = ''
        elif op == 'modify_gene':
            letter = token[0]
            number = random.choice(allowed_numbers)
            face = random.choice(faces)
            new_gene = f"{letter}({number},{face})"
            modified_gene = token
            axiom_tokens[i] = new_gene
    # Mutate rules: operate on rule replacement strings as token sequences
    mutated_rules = rules.copy()
    # If a gene was modified in the axiom, update all rules and rule values
    if modified_gene and new_gene:
        # Update rule keys
        new_rules = {}
        for k, v in mutated_rules.items():
            new_k = new_gene if k == modified_gene else k
            # Update all occurrences in rule values
            new_v = v.replace(modified_gene, new_gene)
            new_rules[new_k] = new_v
        mutated_rules = new_rules
    # Remove any C genes with parameters from axiom tokens
    axiom_tokens = [t for t in axiom_tokens if not re.fullmatch(r"C\((0|90|180|270),(FRONT|LEFT|RIGHT|BACK|TOP|BOTTOM)\)", t)]

    # Remove empty branches: if a gene is removed and is the only one inside [ ], remove the brackets too
    i = 0
    while i < len(axiom_tokens):
        if axiom_tokens[i] == '[':
            # Find the matching closing bracket
            j = i + 1
            while j < len(axiom_tokens) and axiom_tokens[j] != ']':
                j += 1
            # If only one non-empty token inside, and it's empty, remove the brackets
            inside = [t for t in axiom_tokens[i+1:j] if t.strip() != '']
            if len(inside) == 0 and j < len(axiom_tokens):
                axiom_tokens[i] = ''
                axiom_tokens[j] = ''
                # Optionally, remove all empty tokens between i and j
                for k in range(i+1, j):
                    axiom_tokens[k] = ''
                i = j
        i += 1

    # Ensure axiom contains exactly one C (no parameters) at the start
    mutated_axiom = ''.join(axiom_tokens)
    if not mutated_axiom.startswith('C'):
        mutated_axiom = 'C' + mutated_axiom
    mutated_axiom = 'C' + mutated_axiom.replace('C', '', 1)

    # Mutate rules: operate on rule replacement strings as token sequences
    mutated_rules = rules.copy()
    def random_rule_key():
        # Only B or H allowed in rule keys
        letter = random.choice(['B', 'H'])
        number = random.choice(allowed_numbers)
        face = random.choice(faces)
        return f"{letter}({number},{face})"
    # Only mutate one rule per call, and only if random < mutation_rate
    rule_keys = list(mutated_rules.keys())
    modified_rule_gene = None
    new_rule_gene = None
    if rule_keys and random.random() < mutation_rate:
        k = random.choice(rule_keys)
        # If key is not a valid gene, replace it
        if not re.fullmatch(r"([BH]\((0|90|180|270),(FRONT|LEFT|RIGHT|BACK|TOP|BOTTOM)\))|N", k):
            new_k = random_rule_key() if k != 'N' else 'N'
            mutated_rules[new_k] = mutated_rules.pop(k)
            k = new_k
        op = random.choice(['replace', 'delete', 'add', 'modify_gene'])
        if op == 'replace':
            # Only replace with one gene or one branch
            if random.random() < 0.4:
                rule_val = random_branch()
            else:
                rule_val = random_gene()
            valid_pattern = re.compile(r"([BH]\((0|90|180|270),(FRONT|LEFT|RIGHT|BACK|TOP|BOTTOM)\))|N|\[[^\[\]]+\]")
            valid_tokens = [m.group(0) for m in valid_pattern.finditer(rule_val)]
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
            # Only add one gene or one branch
            if random.random() < 0.4:
                add_val = random_branch()
            else:
                add_val = random_gene()
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
        elif op == 'modify_gene':
            # Only allow modify_gene if k is a gene
            if re.fullmatch(r"([BH]\((0|90|180|270),(FRONT|LEFT|RIGHT|BACK|TOP|BOTTOM)\))", k):
                letter = k[0]
                number = random.choice(allowed_numbers)
                face = random.choice(faces)
                new_rule_gene = f"{letter}({number},{face})"
                modified_rule_gene = k
                # Update rule key
                mutated_rules[new_rule_gene] = mutated_rules.pop(k)
    # If a gene was modified in the rules, update all occurrences in axiom and rule values
    if modified_rule_gene and new_rule_gene:
        # Update axiom
        axiom_tokens = [new_rule_gene if t == modified_rule_gene else t for t in axiom_tokens]
        # Update all rule values
        for rk in mutated_rules:
            mutated_rules[rk] = mutated_rules[rk].replace(modified_rule_gene, new_rule_gene)
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
    # Remove rules whose key is not present in the mutated axiom or in any rule value
    gene_token_pattern = re.compile(r"([BH]\((0|90|180|270),(FRONT|LEFT|RIGHT|BACK|TOP|BOTTOM)\))|N")
    # Find all gene tokens in axiom
    present_genes = set(m.group(0) for m in gene_token_pattern.finditer(mutated_axiom))
    # Find all gene tokens in rule values
    for v in mutated_rules.values():
        present_genes.update(m.group(0) for m in gene_token_pattern.finditer(v))
    # Remove rules whose key is not present
    mutated_rules = {k: v for k, v in mutated_rules.items() if k in present_genes}
    return mutated_axiom, mutated_rules

