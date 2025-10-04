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
    mutated_rules = rules.copy()
    mutated_axiom = ''
    mutation = ''
    def random_gene():
        letter = random.choice(['B', 'H','N'])
        number = random.choice(allowed_numbers)
        face = random.choice(faces)
        return f"{letter}({number},{face})"
    def random_branch():
        num_genes = random.randint(1, 3)
        genes = [random_gene() for _ in range(num_genes)]
        return '[' + ''.join(genes) + ']'
    
    mod_gene=random.choice(['axiom','rules'])
    if mod_gene=='axiom':
        mutation+='Mutate axiom'
        # Only mutate one gene/branch in the axiom per call, and only if random < mutation_rate
        gene_indices = [i for i, token in enumerate(axiom_tokens) if re.fullmatch(r"([BH]\((0|90|180|270),(FRONT|LEFT|RIGHT|BACK|TOP|BOTTOM)\))|N", token)]
        modified_gene = None
        new_gene = None
        deleted_gene = None
        if gene_indices and random.random() < mutation_rate:
            i = random.choice(gene_indices)
            is_branch = False
            nb_open = 0
            for j in range(0,i+1):
                if axiom_tokens[j] =='[':
                    nb_open+=1
                elif axiom_tokens[j] ==']': 
                    nb_open-=1
            if nb_open>0:
                is_branch = True    
            token = axiom_tokens[i]
            if is_branch==True:
                op = random.choice(['add_gene', 'remove_gene', 'create_branch', 'remove_branch', 'modify_gene'])
            else:
                op = random.choice(['add_gene', 'remove_gene', 'create_branch', 'modify_gene'])    
            if op == 'add_gene':
                mutation+=' - add_gene'
                # Only add one gene
                axiom_tokens.insert(i+1, random_gene())
            elif op == 'remove_gene':
                mutation+=' - remove_gene'
                # Only remove one gene
                while i<len(axiom_tokens):
                    if axiom_tolens[i] == '[' or axiom_tokens[i] ==']':
                       i+=1
                    else: 
                          break 
                if i<len(axiom_tokens):
                    deleted_gene=axiom_tokens[i]
                    axiom_tokens[i] = ''
            elif op == 'create_branch':
                mutation+=' - create_branch'
                axiom_tokens[i] = '[' + axiom_tokens[i] + ']'
            elif op == 'remove_branch':
                mutation+=' - remove_branch'
                if i > 0:
                    j = i
                    while j>0:
                        if axiom_tokens[j] !='[':
                            break
                        j-=1
                    axiom_tokens[j]=''
                    j = i
                    while axiom_tokens[j] !=']': 
                        if j<len(axiom_tokens):
                            break
                        j+=1
                    axiom_tokens[j]=''
            elif op == 'modify_gene':
                mutation+=' - modify_gene'
                letter = random.choice(['B', 'H','N'])
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
                new_rules[new_k] = v
            mutated_rules = new_rules
        if deleted_gene:
            # Update rule keys
            new_rules = {}
            for k, v in mutated_rules.items():
                if k != deleted_gene:
                    new_rules[k] = v
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

    else:
        # Only mutate one gene/branch in the axiom per call, and only if random < mutation_rate
        mutation='Mutate rule'
        modified_gene = None
        new_gene = None
        deleted_gene = None
        if random.random() < mutation_rate:
            i = random.choice(range(0,len(mutated_rules)))
            mod_element=random.choice(['key','rule'])
            if mod_element =='key':
                mutation+=' - modify key'
                old_gene=list(mutated_rules.keys())[i]
                letter = random.choice(['B', 'H','N'])
                number = random.choice(allowed_numbers)
                face = random.choice(faces)
                new_gene = f"{letter}({number},{face})"
                mutated_rules[new_gene] = mutated_rules.pop(list(mutated_rules.keys())[i])
                mutated_rules[new_gene] = list(mutated_rules.values())[i] 
                for k in range(0,len(axiom_tokens)):
                    if axiom_tokens[k] == old_gene:
                        axiom_tokens[k] = new_gene
            else:
                mutation+=' - modify rule'
                rule=list(mutated_rules.values())[i]
                rule_tokens = [m.group(0) for m in gene_pattern.finditer(rule)]
                gene_indices = [j for j, token in enumerate(rule_tokens) if re.fullmatch(r"([BH]\((0|90|180|270),(FRONT|LEFT|RIGHT|BACK|TOP|BOTTOM)\))|N", token)]
                if gene_indices:
                    pos = random.choice(gene_indices)
                    is_branch = False
                    nb_open = 0
                    for j in range(0,pos+1):
                        if rule_tokens[j] =='[':
                            nb_open+=1
                        elif rule_tokens[j] ==']': 
                            nb_open-=1
                    if nb_open>0:
                        is_branch = True    
                    token = rule_tokens[pos]
                    if is_branch==True:
                        op = random.choice(['add_gene', 'remove_gene', 'create_branch', 'remove_branch', 'modify_gene'])
                    else:
                        op = random.choice(['add_gene', 'remove_gene', 'create_branch', 'modify_gene'])    
                    if op == 'add_gene':
                        mutation+=' - add_gene'
                        # Only add one gene
                        rule_tokens.insert(pos+1, random_gene())
                    elif op == 'remove_gene':
                        mutation+=' - remove_gene'
                        # Only remove one gene
                        while i<len(rule_tokens):
                            if rule_tokens[pos] == '[' or rule_tokens[pos] ==']':
                               pos+=1
                            else: 
                                  break 
                        if i<len(rule_tokens):
                            deleted_gene=rule_tokens[pos]
                            rule_tokens[pos] = ''
                    elif op == 'create_branch':
                        mutation+=' - create_branch'
                        rule_tokens[pos] = '[' + rule_tokens[pos] + ']'
                    elif op == 'remove_branch':
                        mutation+=' - remove_branch'
                        if pos > 0:
                            j = pos
                            while j>0:
                                if rule_tokens[j] !='[':
                                    break
                                j-=1
                            rule_tokens[j]=''
                            j = pos
                            while j<len(rule_tokens):
                                if rule_tokens[j] !=']':
                                    break
                                j+=1
                            rule_tokens[j]=''
                    elif op == 'modify_gene':
                        mutation+=' - modify_gene'
                        letter = random.choice(['B', 'H','N'])
                        number = random.choice(allowed_numbers)
                        face = random.choice(faces)
                        new_gene = f"{letter}({number},{face})"
                        rule_tokens[pos] = new_gene
                        # Update rule key
                    mutated_rules[list(mutated_rules.keys())[i]] = ''.join(rule_tokens)                
    # Ensure axiom contains exactly one C (no parameters) at the start
    mutated_axiom = ''.join(axiom_tokens)
    if not mutated_axiom.startswith('C'):
        mutated_axiom = 'C' + mutated_axiom
    mutated_axiom = 'C' + mutated_axiom.replace('C', '', 1)
    return mutated_axiom, mutated_rules, mutation

