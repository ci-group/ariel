"""Example of L-system-based evolutionary computing algorithm for modular robot graphs.

Author:     omn
Date:       2025-09-26
Py Ver:     3.12
OS:         macOS Tahoe 26
Status:     Prototype

Notes
-----

References
----------

"""

# Standard library

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from enum import Enum
import random


# Local libraries
from ariel.body_phenotypes.robogen_lite.config import ModuleFaces, ModuleRotationsTheta, ModuleType
from ariel.body_phenotypes.robogen_lite.decoders.l_system_genotype import LSystemDecoder

SEED = 42
DPI = 300

def mutate_lsystem(lsystem,mut_rate,add_temperature=0.5):
    op_completed = ""
    if random.random()<mut_rate:
        action=random.choices(['add_rule','rm_rule'],weights=[add_temperature,1-add_temperature])[0]
        rules = lsystem.rules
        rule_to_change=random.choice(range(0,len(rules)))
        rl_tmp = list(rules.values())[rule_to_change]
        splitted_rules=rl_tmp.split()
        gene_to_change=random.choice(range(0,len(splitted_rules)))
        match action:
            case 'add_rule':
                operator=random.choice(['addf','addk','addl','addr','addb','addt','movf','movk','movl','movr','movt','movb'])
                if operator in ['addf','addk','addl','addr','addb','addt']:
                    if splitted_rules[gene_to_change][:4] in ['addf','addk','addl','addr','addb','addt']:
                        rotation = random.choice([0,45,90,135,180,225,270])
                        op_to_add=operator+"("+str(rotation)+")"
                        item_to_add=random.choice(['B','H','N'])
                        splitted_rules.insert(gene_to_change+2,item_to_add)
                        splitted_rules.insert(gene_to_change+2,op_to_add)
                        op_completed="ADDED : "+op_to_add+" "+item_to_add
                    elif splitted_rules[gene_to_change][:4] in ['movf','movk','movl','movr','movb','movt']:
                        rotation = random.choice([0,45,90,135,180,225,270])
                        op_to_add=operator+"("+str(rotation)+")"
                        item_to_add=random.choice(['B','H','N'])
                        splitted_rules.insert(gene_to_change,item_to_add)
                        splitted_rules.insert(gene_to_change,op_to_add)
                        op_completed="ADDED : "+op_to_add+" "+item_to_add
                    elif splitted_rules[gene_to_change] in ['C','B','H','N']:
                        rotation = random.choice([0,45,90,135,180,225,270])
                        op_to_add=operator+"("+str(rotation)+")"
                        item_to_add=random.choice(['B','H','N'])
                        splitted_rules.insert(gene_to_change+1,item_to_add)
                        splitted_rules.insert(gene_to_change+1,op_to_add)
                        op_completed="ADDED : "+op_to_add+" "+item_to_add
                if operator in ['movf','movk','movl','movr','movb','movt']:
                    if splitted_rules[gene_to_change][:4] in ['addf','addk','addl','addr','addb','addt']:
                        splitted_rules.insert(gene_to_change+2,operator)
                        op_completed="ADDED : "+operator
                    elif splitted_rules[gene_to_change][:4] in ['movf','movk','movl','movr','movb','movt']:
                        splitted_rules.insert(gene_to_change,operator)
                        op_completed="ADDED : "+operator
                    elif splitted_rules[gene_to_change] in ['C','B','H','N']:
                        splitted_rules.insert(gene_to_change+1,operator)
                        op_completed="ADDED : "+operator
            case 'rm_rule':
                if splitted_rules[gene_to_change][:4] in ['addf','addk','addl','addr','addb','addt']:
                    op_completed="REMOVED : "+splitted_rules[gene_to_change]+" "+splitted_rules[gene_to_change+1]
                    splitted_rules.pop(gene_to_change)
                    splitted_rules.pop(gene_to_change)
                elif splitted_rules[gene_to_change] in ['H','B','N']:
                    op_completed="REMOVED : "+splitted_rules[gene_to_change-1]+" "+splitted_rules[gene_to_change]
                    splitted_rules.pop(gene_to_change-1)
                    splitted_rules.pop(gene_to_change-1)
                elif splitted_rules[gene_to_change][:4] in ['movf','movk','movl','movr','movt','movb']:
                    op_completed="REMOVED : "+splitted_rules[gene_to_change]
                    splitted_rules.pop(gene_to_change)
        new_rule = ""
        for j in range(0,len(splitted_rules)):
            new_rule+=splitted_rules[j]+" "
        if new_rule!="":
            lsystem.rules[list(rules.keys())[rule_to_change]]=new_rule
        else:
            del lsystem.rules[list(rules.keys())[rule_to_change]]
    return op_completed
