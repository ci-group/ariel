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

def mutate_one_point_lsystem(lsystem,mut_rate,add_temperature=0.5):
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
                    if gene_to_change-1>=0:
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
            lsystem.rules[list(rules.keys())[rule_to_change]]=lsystem.rules[list(rules.keys())[rule_to_change]]
    return op_completed

def crossover_lsystem(lsystem_parent1,lsystem_parent2,mutation_rate):
    axiom_offspring1="C"
    axiom_offspring2="C"
    rules_offspring1={}
    rules_offspring2={}
    iter_offspring1=0
    iter_offspring2=0
    if random.random()>mutation_rate:
        rules_offspring1['C']=lsystem_parent2.rules['C']
        rules_offspring2['C']=lsystem_parent1.rules['C']
        iter_offspring1+=lsystem_parent2.iterations
        iter_offspring2+=lsystem_parent1.iterations
    else:
        rules_offspring1['C']=lsystem_parent1.rules['C']
        rules_offspring2['C']=lsystem_parent2.rules['C']
        iter_offspring1+=lsystem_parent1.iterations
        iter_offspring2+=lsystem_parent2.iterations
    if random.random()>mutation_rate:
        rules_offspring1['B']=lsystem_parent2.rules['B']
        rules_offspring2['B']=lsystem_parent1.rules['B']
        iter_offspring1+=lsystem_parent2.iterations
        iter_offspring2+=lsystem_parent1.iterations
    else:
        rules_offspring1['B']=lsystem_parent1.rules['B']
        rules_offspring2['B']=lsystem_parent2.rules['B']
        iter_offspring1+=lsystem_parent1.iterations
        iter_offspring2+=lsystem_parent2.iterations
    if random.random()>mutation_rate:
        rules_offspring1['H']=lsystem_parent2.rules['H']
        rules_offspring2['H']=lsystem_parent1.rules['H']
        iter_offspring1+=lsystem_parent2.iterations
        iter_offspring2+=lsystem_parent1.iterations
    else:
        rules_offspring1['H']=lsystem_parent1.rules['H']
        rules_offspring2['H']=lsystem_parent2.rules['H']
        iter_offspring1+=lsystem_parent1.iterations
        iter_offspring2+=lsystem_parent2.iterations
    if random.random()>mutation_rate:
        rules_offspring1['N']=lsystem_parent2.rules['N']
        rules_offspring2['N']=lsystem_parent1.rules['N']
        iter_offspring1+=lsystem_parent2.iterations
        iter_offspring2+=lsystem_parent1.iterations
    else:
        rules_offspring1['N']=lsystem_parent1.rules['N']
        rules_offspring2['N']=lsystem_parent2.rules['N']
        iter_offspring1+=lsystem_parent1.iterations
        iter_offspring2+=lsystem_parent2.iterations
    iteration_offspring1=int(iter_offspring1/4)
    iteration_offspring2=int(iter_offspring2/4)
    offspring1=LSystemDecoder(axiom_offspring1,rules_offspring1,iter_offspring1)
    offspring2=LSystemDecoder(axiom_offspring2,rules_offspring2,iter_offspring2)
    return offspring1,offspring2

def initialization_lsystem():
    axiom = "C"
    rules = {}
    nb_item_C=random.choice(range(6,10))
    nb_item_H=random.choice(range(2,10))
    nb_item_B=random.choice(range(2,10))
    nb_item_N=random.choice(range(2,10))
    rule_string_C = "C"
    for i in range(0,nb_item_C):
        what_to_add = random.choice(['add','mov'])
        match what_to_add:
            case 'add':
                operator = random.choice(['addf','addk','addl','addr','addb','addt'])
                rotation = random.choice([0,45,90,135,180,225,270])
                op_to_add=operator+"("+str(rotation)+")"
                item_to_add=random.choice(['B','H','N'])
                rule_string_C+=" "+op_to_add+" "+item_to_add
            case 'mov':
                operator = random.choice(['movf','movk','movl','movr','movb','movt'])
                rule_string_C+=" "+operator
    rule_string_B="B"
    for i in range(0,nb_item_B):
        what_to_add = random.choice(['add','mov'])
        match what_to_add:
            case 'add':
                operator = random.choice(['addf','addk','addl','addr','addb','addt'])
                rotation = random.choice([0,45,90,135,180,225,270])
                op_to_add=operator+"("+str(rotation)+")"
                item_to_add=random.choice(['B','H','N'])
                rule_string_B+=" "+op_to_add+" "+item_to_add
            case 'mov':
                operator = random.choice(['movf','movk','movl','movr','movb','movt'])
                rule_string_B+=" "+operator
    rule_string_H="H"
    for i in range(0,nb_item_H):
        what_to_add = random.choice(['add','mov'])
        match what_to_add:
            case 'add':
                operator = random.choice(['addf','addk','addl','addr','addb','addt'])
                rotation = random.choice([0,45,90,135,180,225,270])
                op_to_add=operator+"("+str(rotation)+")"
                item_to_add=random.choice(['B','H','N'])
                rule_string_H+=" "+op_to_add+" "+item_to_add
            case 'mov':
                operator = random.choice(['movf','movk','movl','movr','movb','movt'])
                rule_string_H+=" "+operator
    rule_string_N="N"
    for i in range(0,nb_item_H):
        what_to_add = random.choice(['add','mov'])
        match what_to_add:
            case 'add':
                operator = random.choice(['addf','addk','addl','addr','addb','addt'])
                rotation = random.choice([0,45,90,135,180,225,270])
                op_to_add=operator+"("+str(rotation)+")"
                item_to_add=random.choice(['B','H','N'])
                rule_string_N+=" "+op_to_add+" "+item_to_add
            case 'mov':
                operator = random.choice(['movf','movk','movl','movr','movb','movt'])
                rule_string_N+=" "+operator
    rules['C']=rule_string_C
    rules['B']=rule_string_B
    rules['H']=rule_string_H
    rules['N']=rule_string_N
    iterations = random.choice(range(0,4))
    ls = LSystemDecoder(axiom,rules,iterations)
    return ls
