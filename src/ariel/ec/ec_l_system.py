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
from ariel.src.ariel.ec.genotypes.lsystem.l_system_genotype import LSystemDecoder

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

def crossover_uniform_rules_lsystem(lsystem_parent1,lsystem_parent2,mutation_rate):
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
    offspring1=LSystemDecoder(axiom_offspring1,rules_offspring1,iteration_offspring1,lsystem_parent1.max_elements,lsystem_parent1.max_depth,lsystem_parent1.verbose)
    offspring2=LSystemDecoder(axiom_offspring2,rules_offspring2,iteration_offspring2,lsystem_parent2.max_elements,lsystem_parent2.max_depth,lsystem_parent2.verbose)
    return offspring1,offspring2


def crossover_uniform_genes_lsystem(lsystem_parent1,lsystem_parent2,mutation_rate):
    axiom_offspring1="C"
    axiom_offspring2="C"
    rules_offspring1={}
    rules_offspring2={}
    iter_offspring1=0
    iter_offspring2=0

    rules_parent1 = lsystem_parent1.rules["C"].split()
    rules_parent2 = lsystem_parent2.rules["C"].split()
    enh_parent1 = []
    enh_parent2 = []
    i = 0
    while i < len(rules_parent1):
        if rules_parent1[i][:4] in ['addf','addk','addl','addr','addb','addt']:
            new_token= rules_parent1[i] + " " + rules_parent1[i+1]
            enh_parent1.append(new_token)
            i+=1
        if rules_parent1[i][:4] in ['movf','movk','movl','movr','movb','movt']:
            enh_parent1.append(rules_parent1[i])
        if rules_parent1[i]=='C':
            enh_parent1.append(rules_parent1[i])
        i+=1 
    i = 0
    while i < len(rules_parent2):
        if rules_parent2[i][:4] in ['addf','addk','addl','addr','addb','addt']:
            new_token= rules_parent2[i] + " " + rules_parent2[i+1]
            enh_parent2.append(new_token)
            i+=1
        if rules_parent2[i][:4] in ['movf','movk','movl','movr','movb','movt']:
            enh_parent2.append(rules_parent2[i])
        if rules_parent2[i]=='C':
            enh_parent2.append(rules_parent2[i])
        i+=1 
    r_offspring1=""
    r_offspring2=""
    le_common = min(len(enh_parent1),len(enh_parent2))
    for i in range(0,le_common):
        if random.random()>mutation_rate:
            r_offspring1+=enh_parent2[i]+" "
            r_offspring2+=enh_parent1[i]+" "
        else:
            r_offspring1+=enh_parent1[i]+" "
            r_offspring2+=enh_parent2[i]+" "
    if len(enh_parent1)>le_common:
        for j in range(le_common,len(enh_parent1)):
            if random.random()>mutation_rate:
                r_offspring1+=enh_parent1[j]+" "
            else:   
                r_offspring2+=enh_parent1[j]+" "
    if len(enh_parent2)>le_common:
        for j in range(le_common,len(enh_parent2)):
            if random.random()>mutation_rate:
                r_offspring1+=enh_parent2[j]+" "
            else:
                r_offspring2+=enh_parent2[j]+" "
    rules_offspring1['C']=r_offspring1
    rules_offspring2['C']=r_offspring2

    rules_parent1 = lsystem_parent1.rules["B"].split()
    rules_parent2 = lsystem_parent2.rules["B"].split()
    enh_parent1 = []
    enh_parent2 = []
    i = 0
    while i < len(rules_parent1):
        if rules_parent1[i][:4] in ['addf','addk','addl','addr','addb','addt']:
            new_token= rules_parent1[i] + " " + rules_parent1[i+1]
            enh_parent1.append(new_token)
            i+=1
        if rules_parent1[i][:4] in ['movf','movk','movl','movr','movb','movt']:
            enh_parent1.append(rules_parent1[i])
        if rules_parent1[i]=='B':
            enh_parent1.append(rules_parent1[i])
        i+=1 
    i = 0
    while i < len(rules_parent2):
        if rules_parent2[i][:4] in ['addf','addk','addl','addr','addb','addt']:
            new_token= rules_parent2[i] + " " + rules_parent2[i+1]
            enh_parent2.append(new_token)
            i+=1
        if rules_parent2[i][:4] in ['movf','movk','movl','movr','movb','movt']:
            enh_parent2.append(rules_parent2[i])
        if rules_parent2[i]=='B':
            enh_parent2.append(rules_parent2[i])
        i+=1 
    r_offspring1=""
    r_offspring2=""
    le_common = min(len(enh_parent1),len(enh_parent2))
    for i in range(0,le_common):
        if random.random()>mutation_rate:
            r_offspring1+=enh_parent2[i]+" "
            r_offspring2+=enh_parent1[i]+" "
        else:
            r_offspring1+=enh_parent1[i]+" "
            r_offspring2+=enh_parent2[i]+" "
    if len(enh_parent1)>le_common:
        for j in range(le_common,len(enh_parent1)):
            if random.random()>mutation_rate:
                r_offspring1+=enh_parent1[j]+" "
            else:   
                r_offspring2+=enh_parent1[j]+" "
    if len(enh_parent2)>le_common:
        for j in range(le_common,len(enh_parent2)):
            if random.random()>mutation_rate:
                r_offspring1+=enh_parent2[j]+" "
            else:
                r_offspring2+=enh_parent2[j]+" "
    rules_offspring1['B']=r_offspring1
    rules_offspring2['B']=r_offspring2

    rules_parent1 = lsystem_parent1.rules["H"].split()
    rules_parent2 = lsystem_parent2.rules["H"].split()
    enh_parent1 = []
    enh_parent2 = []
    i = 0
    while i < len(rules_parent1):
        if rules_parent1[i][:4] in ['addf','addk','addl','addr','addb','addt']:
            new_token= rules_parent1[i] + " " + rules_parent1[i+1]
            enh_parent1.append(new_token)
            i+=1
        if rules_parent1[i][:4] in ['movf','movk','movl','movr','movb','movt']:
            enh_parent1.append(rules_parent1[i])
        if rules_parent1[i]=='H':
            enh_parent1.append(rules_parent1[i])
        i+=1 
    i = 0
    while i < len(rules_parent2):
        if rules_parent2[i][:4] in ['addf','addk','addl','addr','addb','addt']:
            new_token= rules_parent2[i] + " " + rules_parent2[i+1]
            enh_parent2.append(new_token)
            i+=1
        if rules_parent2[i][:4] in ['movf','movk','movl','movr','movb','movt']:
            enh_parent2.append(rules_parent2[i])
        if rules_parent2[i]=='H':
            enh_parent2.append(rules_parent2[i])
        i+=1 
    r_offspring1=""
    r_offspring2=""
    le_common = min(len(enh_parent1),len(enh_parent2))
    for i in range(0,le_common):
        if random.random()>mutation_rate:
            r_offspring1+=enh_parent2[i]+" "
            r_offspring2+=enh_parent1[i]+" "
        else:
            r_offspring1+=enh_parent1[i]+" "
            r_offspring2+=enh_parent2[i]+" "
    if len(enh_parent1)>le_common:
        for j in range(le_common,len(enh_parent1)):
            if random.random()>mutation_rate:
                r_offspring1+=enh_parent1[j]+" "
            else:   
                r_offspring2+=enh_parent1[j]+" "
    if len(enh_parent2)>le_common:
        for j in range(le_common,len(enh_parent2)):
            if random.random()>mutation_rate:
                r_offspring1+=enh_parent2[j]+" "
            else:
                r_offspring2+=enh_parent2[j]+" "
    rules_offspring1['H']=r_offspring1
    rules_offspring2['H']=r_offspring2

    rules_parent1 = lsystem_parent1.rules["N"].split()
    rules_parent2 = lsystem_parent2.rules["N"].split()
    enh_parent1 = []
    enh_parent2 = []
    i = 0
    while i < len(rules_parent1):
        if rules_parent1[i][:4] in ['addf','addk','addl','addr','addb','addt']:
            new_token= rules_parent1[i] + " " + rules_parent1[i+1]
            enh_parent1.append(new_token)
            i+=1
        if rules_parent1[i][:4] in ['movf','movk','movl','movr','movb','movt']:
            enh_parent1.append(rules_parent1[i])
        if rules_parent1[i]=='N':
            enh_parent1.append(rules_parent1[i])
        i+=1 
    i = 0
    while i < len(rules_parent2):
        if rules_parent2[i][:4] in ['addf','addk','addl','addr','addb','addt']:
            new_token= rules_parent2[i] + " " + rules_parent2[i+1]
            enh_parent2.append(new_token)
            i+=1
        if rules_parent2[i][:4] in ['movf','movk','movl','movr','movb','movt']:
            enh_parent2.append(rules_parent2[i])
        if rules_parent2[i]=='N':
            enh_parent2.append(rules_parent2[i])
        i+=1 
    r_offspring1=""
    r_offspring2=""
    le_common = min(len(enh_parent1),len(enh_parent2))
    for i in range(0,le_common):
        if random.random()>mutation_rate:
            r_offspring1+=enh_parent2[i]+" "
            r_offspring2+=enh_parent1[i]+" "
        else:
            r_offspring1+=enh_parent1[i]+" "
            r_offspring2+=enh_parent2[i]+" "
    if len(enh_parent1)>le_common:
        for j in range(le_common,len(enh_parent1)):
            if random.random()>mutation_rate:
                r_offspring1+=enh_parent1[j]+" "
            else:   
                r_offspring2+=enh_parent1[j]+" "
    if len(enh_parent2)>le_common:
        for j in range(le_common,len(enh_parent2)):
            if random.random()>mutation_rate:
                r_offspring1+=enh_parent2[j]+" "
            else:
                r_offspring2+=enh_parent2[j]+" "
    rules_offspring1['N']=r_offspring1
    rules_offspring2['N']=r_offspring2

    iter_offspring1+=lsystem_parent2.iterations
    iter_offspring2+=lsystem_parent1.iterations
    offspring1=LSystemDecoder(axiom_offspring1,rules_offspring1,iter_offspring1,lsystem_parent1.max_elements,lsystem_parent1.max_depth,lsystem_parent1.verbose)
    offspring2=LSystemDecoder(axiom_offspring2,rules_offspring2,iter_offspring2,lsystem_parent2.max_elements,lsystem_parent2.max_depth,lsystem_parent2.verbose)
    return offspring1,offspring2

def initialization_lsystem(max_elements=32,max_depth=8,add_temperature=0.5,none_temperature=0.2,verbose=0):
    axiom = "C"
    rules = {}
    nb_item_C=random.choice(range(6,10))
    nb_item_H=random.choice(range(2,10))
    nb_item_B=random.choice(range(2,10))
    nb_item_N=random.choice(range(2,10))
    rule_string_C = "C"
    for i in range(0,nb_item_C):
        what_to_add = random.choices(['add','mov'],weights=[add_temperature,1-add_temperature])[0]  
        match what_to_add:
            case 'add':
                operator = random.choice(['addf','addk','addl','addr','addb','addt'])
                rotation = random.choice([0,45,90,135,180,225,270])
                op_to_add=operator+"("+str(rotation)+")"
                item_to_add=random.choices(['B','H','N'],weights=[(1-none_temperature)/2,(1-none_temperature)/2,none_temperature])[0]
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
                item_to_add=random.choices(['B','H','N'],weights=[(1-none_temperature)/2,(1-none_temperature)/2,none_temperature])[0]
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
                item_to_add=random.choices(['B','H','N'],weights=[(1-none_temperature)/2,(1-none_temperature)/2,none_temperature])[0]
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
                item_to_add=random.choices(['B','H','N'],weights=[(1-none_temperature)/2,(1-none_temperature)/2,none_temperature])[0]
                rule_string_N+=" "+op_to_add+" "+item_to_add
            case 'mov':
                operator = random.choice(['movf','movk','movl','movr','movb','movt'])
                rule_string_N+=" "+operator
    rules['C']=rule_string_C
    rules['B']=rule_string_B
    rules['H']=rule_string_H
    rules['N']=rule_string_N
    iterations = random.choice(range(2,4))
    ls = LSystemDecoder(axiom,rules,iterations,max_elements=max_elements,max_depth=max_depth,verbose=verbose)
    return ls
