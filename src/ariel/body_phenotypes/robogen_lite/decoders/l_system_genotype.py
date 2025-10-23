"""Example of L-system-based decoding for modular robot graphs.

Author:     omn
Date:       2025-09-26
Py Ver:     3.12
OS:         macOS Tahoe 26
Status:     Prototype

Notes
-----
    * This decoder uses an L-system string as the genotype to generate a directed graph (DiGraph) using NetworkX.
    * The L-system rules and axiom define the growth of the modular robot structure.

References
----------
    [1] https://en.wikipedia.org/wiki/L-system
    [2] https://networkx.org/documentation/stable/reference/readwrite/generated/networkx.readwrite.json_graph.tree_data.html

"""

# Standard library

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from enum import Enum

# Third-party libraries
import matplotlib.pyplot as plt
import networkx as nx
from networkx import DiGraph
from networkx.readwrite import json_graph


# Local libraries
from ariel.body_phenotypes.robogen_lite.config import ModuleFaces, ModuleRotationsTheta, ModuleType

SEED = 42
DPI = 300

class SymbolToModuleType(Enum): # for auto-transcoding between L-system string characters and ModuleType elements
    """Enum for module types."""

    C = 'CORE'
    B = 'BRICK'
    H = 'HINGE'
    N = 'NONE'

class lsystem_element:
    def __init__(self):
        self.rotation=0
        self.front = None
        self.back = None
        self.right = None
        self.left = None
        self.top = None
        self.bottom=None
        self.allowed_connection=['TOP','BOTTOM','LEFT','RIGHT','FRONT','BACK']
        self.name=''

    def connect_to(self,side,obj,rotation):
        if side in obj.allowed_connection:
            match side:
                case 'TOP':
                    if obj.top == None:
                        self.back=obj
                        self.rotation=rotation
                        self.back.top=self
                case 'BOTTOM':
                    if obj.bottom == None:
                        self.back=obj
                        self.rotation=rotation
                        self.back.bottom=self
                case 'LEFT':
                    if obj.left == None:
                        self.back=obj
                        self.rotation=rotation
                        self.back.left=self
                case 'RIGHT':
                    if obj.right == None:
                        self.back=obj
                        self.rotation=rotation
                        self.back.right=self
                case 'FRONT':
                    if obj.front == None:
                        self.back=obj
                        self.rotation=rotation
                        self.back.front=self
                case 'BACK':
                    if obj.back == None:
                        self.back=obj
                        self.rotation=rotation
                        self.back.back=self

    def has_element(self,side):
        has_element=False
        if side in self.allowed_connection:
            match side:
                case 'TOP':
                    if self.top!=None:
                        has_element=True
                case 'BOTTOM':
                    if self.bottom!=None:
                        has_element=True
                case 'LEFT':
                    if self.left!=None:
                        has_element=True
                case 'RIGHT':
                    if self.right!=None:
                        has_element=True
                case 'FRONT':
                    if self.front!=None:
                        has_element=True
                case 'BACK':
                    if self.back!=None:
                        has_element=True
        return has_element


class lsystem_block(lsystem_element):

    def __init__(self):
        self.rotation=0
        self.front = None
        self.back = None
        self.left = None
        self.right = None
        self.top = None
        self.bottom = None
        self.allowed_connection=['TOP','BOTTOM','LEFT','RIGHT','FRONT','BACK']
        self.name='B'

class lsystem_hinge(lsystem_element):

    def __init__(self):
        self.rotation=0
        self.front = None
        self.back = None
        self.allowed_connection=['FRONT','BACK']
        self.name='H'

class lsystem_none(lsystem_element):

    def __init__(self):
        self.rotation=0
        self.back = None
        self.allowed_connection=['BACK']
        self.name='N'

class lsystem_core(lsystem_element):

    def __init__(self):
        self.rotation=0
        self.front = None
        self.back = None
        self.left = None
        self.right = None
        self.top = None
        self.bottom = None
        self.allowed_connection=['TOP','BOTTOM','LEFT','RIGHT','FRONT','BACK']
        self.name='C'

class LSystemDecoder:
    """Implements an L-system-based decoder for modular robot graphs."""

    def __init__(
        self,
        axiom: str,
        rules: Dict[str, str],
        iterations: int = 2,
    ) -> None:
        """
        Initialize the L-system decoder.
        Automatically expands the L-system and builds the graph.
        """
        self.axiom = axiom
        self.rules = rules
        self.iterations = iterations
        self.graph = nx.DiGraph()
        self.expanded_token=None
        self.structure = None

    def expand_lsystem(self):
        expanded_token = []
        expanded_token.append(self.axiom)
        it = 0
        while it<self.iterations:
            for tk in range(0,len(expanded_token)):
                if expanded_token[tk] in self.rules:
                    splitted_token = self.rules[expanded_token[tk]].split()
                    expanded_token[tk]=splitted_token[0]
                    j=1
                    while j<len(splitted_token):
                        expanded_token.insert(tk+j,splitted_token[j])
                        j+=1
                    tk+=j
            it+=1
        self.expanded_token=expanded_token

    def get_rotation(self,cmd):
        start = cmd.find("(") + 1
        end = cmd.find(")")
        rotation = cmd[start:end]
        return int(rotation)


    def build_lsystem_structure(self):
        if self.expanded_token!=None and self.expanded_token[0]=='C':
            self.structure=lsystem_core()
            turtle = self.structure
            tk=1
            while tk < len(self.expanded_token):
                print("token : ",tk)
                match self.expanded_token[tk][:4]:
                    case 'addf':
                        print("add FRONT")
                        rotation = self.get_rotation(self.expanded_token[tk])
                        if tk+1<len(self.expanded_token):
                            if self.expanded_token[tk+1] in ['H','B','N']:
                                tk+=1
                                match self.expanded_token[tk]:
                                    case 'H':
                                        print(" with ",rotation,"degree a HINGE")
                                        new_hinge = lsystem_hinge()
                                        new_hinge.connect_to('FRONT',turtle, rotation)
                                    case 'B':
                                        print(" with ",rotation,"degree a BLOCK")
                                        new_block = lsystem_block()
                                        new_block.connect_to('FRONT',turtle, rotation)
                                    case 'N':
                                        print(" with ",rotation,"degree a NONE")
                                        new_none = lsystem_none()
                                        new_none.connect_to('FRONT',turtle, rotation)
                            else:
                                print("ERROR, unsupported token following ADD, ignored")
                    case 'addk':
                        print("add BACK")
                        rotation = self.get_rotation(self.expanded_token[tk])
                        if tk+1<len(self.expanded_token):
                            if self.expanded_token[tk+1] in ['H','B','N']:
                                tk+=1
                                match self.expanded_token[tk]:
                                    case 'H':
                                        print(" with ",rotation,"degree a HINGE")
                                        new_hinge = lsystem_hinge()
                                        new_hinge.connect_to('BACK',turtle, rotation)
                                    case 'B':
                                        print(" with ",rotation,"degree a BLOCK")
                                        new_block = lsystem_block()
                                        new_block.connect_to('BACK',turtle, rotation)
                                    case 'N':
                                        print(" with ",rotation,"degree a NONE")
                                        new_none = lsystem_none()
                                        new_none.connect_to('BACK',turtle, rotation)
                            else:
                                print("ERROR, unsupported token following ADD, ignored")
                    case 'addl':
                        print("add LEFT")
                        rotation = self.get_rotation(self.expanded_token[tk])
                        if tk+1<len(self.expanded_token):
                            if self.expanded_token[tk+1] in ['H','B','N']:
                                tk+=1
                                match self.expanded_token[tk]:
                                    case 'H':
                                        print(" with ",rotation,"degree a HINGE")
                                        new_hinge = lsystem_hinge()
                                        new_hinge.connect_to('LEFT',turtle, rotation)
                                    case 'B':
                                        print(" with ",rotation,"degree a BLOCK")
                                        new_block = lsystem_block()
                                        new_block.connect_to('LEFT',turtle, rotation)
                                    case 'N':
                                        print(" with ",rotation,"degree a NONE")
                                        new_none = lsystem_none()
                                        new_none.connect_to('LEFT',turtle, rotation)
                            else:
                                print("ERROR, unsupported token following ADD, ignored")
                    case 'addr':
                        print("add RIGHT")
                        rotation = self.get_rotation(self.expanded_token[tk])
                        if tk+1<len(self.expanded_token):
                            if self.expanded_token[tk+1] in ['H','B','N']:
                                tk+=1
                                match self.expanded_token[tk]:
                                    case 'H':
                                        print(" with ",rotation,"degree a HINGE")
                                        new_hinge = lsystem_hinge()
                                        new_hinge.connect_to('RIGHT',turtle, rotation)
                                    case 'B':
                                        print(" with ",rotation,"degree a BLOCK")
                                        new_block = lsystem_block()
                                        new_block.connect_to('RIGHT',turtle, rotation)
                                    case 'N':
                                        print(" with ",rotation,"degree a NONE")
                                        new_none = lsystem_none()
                                        new_none.connect_to('RIGHT',turtle, rotation)
                            else:
                                print("ERROR, unsupported token following ADD, ignored")
                    case 'addt':
                        print("add TOP")
                        rotation = self.get_rotation(self.expanded_token[tk])
                        if tk+1<len(self.expanded_token):
                            if self.expanded_token[tk+1] in ['H','B','N']:
                                tk+=1
                                match self.expanded_token[tk]:
                                    case 'H':
                                        print(" with ",rotation,"degree a HINGE")
                                        new_hinge = lsystem_hinge()
                                        new_hinge.connect_to('TOP',turtle, rotation)
                                    case 'B':
                                        print(" with ",rotation,"degree a BLOCK")
                                        new_block = lsystem_block()
                                        new_block.connect_to('TOP',turtle, rotation)
                                    case 'N':
                                        print(" with ",rotation,"degree a NONE")
                                        new_none = lsystem_none()
                                        new_none.connect_to('TOP',turtle, rotation)
                            else:
                                print("ERROR, unsupported token following ADD, ignored")
                    case 'addb':
                        print("add BOTTOM")
                        rotation = self.get_rotation(self.expanded_token[tk])
                        if tk+1<len(self.expanded_token):
                            if self.expanded_token[tk+1] in ['H','B','N']:
                                tk+=1
                                match self.expanded_token[tk]:
                                    case 'H':
                                        print(" with ",rotation,"degree a HINGE")
                                        new_hinge = lsystem_hinge()
                                        new_hinge.connect_to('BOTTOM',turtle, rotation)
                                    case 'B':
                                        print(" with ",rotation,"degree a BLOCK")
                                        new_block = lsystem_block()
                                        new_block.connect_to('BOTTOM',turtle, rotation)
                                    case 'N':
                                        print(" with ",rotation,"degree a NONE")
                                        new_none = lsystem_none()
                                        new_none.connect_to('BOTTOM',turtle, rotation)
                            else:
                                print("ERROR, unsupported token following ADD, ignored")
                    case 'movf':
                        if turtle.has_element('FRONT')==True:
                            print("move FRONT")
                            turtle=turtle.front
                        else:
                            print("ERROR - Cannot move FRONT, ignored")
                    case 'movk':
                        if turtle.has_element('BACK')==True:
                            print("move BACK")
                            turtle=turtle.back
                        else:
                            print("ERROR - Cannot move BACK, ignored")
                    case 'movl':
                        if turtle.has_element('LEFT')==True:
                            print("move LEFT")
                            turtle=turtle.left
                        else:
                            print("ERROR - Cannot move LEFT, ignored")
                    case 'movr':
                        if turtle.has_element('RIGHT')==True:
                            print("move RIGHT")
                            turtle=turtle.right
                        else:
                            print("ERROR - Cannot move RIGHT, ignored")
                    case 'movt':
                        if turtle.has_element('TOP')==True:
                            print("move TOP")
                            turtle=turtle.top
                        else:
                            print("ERROR - Cannot move TOP, ignored")
                    case 'movb':
                        if turtle.has_element('BOTTOM')==True:
                            print("move BOTTOM")
                            turtle=turtle.bottom
                        else:
                            print("ERROR - Cannot move BOTTOM, ignored")
                    case _:
                        print("ERROR - Unknown command, ignored")
                tk+=1

    def print_lsystem_element(self,element,id):
        connection_map = []
        for j in ['FRONT','BACK','RIGHT','LEFT','TOP','BOTTOM']:
            connection_map.append(element.has_element(j))
        print(element.name,"-",id," - ",connection_map)
        id_tmp = id
        if element.has_element('FRONT')==True:
            print(element.name,"-",id," links FRONT / ",element.front.rotation," degree to ",element.front.name,"-",id_tmp+1)
            id_tmp=self.print_lsystem_element(element.front,id_tmp+1)
        if element.name=="C":
            if element.has_element('BACK')==True:
                print(element.name,"-",id," links BACK / ",element.back.rotation," degree to ",element.back.name,"-",id_tmp+1)
                id_tmp=self.print_lsystem_element(element.back,id_tmp+1)
        if element.has_element('LEFT')==True:
            print(element.name,"-",id," links LEFT / ",element.left.rotation," degree to ",element.left.name,"-",id_tmp+1)
            id_tmp=self.print_lsystem_element(element.left,id_tmp+1)
        if element.has_element('RIGHT')==True:
            print(element.name,"-",id," links RIGHT / ",element.right.rotation," degree to ",element.right.name,"-",id_tmp+1)
            id_tmp=self.print_lsystem_element(element.right,id_tmp+1)
        if element.has_element('TOP')==True:
            print(element.name,"-",id," links TOP / ",element.top.rotation," degree to ",element.top.name,"-",id_tmp+1)
            id_tmp=self.print_lsystem_element(element.top,id_tmp+1)
        if element.has_element('BOTTOM')==True:
            print(element.name,"-",id," links BOTTOM / ",element.bottom.rotation," degree to ",element.bottom.name,"-",id_tmp+1)
            id_tmp=self.print_lsystem_element(element.bottom,id_tmp+1)
        return id_tmp


    def print_lsystem_structure(self):
        self.print_lsystem_element(self.structure,0)

    def generate_lsystem_graph_element(self,element,id):
        if id==0:
            self.graph.add_node(
                element.name+"-"+str(id),
                type=ModuleType.CORE.value,
                rotation=0,
            )
        id_tmp = id
        if element.has_element('FRONT')==True:
            eltype = ModuleType.NONE.value
            match element.front.name:
                case 'B':
                    eltype = ModuleType.BRICK.value
                case 'H':
                    eltype = ModuleType.HINGE.value
            self.graph.add_node(element.front.name+"-"+str(id_tmp+1),type=eltype,rotation=element.front.rotation)
            self.graph.add_edge(element.name+"-"+str(id),element.front.name+"-"+str(id_tmp+1),face='FRONT')
            id_tmp=self.generate_lsystem_graph_element(element.front,id_tmp+1)
        if element.name=="C":
            if element.has_element('BACK')==True:
                eltype = ModuleType.NONE.value
                match element.back.name:
                    case 'B':
                        eltype = ModuleType.BRICK.value
                    case 'H':
                        eltype = ModuleType.HINGE.value
                self.graph.add_node(element.back.name+"-"+str(id_tmp+1),type=eltype,rotation=element.back.rotation)
                self.graph.add_edge(element.name+"-"+str(id),element.back.name+"-"+str(id_tmp+1),face='BACK')
                id_tmp=self.generate_lsystem_graph_element(element.back,id_tmp+1)
        if element.has_element('RIGHT')==True:
            eltype = ModuleType.NONE.value
            match element.right.name:
                case 'B':
                    eltype = ModuleType.BRICK.value
                case 'H':
                    eltype = ModuleType.HINGE.value
            self.graph.add_node(element.right.name+"-"+str(id_tmp+1),type=eltype,rotation=element.right.rotation)
            self.graph.add_edge(element.name+"-"+str(id),element.right.name+"-"+str(id_tmp+1),face='RIGHT')
            id_tmp=self.generate_lsystem_graph_element(element.right,id_tmp+1)
        if element.has_element('LEFT')==True:
            eltype = ModuleType.NONE.value
            match element.left.name:
                case 'B':
                    eltype = ModuleType.BRICK.value
                case 'H':
                    eltype = ModuleType.HINGE.value
            self.graph.add_node(element.left.name+"-"+str(id_tmp+1),type=eltype,rotation=element.left.rotation)
            self.graph.add_edge(element.name+"-"+str(id),element.left.name+"-"+str(id_tmp+1),face='LEFT')
            id_tmp=self.generate_lsystem_graph_element(element.left,id_tmp+1)
        if element.has_element('TOP')==True:
            eltype = ModuleType.NONE.value
            match element.top.name:
                case 'B':
                    eltype = ModuleType.BRICK.value
                case 'H':
                    eltype = ModuleType.HINGE.value
            self.graph.add_node(element.top.name+"-"+str(id_tmp+1),type=eltype,rotation=element.top.rotation)
            self.graph.add_edge(element.name+"-"+str(id),element.top.name+"-"+str(id_tmp+1),face='TOP')
            id_tmp=self.generate_lsystem_graph_element(element.top,id_tmp+1)
        if element.has_element('BOTTOM')==True:
            eltype = ModuleType.NONE.value
            match element.bottom.name:
                case 'B':
                    eltype = ModuleType.BRICK.value
                case 'H':
                    eltype = ModuleType.HINGE.value
            self.graph.add_node(element.bottom.name+"-"+str(id_tmp+1),type=eltype,rotation=element.bottom.rotation)
            self.graph.add_edge(element.name+"-"+str(id),element.bottom.name+"-"+str(id_tmp+1),face='BOTTOM')
            id_tmp=self.generate_lsystem_graph_element(element.bottom,id_tmp+1)
        return id_tmp


    def generate_lsystem_graph(self):
        self.graph = nx.DiGraph()
        self.generate_lsystem_graph_element(self.structure,0)

    def print_lsystem_expanded(self):
        print(self.expanded_token)

    def save_graph_as_json(self,save_file):
        if save_file is None:
            return
        data = json_graph.node_link_data(self.graph, edges="edges")
        json_string = json.dumps(data, indent=4)
        with Path(save_file).open("w", encoding="utf-8") as f:
            f.write(json_string)

    def draw_graph(self, title = "L-System Decoded Graph",save_file = None):
        """Draw the decoded graph using matplotlib and networkx."""
        plt.figure()
        pos = nx.spring_layout(self.graph, seed=SEED)
        options = {
            "with_labels": True,
            "node_size": 200,
            "node_color": "#FFFFFF00",
            "edgecolors": "blue",
            "font_size": 8,
            "width": 0.5,
        }
        nx.draw(
            self.graph,
            pos,
            with_labels=True,
            node_size=150,
            node_color="#FFFFFF00",
            edgecolors="blue",
            font_size=8,
            width=0.5,
        )
        edge_labels = nx.get_edge_attributes(self.graph, "face")
        nx.draw_networkx_edge_labels(
            self.graph,
            pos,
            edge_labels=edge_labels,
            font_color="red",
            font_size=8,
        )
        plt.title(title)
        if save_file!=None:
            plt.savefig(save_file, dpi=DPI)
        else:
            plt.show()


