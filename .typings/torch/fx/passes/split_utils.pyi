import torch.fx
from dataclasses import dataclass, field

__all__ = ['getattr_recursive', 'setattr_recursive', 'Component', 'split_by_tags']

def getattr_recursive(obj, name): ...
def setattr_recursive(obj, attr, value) -> None: ...

@dataclass
class Component:
    """
    A component serves as a container for a subgraph we want to create afterwards.
    """
    graph: torch.fx.Graph
    order: int
    name: str
    input_placeholders: list = field(default_factory=list)
    orig_inputs: list = field(default_factory=list)
    orig_outputs: list = field(default_factory=list)
    getattr_maps: dict[torch.fx.Node, torch.fx.Node] = field(default_factory=dict)
    constructor_args: list[str] = field(default_factory=list)
    gm: torch.fx.GraphModule | None = ...

def split_by_tags(gm: torch.fx.GraphModule, tags: list[str], return_fqn_mapping: bool = False, return_tuple: bool = False, GraphModuleCls: type[torch.fx.GraphModule] = ...) -> torch.fx.GraphModule | tuple[torch.fx.GraphModule, dict[str, str]]:
    '''
    Splits a GraphModule using tags on its graph nodes. We honor the order of
    tags. For example, we have tags = ["a", "b", "c"], the function will create
    the initial submodules in the order of "a", "b", "c".

    To set a tag:
    gm.graph.nodes[idx].tag = "mytag"

    This will result in all nodes with the same tag being extracted and placed in their
    own submodule. For placeholder, output and get_attr node, the tag is ignored. placeholder
    and output nodes are created when needed while get_attr nodes get copied to submodules
    where they are used.

    Given the following module def:

    class SimpleModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear1 = torch.nn.Linear(...)
            self.linear2 = torch.nn.Linear(...)
            self.linear3 = torch.nn.Linear(...)

        def forward(self, in1, in2):
            r1 = self.linear1(in1)
            r2 = self.linear2(in2)
            r3 = torch.cat([r1, r2])
            return self.linear3(r3)

    Marking the node corresponding to in1 with the tag sc.REQUEST_ONLY.lower() results in the following split:

    ro:
    def forward(self, in1):
        self = self.root
        linear1 = self.linear1(in1)
        return linear1

    main:
    def forward(self, in2, linear1):
        self = self.root
        linear2 = self.linear2(in2)
        cat_1 = torch.cat([linear1, linear2])
        linear3 = self.linear3(cat_1)
        return linear3

    main:
    def forward(self, in1, in2):
        self = self.root
        ro_0 = self.ro_0(in1)
        main_1 = self.main_1(in2, ro_0)
        return main_1

    Returns:
        split_gm: torch fx graph after split
        orig_to_split_fqn_mapping: a map between the original fqn and the fqn
            after split for call_module and get_attr.
    '''
