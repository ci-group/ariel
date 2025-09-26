from _typeshed import Incomplete
from torch.distributed.pipelining.schedules import PipelineScheduleMulti as PipelineScheduleMulti, PipelineScheduleSingle as PipelineScheduleSingle, _Action as _Action, _ComputationType as _ComputationType, _PipelineSchedule as _PipelineSchedule, get_schedule_class as get_schedule_class
from torch.distributed.pipelining.stage import PipelineStage as PipelineStage

def get_schedule_ops(schedule: str | _PipelineSchedule, pp_degree: int, num_microbatches: int, num_stages_per_rank: int | None = None) -> list[list[_Action | None]]:
    """
    Get all actions for a given schedule, pp_degree, and num_microbatches. The actions are returned in a list of lists
    where each inner list represents a rank and each element in the inner list represents an action.

    The schedule can be specified as a string which is passed into get_schedule_class() or a _PipelineSchedule instance.
    """

class _ComputationTypeColor:
    color: Incomplete
    width: Incomplete
    text: Incomplete
    def __init__(self, color: str, text: str = '', width: int = 1) -> None: ...

action_type_to_color_mapping: Incomplete

def visualize_schedule(schedule: list[list[_Action | None]], filename: str | None = None) -> None:
    """
    Visualize the schedule using matplotlib.
    The schedule is a list of lists where each inner list represents a rank and each element in the inner list represents an action.
    The actions are represented as rectangles with different colors based on their computation type.
    The filename is optional and if provided, the plot will be saved to that file.
    """
