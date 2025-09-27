from torch.distributed.checkpoint.planner import SavePlan

__all__ = ['dedup_tensors']

def dedup_tensors(all_plans: list[SavePlan]) -> list[SavePlan]: ...
