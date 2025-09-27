from torch.distributed.checkpoint.planner import SavePlan

__all__ = ['dedup_save_plans']

def dedup_save_plans(all_plans: list[SavePlan], save_to_lowest_rank: bool = False) -> list[SavePlan]:
    """
    Removes duplicate entries from appearing on multiple SavePlans. For each duplicate across
    a set of SavePlans, only the smallest SavePlan in terms of planned storage keeps the entry.

    Please note that this function does not modify the original SavePlans, but rather returns
    """
