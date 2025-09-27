from torch._dynamo import register_backend as register_backend
from torch._dynamo.utils import dynamo_timed as dynamo_timed

@register_backend
def inductor(*args, **kwargs): ...
