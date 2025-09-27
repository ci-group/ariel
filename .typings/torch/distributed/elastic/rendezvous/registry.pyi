from .api import RendezvousHandler, RendezvousParameters

__all__ = ['get_rendezvous_handler']

def get_rendezvous_handler(params: RendezvousParameters) -> RendezvousHandler:
    '''
    Obtain a reference to a :py:class`RendezvousHandler`.

    Custom rendezvous handlers can be registered by

    ::

      from torch.distributed.elastic.rendezvous import rendezvous_handler_registry
      from torch.distributed.elastic.rendezvous.registry import get_rendezvous_handler


      def create_my_rdzv(params: RendezvousParameters):
          return MyCustomRdzv(params)


      rendezvous_handler_registry.register("my_rdzv_backend_name", create_my_rdzv)

      my_rdzv_handler = get_rendezvous_handler(
          "my_rdzv_backend_name", RendezvousParameters
      )
    '''
