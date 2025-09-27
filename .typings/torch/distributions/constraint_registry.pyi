from _typeshed import Incomplete

__all__ = ['ConstraintRegistry', 'biject_to', 'transform_to']

class ConstraintRegistry:
    """
    Registry to link constraints to transforms.
    """
    _registry: Incomplete
    def __init__(self) -> None: ...
    def register(self, constraint, factory=None):
        """
        Registers a :class:`~torch.distributions.constraints.Constraint`
        subclass in this registry. Usage::

            @my_registry.register(MyConstraintClass)
            def construct_transform(constraint):
                assert isinstance(constraint, MyConstraint)
                return MyTransform(constraint.arg_constraints)

        Args:
            constraint (subclass of :class:`~torch.distributions.constraints.Constraint`):
                A subclass of :class:`~torch.distributions.constraints.Constraint`, or
                a singleton object of the desired class.
            factory (Callable): A callable that inputs a constraint object and returns
                a  :class:`~torch.distributions.transforms.Transform` object.
        """
    def __call__(self, constraint):
        '''
        Looks up a transform to constrained space, given a constraint object.
        Usage::

            constraint = Normal.arg_constraints["scale"]
            scale = transform_to(constraint)(torch.zeros(1))  # constrained
            u = transform_to(constraint).inv(scale)  # unconstrained

        Args:
            constraint (:class:`~torch.distributions.constraints.Constraint`):
                A constraint object.

        Returns:
            A :class:`~torch.distributions.transforms.Transform` object.

        Raises:
            `NotImplementedError` if no transform has been registered.
        '''

biject_to: Incomplete
transform_to: Incomplete
