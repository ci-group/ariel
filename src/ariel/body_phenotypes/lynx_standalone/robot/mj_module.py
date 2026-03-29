import mujoco

class MjModule:
    """Base class for MuJoCo modules using MjSpec."""

    def __init__(self, name: str = "module", pos=None, quat=None) -> None:
        self.name = name
        # Create a fresh spec for each module
        self.spec = mujoco.MjSpec()
        # The main body of the module in its own spec
        self.body: mujoco.MjsBody = self.spec.worldbody.add_body(name=name)

        if pos is not None:
            self.body.pos = pos
        if quat is not None:
            self.body.quat = quat
        # Dictionary to store attachment sites: {site_name: MjsSite}
        self.sites: dict[str, mujoco.MjsSite] = {}

    def rotate(self, angle: float) -> None:
        """
        Rotate the module by a specified angle.
        To be implemented by subclasses if needed.
        """
