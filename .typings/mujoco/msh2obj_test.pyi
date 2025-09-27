from _typeshed import Incomplete
from absl.testing import absltest
from mujoco import msh2obj as msh2obj

_MESH_FIELDS: Incomplete
_XML: str

class MshTest(absltest.TestCase):
    def test_obj_model_matches_msh_model(self) -> None: ...
