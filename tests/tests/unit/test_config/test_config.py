"""Test: initialization of config classes."""

# Local libraries
from ariel.parameters import ArielConfig
from ariel.parameters.ariel_modules import SER0019, ArielModulesConfig
from ariel.parameters.mujoco_params import MujocoConfig

import pytest
from pydantic import ValidationError
from ariel.body_phenotypes.robogen_lite import config as rc


def test_ariel_config_initialization() -> None:
    """Simply instantiate the ArielConfig class."""
    ArielConfig()


def test_ariel_modules_config_initialization() -> None:
    """Simply instantiate the ArielModulesConfig class."""
    ArielModulesConfig()


def test_ser0019_initialization() -> None:
    """Simply instantiate the SER0019 class."""
    SER0019()


def test_mujoco_config_initialization() -> None:
    """Simply instantiate the MujocoConfig class."""
    MujocoConfig()


"""Test: robogen_lite config enums, constants, and validation."""
class TestModuleTypeEnum:
    """Tests for ModuleType enum."""

    def test_module_type_has_all_members(self) -> None:
        """Test ModuleType enum has expected members."""
        assert hasattr(rc.ModuleType, "CORE")
        assert hasattr(rc.ModuleType, "BRICK")
        assert hasattr(rc.ModuleType, "HINGE")
        assert hasattr(rc.ModuleType, "NONE")

    def test_module_type_values(self) -> None:
        """Test ModuleType enum values are correct."""
        assert rc.ModuleType.CORE.value == 0
        assert rc.ModuleType.BRICK.value == 1
        assert rc.ModuleType.HINGE.value == 2
        assert rc.ModuleType.NONE.value == 3

    def test_module_type_count(self) -> None:
        """Test ModuleType enum has 4 members."""
        assert len(list(rc.ModuleType)) == 4


class TestModuleFacesEnum:
    """Tests for ModuleFaces enum."""

    def test_module_faces_has_all_members(self) -> None:
        """Test ModuleFaces enum has expected members."""
        assert hasattr(rc.ModuleFaces, "FRONT")
        assert hasattr(rc.ModuleFaces, "BACK")
        assert hasattr(rc.ModuleFaces, "RIGHT")
        assert hasattr(rc.ModuleFaces, "LEFT")
        assert hasattr(rc.ModuleFaces, "TOP")
        assert hasattr(rc.ModuleFaces, "BOTTOM")

    def test_module_faces_values(self) -> None:
        """Test ModuleFaces enum values are correct."""
        assert rc.ModuleFaces.FRONT.value == 0
        assert rc.ModuleFaces.BACK.value == 1
        assert rc.ModuleFaces.RIGHT.value == 2
        assert rc.ModuleFaces.LEFT.value == 3
        assert rc.ModuleFaces.TOP.value == 4
        assert rc.ModuleFaces.BOTTOM.value == 5

    def test_module_faces_count(self) -> None:
        """Test ModuleFaces enum has 6 members."""
        assert len(list(rc.ModuleFaces)) == 6


class TestModuleRotationsIdxEnum:
    """Tests for ModuleRotationsIdx enum."""

    def test_module_rotations_idx_has_all_members(self) -> None:
        """Test ModuleRotationsIdx enum has expected members."""
        for rotation in ["DEG_0", "DEG_45", "DEG_90", "DEG_135", 
                        "DEG_180", "DEG_225", "DEG_270", "DEG_315"]:
            assert hasattr(rc.ModuleRotationsIdx, rotation)

    def test_module_rotations_idx_values(self) -> None:
        """Test ModuleRotationsIdx enum values are indices 0-7."""
        assert rc.ModuleRotationsIdx.DEG_0.value == 0
        assert rc.ModuleRotationsIdx.DEG_45.value == 1
        assert rc.ModuleRotationsIdx.DEG_90.value == 2
        assert rc.ModuleRotationsIdx.DEG_135.value == 3
        assert rc.ModuleRotationsIdx.DEG_180.value == 4
        assert rc.ModuleRotationsIdx.DEG_225.value == 5
        assert rc.ModuleRotationsIdx.DEG_270.value == 6
        assert rc.ModuleRotationsIdx.DEG_315.value == 7

    def test_module_rotations_idx_count(self) -> None:
        """Test ModuleRotationsIdx enum has 8 members."""
        assert len(list(rc.ModuleRotationsIdx)) == 8


class TestModuleRotationsThetaEnum:
    """Tests for ModuleRotationsTheta enum."""

    def test_module_rotations_theta_has_all_members(self) -> None:
        """Test ModuleRotationsTheta enum has expected members."""
        for rotation in ["DEG_0", "DEG_45", "DEG_90", "DEG_135", 
                        "DEG_180", "DEG_225", "DEG_270", "DEG_315"]:
            assert hasattr(rc.ModuleRotationsTheta, rotation)

    def test_module_rotations_theta_values_in_degrees(self) -> None:
        """Test ModuleRotationsTheta enum values are degrees."""
        assert rc.ModuleRotationsTheta.DEG_0.value == 0
        assert rc.ModuleRotationsTheta.DEG_45.value == 45
        assert rc.ModuleRotationsTheta.DEG_90.value == 90
        assert rc.ModuleRotationsTheta.DEG_135.value == 135
        assert rc.ModuleRotationsTheta.DEG_180.value == 180
        assert rc.ModuleRotationsTheta.DEG_225.value == 225
        assert rc.ModuleRotationsTheta.DEG_270.value == 270
        assert rc.ModuleRotationsTheta.DEG_315.value == 315

    def test_module_rotations_theta_count(self) -> None:
        """Test ModuleRotationsTheta enum has 8 members."""
        assert len(list(rc.ModuleRotationsTheta)) == 8


class TestModuleInstance:
    """Tests for ModuleInstance pydantic model."""

    def test_module_instance_valid_creation(self) -> None:
        """Test creating a valid ModuleInstance."""
        instance = rc.ModuleInstance(
            type=rc.ModuleType.BRICK,
            rotation=rc.ModuleRotationsIdx.DEG_45,
            links={rc.ModuleFaces.FRONT: 1, rc.ModuleFaces.BACK: 2},
        )
        assert instance.type == rc.ModuleType.BRICK
        assert instance.rotation == rc.ModuleRotationsIdx.DEG_45
        assert instance.links[rc.ModuleFaces.FRONT] == 1
        assert instance.links[rc.ModuleFaces.BACK] == 2

    def test_module_instance_all_module_types(self) -> None:
        """Test creating ModuleInstance with each module type."""
        for module_type in rc.ModuleType:
            instance = rc.ModuleInstance(
                type=module_type,
                rotation=rc.ModuleRotationsIdx.DEG_0,
                links={rc.ModuleFaces.FRONT: 0},
            )
            assert instance.type == module_type

    def test_module_instance_all_rotations(self) -> None:
        """Test creating ModuleInstance with each rotation."""
        for rotation in rc.ModuleRotationsIdx:
            instance = rc.ModuleInstance(
                type=rc.ModuleType.BRICK,
                rotation=rotation,
                links={rc.ModuleFaces.FRONT: 1},
            )
            assert instance.rotation == rotation

    def test_module_instance_empty_links(self) -> None:
        """Test creating ModuleInstance with empty links."""
        instance = rc.ModuleInstance(
            type=rc.ModuleType.CORE,
            rotation=rc.ModuleRotationsIdx.DEG_0,
            links={},
        )
        assert instance.links == {}

    def test_module_instance_all_faces_as_keys(self) -> None:
        """Test ModuleInstance can have all faces as link keys."""
        links = {face: i for i, face in enumerate(rc.ModuleFaces)}
        instance = rc.ModuleInstance(
            type=rc.ModuleType.CORE,
            rotation=rc.ModuleRotationsIdx.DEG_0,
            links=links,
        )
        assert len(instance.links) == 6
        assert instance.links[rc.ModuleFaces.TOP] == 4
        assert instance.links[rc.ModuleFaces.BOTTOM] == 5

    def test_module_instance_missing_type_raises(self) -> None:
        """Test ModuleInstance raises ValidationError if type is missing."""
        with pytest.raises(ValidationError):
            rc.ModuleInstance(
                rotation=rc.ModuleRotationsIdx.DEG_0,
                links={rc.ModuleFaces.FRONT: 1},
                # type omitted
            )

    def test_module_instance_missing_rotation_raises(self) -> None:
        """Test ModuleInstance raises ValidationError if rotation is missing."""
        with pytest.raises(ValidationError):
            rc.ModuleInstance(
                type=rc.ModuleType.BRICK,
                links={rc.ModuleFaces.FRONT: 1},
                # rotation omitted
            )

    def test_module_instance_missing_links_raises(self) -> None:
        """Test ModuleInstance raises ValidationError if links is missing."""
        with pytest.raises(ValidationError):
            rc.ModuleInstance(
                type=rc.ModuleType.BRICK,
                rotation=rc.ModuleRotationsIdx.DEG_0,
                # links omitted
            )

    def test_module_instance_invalid_type_raises(self) -> None:
        """Test ModuleInstance raises ValidationError for invalid type."""
        with pytest.raises(ValidationError):
            rc.ModuleInstance(
                type="invalid_type",  # type: ignore
                rotation=rc.ModuleRotationsIdx.DEG_0,
                links={rc.ModuleFaces.FRONT: 1},
            )

    def test_module_instance_invalid_rotation_raises(self) -> None:
        """Test ModuleInstance raises ValidationError for invalid rotation."""
        with pytest.raises(ValidationError):
            rc.ModuleInstance(
                type=rc.ModuleType.BRICK,
                rotation="invalid_rotation",  # type: ignore
                links={rc.ModuleFaces.FRONT: 1},
            )

    def test_module_instance_invalid_face_key_raises(self) -> None:
        """Test ModuleInstance raises ValidationError for invalid face key."""
        with pytest.raises(ValidationError):
            rc.ModuleInstance(
                type=rc.ModuleType.BRICK,
                rotation=rc.ModuleRotationsIdx.DEG_0,
                links={"INVALID_FACE": 1},  # type: ignore
            )

    def test_module_instance_invalid_link_value_type_raises(self) -> None:
        """Test ModuleInstance raises ValidationError for non-int link value."""
        with pytest.raises(ValidationError):
            rc.ModuleInstance(
                type=rc.ModuleType.BRICK,
                rotation=rc.ModuleRotationsIdx.DEG_0,
                links={rc.ModuleFaces.FRONT: "not_an_int"},  # type: ignore
            )

    def test_module_instance_negative_link_index(self) -> None:
        """Test ModuleInstance allows negative link indices."""
        instance = rc.ModuleInstance(
            type=rc.ModuleType.BRICK,
            rotation=rc.ModuleRotationsIdx.DEG_0,
            links={rc.ModuleFaces.FRONT: -1},
        )
        assert instance.links[rc.ModuleFaces.FRONT] == -1

    def test_module_instance_large_link_index(self) -> None:
        """Test ModuleInstance allows large link indices."""
        instance = rc.ModuleInstance(
            type=rc.ModuleType.BRICK,
            rotation=rc.ModuleRotationsIdx.DEG_0,
            links={rc.ModuleFaces.FRONT: 9999},
        )
        assert instance.links[rc.ModuleFaces.FRONT] == 9999


class TestAllowedFacesMapping:
    """Tests for ALLOWED_FACES constant."""

    def test_allowed_faces_keys_are_module_types(self) -> None:
        """Test ALLOWED_FACES keys match ModuleType enum."""
        assert set(rc.ALLOWED_FACES.keys()) == set(rc.ModuleType)

    def test_allowed_faces_values_are_lists(self) -> None:
        """Test ALLOWED_FACES values are lists of ModuleFaces."""
        for faces in rc.ALLOWED_FACES.values():
            assert isinstance(faces, list)
            assert all(isinstance(f, rc.ModuleFaces) for f in faces)

    def test_allowed_faces_core_has_all_faces(self) -> None:
        """Test CORE module type has all 6 faces."""
        core_faces = rc.ALLOWED_FACES[rc.ModuleType.CORE]
        assert len(core_faces) == 6
        assert set(core_faces) == set(rc.ModuleFaces)

    def test_allowed_faces_brick_missing_back(self) -> None:
        """Test BRICK module type does not have BACK face."""
        brick_faces = rc.ALLOWED_FACES[rc.ModuleType.BRICK]
        assert rc.ModuleFaces.BACK not in brick_faces
        assert rc.ModuleFaces.FRONT in brick_faces
        assert len(brick_faces) == 5

    def test_allowed_faces_hinge_only_front(self) -> None:
        """Test HINGE module type only has FRONT face."""
        hinge_faces = rc.ALLOWED_FACES[rc.ModuleType.HINGE]
        assert hinge_faces == [rc.ModuleFaces.FRONT]

    def test_allowed_faces_none_empty(self) -> None:
        """Test NONE module type has no faces."""
        none_faces = rc.ALLOWED_FACES[rc.ModuleType.NONE]
        assert none_faces == []

    def test_allowed_faces_no_duplicates(self) -> None:
        """Test ALLOWED_FACES lists have no duplicates."""
        for faces in rc.ALLOWED_FACES.values():
            assert len(faces) == len(set(faces))


class TestAllowedRotationsMapping:
    """Tests for ALLOWED_ROTATIONS constant."""

    def test_allowed_rotations_keys_are_module_types(self) -> None:
        """Test ALLOWED_ROTATIONS keys match ModuleType enum."""
        assert set(rc.ALLOWED_ROTATIONS.keys()) == set(rc.ModuleType)

    def test_allowed_rotations_values_are_lists(self) -> None:
        """Test ALLOWED_ROTATIONS values are lists of ModuleRotationsIdx."""
        for rotations in rc.ALLOWED_ROTATIONS.values():
            assert isinstance(rotations, list)
            assert all(isinstance(r, rc.ModuleRotationsIdx) for r in rotations)

    def test_allowed_rotations_core_only_zero(self) -> None:
        """Test CORE module type only allows 0 degree rotation."""
        core_rotations = rc.ALLOWED_ROTATIONS[rc.ModuleType.CORE]
        assert core_rotations == [rc.ModuleRotationsIdx.DEG_0]

    def test_allowed_rotations_brick_all_rotations(self) -> None:
        """Test BRICK module type allows all 8 rotations."""
        brick_rotations = rc.ALLOWED_ROTATIONS[rc.ModuleType.BRICK]
        assert len(brick_rotations) == 8
        assert set(brick_rotations) == set(rc.ModuleRotationsIdx)

    def test_allowed_rotations_hinge_all_rotations(self) -> None:
        """Test HINGE module type allows all 8 rotations."""
        hinge_rotations = rc.ALLOWED_ROTATIONS[rc.ModuleType.HINGE]
        assert len(hinge_rotations) == 8
        assert set(hinge_rotations) == set(rc.ModuleRotationsIdx)

    def test_allowed_rotations_none_only_zero(self) -> None:
        """Test NONE module type only allows 0 degree rotation."""
        none_rotations = rc.ALLOWED_ROTATIONS[rc.ModuleType.NONE]
        assert none_rotations == [rc.ModuleRotationsIdx.DEG_0]

    def test_allowed_rotations_no_duplicates(self) -> None:
        """Test ALLOWED_ROTATIONS lists have no duplicates."""
        for rotations in rc.ALLOWED_ROTATIONS.values():
            assert len(rotations) == len(set(rotations))


class TestGlobalConstants:
    """Tests for global constants."""

    def test_idx_of_core_is_zero(self) -> None:
        """Test IDX_OF_CORE is 0."""
        assert rc.IDX_OF_CORE == 0

    def test_num_of_types_of_modules(self) -> None:
        """Test NUM_OF_TYPES_OF_MODULES equals ModuleType count."""
        assert rc.NUM_OF_TYPES_OF_MODULES == 4
        assert rc.NUM_OF_TYPES_OF_MODULES == len(list(rc.ModuleType))

    def test_num_of_faces(self) -> None:
        """Test NUM_OF_FACES equals ModuleFaces count."""
        assert rc.NUM_OF_FACES == 6
        assert rc.NUM_OF_FACES == len(list(rc.ModuleFaces))

    def test_num_of_rotations(self) -> None:
        """Test NUM_OF_ROTATIONS equals ModuleRotationsIdx count."""
        assert rc.NUM_OF_ROTATIONS == 8
        assert rc.NUM_OF_ROTATIONS == len(list(rc.ModuleRotationsIdx))


class TestConfigConsistency:
    """Tests for consistency across the config module."""

    def test_allowed_faces_contains_valid_faces(self) -> None:
        """Test all faces in ALLOWED_FACES are valid ModuleFaces."""
        for faces in rc.ALLOWED_FACES.values():
            for face in faces:
                assert face in rc.ModuleFaces

    def test_allowed_rotations_contains_valid_rotations(self) -> None:
        """Test all rotations in ALLOWED_ROTATIONS are valid ModuleRotationsIdx."""
        for rotations in rc.ALLOWED_ROTATIONS.values():
            for rotation in rotations:
                assert rotation in rc.ModuleRotationsIdx

    def test_rotation_idx_and_theta_have_same_members(self) -> None:
        """Test ModuleRotationsIdx and ModuleRotationsTheta have same names."""
        idx_names = {m.name for m in rc.ModuleRotationsIdx}
        theta_names = {m.name for m in rc.ModuleRotationsTheta}
        assert idx_names == theta_names

    def test_module_instance_validates_against_allowed_faces(self) -> None:
        """Test creating ModuleInstance respects ALLOWED_FACES (no validation enforced, but documents behavior)."""
        # Note: Pydantic doesn't enforce ALLOWED_FACES constraints automatically.
        # This test documents that limitation and serves as a reminder to add
        # custom validators if needed.
        instance = rc.ModuleInstance(
            type=rc.ModuleType.CORE,
            rotation=rc.ModuleRotationsIdx.DEG_0,
            links={rc.ModuleFaces.FRONT: 1},
        )
        # Current implementation allows any face, even if not in ALLOWED_FACES
        assert instance.links[rc.ModuleFaces.FRONT] == 1