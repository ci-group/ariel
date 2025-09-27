from torch._inductor.codegen.cuda.cutlass_utils import try_import_cutlass as try_import_cutlass

class CUTLASSOperationSerializer:
    """Serializes and deserializes CUTLASS GEMM operations to/from JSON.

    Handles GemmOperation objects and their nested components (TileDescription, TensorDescription).
    """
    _SUPPORTED_CLASSES: list[str]
    @classmethod
    def serialize(cls, operation: GemmOperation):
        """Serialize a GEMM operation to JSON string.

        Args:
            operation: GemmOperation object
            indent: JSON indentation spaces

        Returns:
            str: JSON representation of the operation
        """
    @classmethod
    def deserialize(cls, json_str: str) -> GemmOperation:
        """Deserialize JSON string to a GEMM operation.

        Args:
            json_str: JSON string of a GEMM operation

        Returns:
            GemmOperation: Reconstructed operation
        """
    @classmethod
    def _gemm_operation_to_json(cls, operation):
        """Convert GemmOperation to JSON-serializable dict.

        Args:
            operation: GemmOperation object

        Returns:
            dict: Dictionary representation
        """
    @classmethod
    def _json_to_gemm_operation(cls, json_dict):
        """Convert JSON dict to GemmOperation object.

        Args:
            json_dict: Dictionary representation

        Returns:
            GemmOperation: Reconstructed object
        """
    @classmethod
    def _tile_description_to_json(cls, tile_desc):
        """
        Convert TileDescription to JSON dict.

        Args:
            tile_desc: TileDescription object

        Returns:
            dict: Dictionary representation
        """
    @classmethod
    def _json_to_tile_description(cls, json_dict):
        """
        Convert JSON dict to TileDescription object.

        Args:
            json_dict: Dictionary representation

        Returns:
            TileDescription: Reconstructed object
        """
    @classmethod
    def _tensor_description_to_json(cls, tensor_desc):
        """Convert TensorDescription to JSON dict.

        Args:
            tensor_desc: TensorDescription object

        Returns:
            dict: Dictionary representation
        """
    @classmethod
    def _json_to_tensor_description(cls, tensor_json):
        """Convert JSON dict to TensorDescription object.

        Args:
            tensor_json: Dictionary representation

        Returns:
            TensorDescription: Reconstructed object
        """
    @classmethod
    def _enum_to_json(cls, enum_value):
        """Convert enum value to JSON dict.

        Args:
            enum_value: Enum value

        Returns:
            dict: Dictionary representation
        """
    @classmethod
    def _json_to_enum(cls, json_dict, enum_class):
        '''Convert JSON dict to enum value.

        Format: {name: "EnumName", value: 1}

        Args:
            json_dict: Dictionary representation
            enum_class: Target enum class

        Returns:
            Reconstructed enum value
        '''

def get_cutlass_operation_serializer() -> CUTLASSOperationSerializer | None: ...
