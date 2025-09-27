from torch.package import Importer as Importer, OrderedImporter as OrderedImporter, PackageImporter as PackageImporter, sys_importer as sys_importer
from torch.package._package_pickler import create_pickler as create_pickler
from torch.package._package_unpickler import PackageUnpickler as PackageUnpickler
from torch.serialization import _maybe_decode_ascii as _maybe_decode_ascii

def _save_storages(importer, obj): ...
def _load_storages(id, zip_reader, obj_bytes, serialized_storages, serialized_dtypes): ...
def _get_package(zip_reader): ...

_raw_packages: dict
_deploy_objects: dict
_serialized_reduces: dict
_loaded_reduces: dict
