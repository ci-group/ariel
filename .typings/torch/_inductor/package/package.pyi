from _typeshed import Incomplete
from torch._inductor import config as config
from torch._inductor.cpp_builder import BuildOptionsBase as BuildOptionsBase, CppBuilder as CppBuilder
from torch.export.pt2_archive._package import AOTICompiledModel as AOTICompiledModel, AOTI_FILES as AOTI_FILES, load_pt2 as load_pt2, package_pt2 as package_pt2
from torch.types import FileLike as FileLike

log: Incomplete

def compile_so(aoti_dir: str, aoti_files: list[str], so_path: str) -> str: ...
def package_aoti(archive_file: FileLike, aoti_files: AOTI_FILES) -> FileLike:
    """
    Saves the AOTInductor generated files to the PT2Archive format.

    Args:
        archive_file: The file name to save the package to.
        aoti_files: This can either be a singular path to a directory containing
        the AOTInductor files, or a dictionary mapping the model name to the
        path to its AOTInductor generated files.
    """
def load_package(path: FileLike, model_name: str = 'model', run_single_threaded: bool = False, num_runners: int = 1, device_index: int = -1) -> AOTICompiledModel: ...
