import subprocess
import os
import platform
import shutil

base_dir = os.path.dirname(os.path.abspath(__file__))


def create_venv():
    print("Creating the virtual environment...")
    subprocess.run(["python", "-m", "venv", "venv"], check=True)


def activate_venv():
    print("Activating the virtual environment...")
    if platform.system() == "Windows":
        os.system(".\\venv\\Scripts\\activate")
    else:
        os.system("source ./venv/bin/activate")


def install_requirements():
    print("Installing the requirements...")
    if platform.system() == "Windows":
        subprocess.run([".\\venv\\Scripts\\pip3", "install",
                       "--pre", "-r", "requirements.txt"], check=True)
    else:
        subprocess.run(["./venv/bin/pip3", "install", "--pre",
                       "-r", "requirements.txt"], check=True)


def install_pyinstaller():
    print("Installing PyInstaller...")
    if platform.system() == "Windows":
        subprocess.run([".\\venv\\Scripts\\python", "-m", "pip",
                       "install", "pyinstaller"], check=True)
    else:
        subprocess.run(["./venv/bin/python", "-m", "pip",
                       "install", "pyinstaller"], check=True)


def create_executable():
    print("Creating executable with PyInstaller...")
    src_path = os.path.join(base_dir, "src")
    jsx_path = os.path.join(base_dir, "TheAnimeScripter.jsx")
    main_path = os.path.join(base_dir, "main.py")

    """
    # This should fix CUPY CUDA freezing errors but I may have missed some import
    
    Failed to import CuPy.

    If you installed CuPy via wheels (cupy-cudaXXX or cupy-rocm-X-X), make sure that the package matches with the version of CUDA or ROCm installed.

    On Linux, you may need to set LD_LIBRARY_PATH environment variable depending on how you installed CUDA/ROCm.
    On Windows, try setting CUDA_PATH environment variable.

    Check the Installation Guide for details:
    https://docs.cupy.dev/en/latest/install.html

    Original error:
    ModuleNotFoundError: No module named 'cupy_backends.cuda.api._runtime_enum
    
    hiddenimports = ['cupy._environment', 'cupy._version', 'cupy_backends.cuda', 'cupy_backends.cuda.api', 'cupy_backends.cuda.api._runtime_enum', 'cupy_backends.cuda.api.runtime', 'cupy._core.syncdetect', 'cupy_backends.cuda.libs', 'cupy_backends.cuda.stream', 'cupy_backends.cuda.libs.cublas', 'cupy_backends.cuda.libs.cusolver', 'cupy_backends.cuda._softlink', 'cupy_backends.cuda.libs.cusparse', 'cupy._util', 'cupy.cuda.device', 'cupy.cuda.memory_hook', 'cupy.cuda.graph', 'cupy.cuda.stream', 'cupy_backends.cuda.api._driver_enum', 'cupy_backends.cuda.api.driver', 'cupy.cuda.memory', 'cupy._core.internal', 'cupy._core._carray', 'cupy.cuda.texture', 'cupy.cuda.function', 'cupy_backends.cuda.libs.nvrtc', 'cupy.cuda.jitify', 'cupy.cuda.compiler', 'cupy.cuda.memory_hooks.debug_print', 'cupy.cuda.memory_hooks.line_profile', 'cupy.cuda.memory_hooks', 'cupy.cuda.pinned_memory', 'cupy_backends.cuda.libs.curand', 'cupy_backends.cuda.libs.profiler', 'cupy.cuda.common', 'cupy.cuda.cub', 'cupy_backends.cuda.libs.nvtx', 'cupy.cuda.thrust', 'cupy.cuda', 'cupy._core._dtype', 'cupy._core._scalar', 'cupy._core._accelerator', 'cupy._core._memory_range', 'cupy._core._fusion_thread_local', 'cupy._core._kernel', 'cupy._core._ufuncs', 'cupy._core._routines_manipulation', 'cupy._core._routines_binary', 'cupy._core._optimize_config', 'cupy._core._cub_reduction', 'cupy._core._reduction', 'cupy._core._routines_math', 'cupy._core._routines_indexing', 'cupy._core._routines_linalg', 'cupy._core._routines_logic', 'cupy._core._routines_sorting', 'cupy._core._routines_statistics', 'cupy._core.dlpack', 'cupy._core.flags', 'cupy._core.core', 'cupy._core._fusion_interface', 'cupy._core._fusion_variable', 'cupy._core._codeblock', 'cupy._core._fusion_op', 'cupy._core._fusion_optimization', 'cupy._core._fusion_trace', 'cupy._core._fusion_kernel', 'cupy._core.new_fusion', 'cupy._core.fusion', 'cupy._core.raw', 'cupy._core', 'cupyx._rsqrt', 'cupyx._runtime', 'cupyx._scatter', 'cupy._core._gufuncs', 'cupyx.cusolver', 'cupy.linalg._util', 'cupy.linalg._decomposition', 'cupy.cublas', 'cupy.linalg._solve', 'cupy.linalg._product', 'cupy.linalg._eigenvalue', 'cupy.linalg._norms', 'cupy.linalg', 'cupyx.scipy.sparse._util', 'cupyx.scipy.sparse._sputils', 'cupyx.scipy.sparse._base', 'cupyx.cusparse', 'cupy._creation', 'cupy._creation.basic', 'cupyx.scipy.sparse._data', 'cupyx.scipy.sparse._index', 'cupyx.scipy.sparse._compressed', 'cupyx.scipy.sparse._csc', 'cupyx.scipy.sparse._csr', 'cupyx.scipy.sparse._coo', 'cupyx.scipy.sparse._dia', 'cupyx.scipy.sparse._construct', 'cupyx.scipy.sparse._extract', 'cupyx.scipy.sparse', 'cupyx.scipy', 'cupyx.linalg.sparse._solve', 'cupyx.linalg.sparse', 'cupyx.lapack', 'cupyx.linalg._solve', 'cupyx.linalg', 'cupyx.profiler._time', 'cupyx.profiler._time_range', 'cupyx.profiler', 'cupyx.time', 'cupyx.optimizing._optimize', 'cupyx.optimizing', 'cupyx._ufunc_config', 'cupyx._pinned_array', 'cupyx._gufunc', 'cupy.cuda.cufft', 'cupy.fft._cache', 'cupy.fft.config', 'cupy.fft._fft', 'cupy.fft', 'cupy.polynomial.polynomial', 'cupy.polynomial.polyutils', 'cupy.polynomial', 'cupy.random._kernels', 'cupy.random._generator', 'cupy.random._distributions', 'cupy.random._permutations', 'cupy.random._sample', 'cupy.random._generator_api',
                     'cupy.random._bit_generator', 'cupy.random', 'cupy.sparse', 'cupy.testing._array', 'cupy.testing._bundle', 'cupy.testing._parameterized', 'cupy.testing._pytest_impl', 'cupy.testing._attr', 'cupy.testing._helper', 'cupy.testing._loops', 'cupy.testing._random', 'cupy.testing', 'cupy._creation.from_data', 'cupy._creation.ranges', 'cupy._creation.matrix', 'cupy._functional', 'cupy._functional.piecewise', 'cupy._logic', 'cupy._logic.ops', 'cupy._math', 'cupy._math.arithmetic', 'cupy._logic.content', 'cupy._logic.comparison', 'cupy._binary', 'cupy._binary.elementwise', 'cupyx.jit._cuda_typerules', 'cupyx.jit._internal_types', 'cupyx.jit._cuda_types', 'cupyx.jit._builtin_funcs', 'cupyx.jit._compile', 'cupyx.jit._interface', 'cupyx.jit.cg', 'cupyx.jit.cub', 'cupyx.jit.thrust', 'cupyx.jit', 'cupy._functional.vectorize', 'cupy.lib.stride_tricks', 'cupy.lib', 'cupy.lib._shape_base', 'cupy._manipulation', 'cupy._sorting', 'cupy._sorting.search', 'cupy._manipulation.basic', 'cupy._manipulation.shape', 'cupy._manipulation.transpose', 'cupy._manipulation.dims', 'cupy._manipulation.join', 'cupy._manipulation.kind', 'cupy._manipulation.split', 'cupy._manipulation.tiling', 'cupy._manipulation.add_remove', 'cupy._manipulation.rearrange', 'cupy._binary.packing', 'cupy._indexing', 'cupy._indexing.generate', 'cupy._indexing.indexing', 'cupy._indexing.insert', 'cupy._indexing.iterate', 'cupy._io', 'cupy._io.npz', 'cupy._io.formatting', 'cupy._io.text', 'cupy.linalg._einsum_opt', 'cupy.linalg._einsum_cutn', 'cupy.linalg._einsum', 'cupy._logic.truth', 'cupy._logic.type_testing', 'cupyx.scipy.fft._fft', 'cupyx.scipy.special._complexstuff', 'cupyx.scipy.special._trig', 'cupyx.scipy.special._loggamma', 'cupyx.scipy.special._gamma', 'cupyx.scipy.special._bessel', 'cupyx.scipy.special._digamma', 'cupyx.scipy.special._zeta', 'cupyx.scipy.special._gammainc', 'cupyx.scipy.special._beta', 'cupyx.scipy.special._stats_distributions', 'cupyx.scipy.special._statistics', 'cupyx.scipy.special._convex_analysis', 'cupyx.scipy.special._gammaln', 'cupyx.scipy.special._gammasgn', 'cupyx.scipy.special._polygamma', 'cupyx.scipy.special._poch', 'cupyx.scipy.special._erf', 'cupyx.scipy.special._lpmv', 'cupyx.scipy.special._sph_harm', 'cupyx.scipy.special._exp1', 'cupyx.scipy.special._expi', 'cupyx.scipy.special._expn', 'cupyx.scipy.special._softmax', 'cupyx.scipy.special._logsoftmax', 'cupyx.scipy.special._basic', 'cupy._math.ufunc', 'cupy._math.rounding', 'cupyx.scipy.special._xlogy', 'cupyx.scipy.special._logsumexp', 'cupy._math.special', 'cupyx.scipy.special', 'cupyx.scipy.fft._fftlog', 'cupyx.scipy.fft._helper', 'cupyx.scipy.fftpack._fft', 'cupyx.scipy.fftpack', 'cupyx.scipy.fft._realtransforms', 'cupyx.scipy.fft', 'cupy.lib._routines_poly', 'cupy.lib._polynomial', 'cupy._math.sumprod', 'cupy._math.trigonometric', 'cupy._math.hyperbolic', 'cupy._math.window', 'cupy._math.explog', 'cupy._math.floating', 'cupy._math.rational', 'cupy._math.misc', 'cupy._misc', 'cupy._misc.byte_bounds', 'cupy._misc.memory_ranges', 'cupy._misc.who', 'cupy._padding', 'cupy._padding.pad', 'cupy._sorting.count', 'cupy._sorting.sort', 'cupy._statistics', 'cupy._statistics.correlation', 'cupy._statistics.order', 'cupy._statistics.meanvar', 'cupy._statistics.histogram', 'fastrlock', 'fastrlock.rlock']

    venv_root = os.path.join(base_dir, "venv")
    cupy_include_path = os.path.join(
        venv_root, "Lib", "site-packages", "cupy", "_core", "include", "cupy")
    cupy_include_dest = ".\\cupy\\core\\include\\cupy"
    subprocess.run([
        "./venv/bin/pyinstaller" if platform.system() != "Windows" else ".\\venv\\Scripts\\pyinstaller",
        "--noconfirm",
        "--onedir",
        "--console",
        "--noupx",
        "--clean",
        "--add-data", f"{src_path};src/",
        "--add-data", f"{cupy_include_path};{cupy_include_dest}",
        "--hidden-import", ','.join(hiddenimports),
        main_path
    ], check=True)
    """
    subprocess.run([
        "./venv/bin/pyinstaller" if platform.system() != "Windows" else ".\\venv\\Scripts\\pyinstaller",
        "--noconfirm",
        "--onedir",
        "--console",
        "--noupx",
        "--clean",
        "--add-data", f"{src_path};src/",
        main_path
    ], check=True)

    move_jsx_file(jsx_path)


def move_jsx_file(jsx_path):
    dist_dir = os.path.join(base_dir, "dist")
    main_dir = os.path.join(dist_dir, "main")
    target_path = os.path.join(main_dir, os.path.basename(jsx_path))
    try:
        shutil.copy(jsx_path, target_path)
    except Exception as e:
        print("Error while copying jsx file: ", e)


def clean_up():
    clean_permission = input(
        "Do you want to clean up the residue files? (y/n) ")

    if clean_permission.lower() == "y":
        print("Cleaning up...")
        try:
            spec_file = os.path.join(base_dir, "main.spec")
            if os.path.exists(spec_file):
                os.remove(spec_file)
        except Exception as e:
            print("Error while removing spec file: ", e)

        try:
            venv_file = os.path.join(base_dir, "venv")
            if os.path.exists(venv_file):
                shutil.rmtree(venv_file)
        except Exception as e:
            print("Error while removing the venv: ", e)

        try:
            build_dir = os.path.join(base_dir, "build")
            if os.path.exists(build_dir):
                shutil.rmtree(build_dir)
        except Exception as e:
            print("Error while removing the build directory: ", e)

    else:
        print("Skipping clean up...")

    print("Done!, you can find the built executable in the dist folder")


if __name__ == "__main__":
    create_venv()
    activate_venv()
    install_requirements()
    install_pyinstaller()
    create_executable()
    clean_up()
