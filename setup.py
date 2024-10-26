import os
import subprocess

from Cython.Distutils import build_ext
from setuptools import Extension, find_packages, setup

# get __version__ variable
here = os.path.abspath(os.path.dirname(__file__))
exec(open(os.path.join(here, "tensor_bridge", "_version.py")).read())


def customize_compiler_for_nvcc(self):
    """Inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.

    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on.
    """

    # Get nvcc path
    nvcc_path = (
        subprocess.run(
            "which nvcc", shell=True, check=True, capture_output=True
        )
        .stdout.decode()
        .strip()
    )

    # Tell the compiler it can processes .cu
    self.src_extensions.append(".cu")

    # Save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # Now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == ".cu":
            # use the cuda for .cu files
            self.set_executable("compiler_so", nvcc_path)
            # use only a subset of the extra_postargs, which are 1-1
            # translated from the extra_compile_args in the Extension class
            postargs = extra_postargs["nvcc"]
        else:
            postargs = extra_postargs["gcc"]

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # Reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # Inject our redefined _compile method into the class
    self._compile = _compile


# Run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)


if __name__ == "__main__":
    from Cython.Build import cythonize

    # setup Cython build
    ext = Extension(
        "tensor_bridge.tensor_bridge",
        sources=[
            "tensor_bridge/native_tensor_bridge.cu",
            "tensor_bridge/tensor_bridge.pyx",
        ],
        language="c++",
        extra_compile_args={
            "nvcc": ["--compiler-options", "'-fPIC'"],
            "gcc": [],
        },
        extra_link_args=[],
    )

    ext_modules = cythonize(
        [ext], compiler_directives={"linetrace": True, "binding": True}
    )

    setup(
        name="tensor_bridge",
        version=__version__,
        description="Transfer tensors between PyTorch, Jax and more",
        long_description=open("README.md").read(),
        long_description_content_type="text/markdown",
        url="https://github.com/takuseno/tensor-bridge",
        author="Takuma Seno",
        author_email="takuma.seno@gmail.com",
        license="MIT License",
        packages=find_packages(exclude=["tests*"]),
        python_requires=">=3.8.0",
        zip_safe=False,
        package_data={
            "tensor_bridge": [
                "*.pyx",
                "*.pxd",
                "*.hpp",
                "*.cu",
                "*.pyi",
                "py.typed",
            ]
        },
        cmdclass={"build_ext": custom_build_ext},
        ext_modules=ext_modules,
    )
