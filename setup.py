'''
    setup.py file for log_erfc.c
    Note that since this is a numpy extension
    we use numpy.distutils instead of
    distutils from the python standard library.

    Calling
    $python setup.py build_ext --inplace
    will build the extension library in the current file.

    Calling
    $python setup.py build
    will build a file that looks like ./build/lib*, where
    lib* is a file that begins with lib. The library will
    be in this file and end with a C library extension,
    such as .so

    Calling
    $python setup.py install
    will install the module in your site-packages file.

    See the distutils section of
    'Extending and Embedding the Python Interpreter'
    at docs.python.org  and the documentation
    on numpy.distutils for more information.
'''
import sys
from pathlib import Path
from setuptools import setup, find_packages


def configuration(parent_package='', top_path=None):
    import numpy
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.misc_util import get_info

    info = get_info('npymath')

    config = Configuration('sfa_utils',
                            parent_package,
                            top_path)
    config.add_extension('npufunc',
                         ['src/pysfa/log_erfc.c'],
                         extra_info=info)

    return config

if __name__ == "__main__":
    base_dir = Path(__file__).parent
    src_dir = base_dir/"src"/"pysfa"
    
    sys.path.insert(0, src_dir.as_posix())
    import __about__ as about

    with (base_dir/"README.md").open() as f:
        long_description = f.read()
        
    install_requirements = [
        "numpy",
        "scipy"
    ]

    test_requirements = [
        "pytest",
        "pytest-mock",
    ]

    doc_requirements = []
    
    from numpy.distutils.core import setup
    setup(name=about.__title__,
          version=about.__version__,

          description=about.__summary__,
          long_description=long_description,
          license=about.__license__,
          url=about.__uri__,

          author=about.__author__,
          author_email=about.__email__,

          package_dir={"": "src"},
          packages=find_packages(where="src"),
          include_package_data=True,

          install_requires=install_requirements,
          tests_require=test_requirements,
          extras_require={
              "docs": doc_requirements,
              "test": test_requirements,
              "dev": doc_requirements + test_requirements
          },
          configuration=configuration,
          zip_safe=False,)
