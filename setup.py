import os
import re
from setuptools import find_packages, setup


install_requires = ['numpy>=1.11.1', 'opencv-python', 'pillow', 'lxml', 'requests']


def get_version() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    version_file = os.path.join(current_dir,  "khandy/version.py")
    with open(version_file, encoding="utf-8") as f:
        version_match = re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M)
        if version_match:
            return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def get_long_description() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_dir, "README.md"), encoding="utf-8") as f:
        return f.read()
    

setup(
    name='khandy',
    version=get_version(),
    description='Handy Utilities for Computer Vision',
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    keywords='computer vision',
    packages=find_packages(),
    
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Utilities',
    ],
    url='https://github.com/quarrying/khandy',
    author='quarryman',
    author_email='quarrying@qq.com',
    maintainer='quarryman',
    maintainer_email='quarrying@qq.com',
    license='MIT',
    install_requires=install_requires,
    zip_safe=False)
