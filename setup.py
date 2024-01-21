from setuptools import setup

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(name='segmix',
      version='0.0.1',
      description='Package for Mixing Stable Diffusiion XL Models by Segmind',
      url='https://www.segmind.com/models',
      author='Segmind',
      license='MIT',
      packages=['segmix'],
      author_email='yatharthg@segmind.com',
      install_requires=[
          'torch>=2.0.0',
          'safetensors',
          'diffusers',
          'transformers'
      ],
      long_description=long_description,
      long_description_content_type='text/markdown',
      zip_safe=False)