import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
     name='btjenesten',  
     version='0.14',
     author="Audun Skau Hansen",
     author_email="a.s.hansen@kjemi.uio.no",
     description="Tools for a brilliant future",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.uio.no/audunsh/btjenesten",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )
