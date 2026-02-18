from setuptools import setup, find_packages

# install any packages listed under the `src` directory
setup(
    name = 'local_repo_packages', # just a dummy name
    packages = find_packages('src'),
    package_dir = {'': 'src'},
    include_package_data = True,
    zip_safe = False, # apparently, this is needed to include the test dir
)
