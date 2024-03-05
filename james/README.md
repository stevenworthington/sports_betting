# Background

This code uses [PDM](https://pdm-project.org/latest/) to manage dependencies within a virtual environment, as well as to manage and build python packages/libraries for code reuse.

## Setup

1. Ensure you have `python 3.9` or newer
   1. [Optional] If you want to maintain multiple python versions on your computer look into [pyenv](https://github.com/pyenv/pyenv)
2. `pip install pdm`
3. Navigate to one of the subdirectories here e.g. "spam_data"
4. Run `pdm install`
   1. [Optional] Some projects have optional or dev depencies for things like jupyter support, code formatting etc. Look at the README.md for those specific packages for details
5. You now should have a `.venv` folder with the necessary depencies to run
6. `pdm run` will allow you to use this `venv`, e.g. `pdm run python -m spam_data.feature_engineering` or `pdm run jupyter notebook`
7. `pdm add` or `pdm remove` to manage dependencies

## Caveats

`spam_ml` relies on `spam_data` and there may be other dependencie arrangements. If you edit `spam_data` it will not be automatically "picked up" by `spam_ml` and you'll need to re-install `spam_data`

There is a concept called "editable install" that would work-around this, and any changes to `spam_data` would automatically be seen by `spam_ml` without a reinstall, but for some reason they do not seem to be working with `pdm` and a local package.
