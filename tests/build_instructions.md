Remove old package
`rm -r ./dist`

Build Package
`pip install --upgrade hatch twine`
`hatch build`

Deploy to TestPyPi
`python -m twine upload --repository testpypi dist/*`

Test Install
`./testpypi-env.sh`

Publish to PyPi
`python -m twine upload dist/*`