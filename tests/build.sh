cd ../
rm -r ./dist

pip install --upgrade hatch twine
hatch build