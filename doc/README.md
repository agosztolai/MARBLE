

# Documentation compilation
We followed this webpage to create the doc with sphinx: https://daler.github.io/sphinxdoc-test/includeme.html

## Create the doc structure
This is to be done the first time, then the next to upate it. 
1) Create a folder called MARBLE-docs near the MARBLE git folder (same level):
``` mkdir MARBLE-docs```
2) Clone the MARBLE repo in subfolder html: 
```git clone git@github.com:agosztolai/MARBLE.git html```
3) move into it: 
```cd html```
4) Switch branches to gh-pages:
```
git branch gh-pages
git symbolic-ref HEAD refs/heads/gh-pages  # auto-switches branches to gh-pages
rm .git/index
git clean -fdx
```

## Update the doc
1) in the main repo (MARBLE), do
```
cd docs
make html
```
this will update the files in MARBLE-docs. 

2) To update the git at the same time as compiling, just do 
```
make full
```

Alternatively, you can update the git by hand using:
```
cd ../../MARBLE-docs/html
git add .
git commit -m "rebuilt docs"
git push origin gh-pages
```
