# ML-from-scratch

### to test any model 
- create a fork
- clone the repo to your system
- create virtual environment using 
```python
python -m venv venv
```
- activate venv
```python
source venv/bin/activate
```
- go to the test folder 
- and create a file `filename.py`
- then in the top add these lines (change the modelname with your modelname and same for the classname)
```python
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.modelname import model_class
```
- run the file 
```python
python filename.py
```

### to push 
- come to the base directory
```git
cd ..
```
- create a new branch using
```git
git checkout -b your_branch_name
```
- add, commit , push
```git
git add test/filename.py
git commit -m "your commit messege"
git push origin your_branch_name
```