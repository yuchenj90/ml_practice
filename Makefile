.PHONY: clean purge lab lab-win venv venv-win

venv:
	virtualenv -p python3 venv
	venv/bin/pip install --upgrade pip
	venv/bin/pip install -r requirements.txt
	venv/bin/pip install -e .

venv-win:
	python -m venv venv
	venv\Scripts\python.exe -m pip install --upgrade pip
	venv\Scripts\python.exe -m pip install -r requirements.txt
	venv\Scripts\python.exe -m pip install -e .


lab:
	venv/bin/jupyter lab

lab-win:
	.\venv\Scripts\python -m jupyter lab

clean:
	find . -name '*.pyc' | xargs rm -r
	find . -name '*.ipynb_checkpoints' | xargs rm -r
	find . -name '__pycache__' | xargs rm -r
	find . -name '.pytest_cache' | xargs rm -r
	find . -name '*.egg-info' | xargs rm -r

purge: clean
	rm -rf env
