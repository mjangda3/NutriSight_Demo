echo "Setting up the environment..."

python3 -m venv env

source env/bin/activate

pip install --upgrade pip

pip install -r requirements.txt

echo "Installation complete. To activate the virtual environment, run 'source env/bin/activate'."