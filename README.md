# NutriSight_Submission

## 1. System Requirements

### Software Dependencies

- **Operating Systems:** 
  - **macOS:** 14.6 (23G80)
- **Programming Languages & Frameworks:** 
  - Python 3.10.12
- **Libraries:**
  - NumPy 1.26.4
  - Pandas 2.2.2
  - Matplotlib 3.8.0
  - Seaborn 0.13.2
  - TensorFlow 2.17.1
  - Scikit-learn 1.6.0
- **Other Dependencies:** 
  - Git 2.28

### Non-Standard Hardware
- **Memory:** Minimum 16 GB RAM recommended

---

## 2. Installation Guide


1. **Open Terminal** on your computer.

2. **Navigate** to the directory where you want to clone the project and run the installer:
   ```bash
   cd path/to/your/desired/directory
   ```

3. **Clone the Repository, Navigate to the Project Directory, and Run the Installer Script:**
   ```bash
   git clone https://github.com/mjangda3/NutriSight_Demo
   cd NutriSight_Demo
   ./install.sh
   ```
   *If you encounter a permission error, make the script executable:*
   ```bash
   chmod +x install.sh
   ./install.sh
   ```

### Setting Up the Virtual Environment

1. **Activate the Virtual Environment:**
   ```bash
   source env/bin/activate
   ```

2. **Verify Activation:**
   - Your terminal prompt should now show `(env)` indicating the virtual environment is active.
   
   *If you need to deactivate the environment later, simply run:*
   ```bash
   deactivate
   ```

---

## 3. Demo

### Preparing the Dataset

1. **Place Your Data Files:**
   - Ensure that `X.csv` and `y.csv` are located in the `data/` directory of the project.
   
2. **Verify Data Format:**
   - `X.csv`: Should contain your feature data structured as required by the model.
   - `y.csv`: Should contain your label data structured accordingly.

### Running the Demo Script

1. **Ensure the Virtual Environment is Activated:**
   ```bash
   source env/bin/activate
   ```

2. **Execute the Demo Script:**
   ```bash
   python src/demo.py
   ```

3. **Expected Output:**
   - The demo will process the dataset and display metrics such as ROC AUC and accuracy scores.
   - Visualizations like ROC curves for each timestep will be displayed.

---

## 4. Usage

### Running the Main Script

1. **Ensure the Virtual Environment is Activated:**
   ```bash
   source env/bin/activate
   ```

2. **Execute the Main Training Script:**
   ```bash
   python src/main.py --input_X data/X.npy --input_y data/y.npy --output_dir results/
   ```

3. **Review the Results:**
   - The processed data and model outputs will be available in the `results/` directory.
   - Plots such as training and validation loss will be displayed.

---

## 5. Additional Information

- **Version Control:**
  - The project uses Git for version control. Ensure you have Git installed and configured.
  
- **Documentation:**
  - Additional documentation can be found in the `docs/` directory, specifically `additional_documentation.pdf`.

- **Testing:**
  - Incorporate test cases to ensure the reliability of the software. Consider adding a `tests/` directory with relevant test scripts.

- **Licensing:**
  - Specify the license under which your software is distributed by including a `LICENSE` file if applicable.

- **Contact Information:**
  - For any questions or issues, please contact [your email/contact information].

---

## 6. Final Checklist

Before finalizing your submission, ensure the following:

1. **Functionality:**
   - Test the installation and demo steps on a fresh environment to confirm that all instructions are accurate and complete.

2. **Data Integrity:**
   - Verify that `X.csv` and `y.csv` are correctly formatted and compatible with your scripts.

3. **Code Quality:**
   - Ensure your code is clean, well-documented, and free of errors.

4. **Completeness:**
   - Confirm that all required files are included and properly organized within the ZIP file.

5. **Accessibility:**
   - If using a repository link, double-check that reviewers have the necessary access permissions.

6. **Compliance:**
   - Ensure that your submission adheres to Nature Communications' guidelines and any specific instructions provided.

---

## 7. Example Usage

### Training the Model

```bash
python src/main.py --input_X data/X.npy --input_y data/y.npy --output_dir results/
```

**Notes:**

- Ensure that `X.npy` and `y.npy` are present in the `data/` directory. If starting from `X.csv` and `y.csv`, you need to preprocess them using the provided `preprocess.py` script.

### Running the Demo

```bash
python src/demo.py --input_X data/X.csv --input_y data/y.csv --output_dir demo_results/
```

**Notes:**

- The demo script assumes that `X.csv` and `y.csv` are in the correct format and shape expected by the model.

---
