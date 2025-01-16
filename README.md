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


1. **Execute the Demo Script:**
   ```bash
   python src/demo.py
   ```

3. **Expected Output:**
   - The demo will process the dataset and display metrics such as ROC AUC and accuracy scores.
   - Visualizations like ROC curves for each timestep will be displayed.

---
