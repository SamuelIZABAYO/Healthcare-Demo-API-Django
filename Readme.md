## **Healthcare System Demo**

<br>

- Uses `python-3.8.10`
- API built using Django
- Local hosting at `http://127.0.0.1:10000`
- Functionalities Present:
    - Diabetes Prediction `/diabetes-infer/`<br>
    - Liver Disease Prediction `/liver-disease-infer/`<br>
    - Cardiovascular Disease Prediction `/cardiovascular-disease-infer/`<br>
    - Brain MRI Abnormality Segmentation `/brain-mri-abnormality-infer/`<br>
    - Pneumonia Prediction using Chest X-Rays `/pneumonia-pre-infer/`<br>
    - Tuberculosis Prediction using Chest X-Rays `/tuberculosis-pre-infer/`<br>

<br>

### **Detailed Information**

<br>

1. Install `python 3.8.x`
2. Run `pip install virtualenv`
3. Run `make-env.bat`
4. Run `start-local-server.bat`. The API will now be served at `http://127.0.0.1:10000`
5. To run in production mode
    - Change `DEBUG` to `False` in Main.settings
    - Ensure appropriate environment variable is set
    - Run `collect-static.bat` before `start-server.bat` 

<br>

### **Notes on Models**

<br>

<pre>
1. brain-abnormality : pretrained
2. diabetes          : gbc
3. liver-disease     : etcs (Test a bit)
4. pneumonia         : d121-na384-oclr-f2-0.5
5. tuberculosis      : d121-na384-oclr-f4
</pre>

