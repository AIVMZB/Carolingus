# Medieval documents reader

The project was built to transcribe medieval Latin handwritten documents. The application detects text lines, words and clasifies each detected word.

### Set up

1. Clone the repo
    ```bash
    git clone https://github.com/AIVMZB/Carolingus.git
    cd Carolingus
    ```

2. Create Python virtual enviroment (`python=3.10.0` version is recommended)
    - Windows
        ```bash
        py -m venv venv
        venv\Scripts\activate
        ```
    
    - Linux
        ```bash
        python -m venv venv
        source venv/bin/activate
        ```

3. Install pytorch for your CUDA version
    - Linux
        ```bash
        pip install torch
        ```
    
    - Windows

        Navigate to [pytorch page](https://pytorch.org/get-started/locally/) to find command for your CUDA version. For example:
        ```bash
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ```

4. Install `ultralytics`
    ```bash
    pip install ultralytics
    ```

5. Install other libraries
    ```bash
    pip install -r requirements.txt
    ```

6. Download the [weights](https://drive.google.com/drive/folders/1nVQtXSZoo25pWf3lLEyjFa20DNbecqYa?usp=drive_link) and place the files into `models` directory.

### Run the app
The web UI runs as a [streamlit](https://streamlit.io/) application. 

```bash
streamlit run src/main.py
```

Navigate to the page at `localhost:8501`.

Images for tests are located in `assets` directory.
