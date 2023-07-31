1. Please install the required packages/libraries from the "requirements.txt" file using "!pip install -r requirements". Also if required, you may need to install other missing dependencies.


2. Install the pytorch along with cuda in your v-environment since this model requires CUDA for inferencing. For that type: 
"conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia"




3. ----------RUUNING THE INFERENCE----------

A) Using the python (.py) file:
In the terminal:
step1: "conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia"
step2: "pip install -r requirements.txt"
step3: "python main.py" command will run the script for inference (prompts user to press enter to start recording voice and again to stop the recording).
step5: The recorded voice will be saved in the Rec folder and the system will provide the prediction for the corresponding text for the audio.

B) The jupyter notebook (.ipynb) file:
step1: install the requirements
step2: Run the cells one by one, 
step3: record cell will record while transcrption 


----------Details about the folder----------
The saved fine tuned model is kept in ShishirAI_NepWav2Vec2 from where it can be loaded in the python code. 

