**Build the Docker Image first**

`docker build -t aeseg .`


**Run the Gradio Demo App**

Download the `.pth` models and put them in `pretrain_weights/`.

`docker run -p 7860:7860 -v $(pwd)/pretrained_weights:/app/pretrained_weights aeseg`

Open `http://localhost:7860` in your browser.


**Or run the training script**

`docker run --gpus all aeseg python train.py` 

Change the model and train paramerters as your desire before training.


**Note:** 

IF YOU DON'T WANT TO USE DOCKER, create a new 3.11 conda env. Then `pip install -r requirements.txt` and `python demo.py`