# Using GANs to generate faces with emojis

## Setup 

We use `conda` to manage our dependencies. Create an environment called `emoji-gan` with the required dependencies and activate it with. 
   ```
   conda env create -n emoji-gan python==3.7
   pip install -r requirements.txt
   conda activate emoji-gan
   ```
Good to go!

## Image preprocessing

To clean the dataset, we crop the original video frames to include *only* faces (using `dlib`'s face detector) and downsize them to 128x128 images for the sake of quicker experimentation. We also use `dlib`'s face detector to remove faces with low confidence levels (these tend to be emojis and corrupt the generated images). 

Run image preprocessing with the following commands: 
   ```
   conda activate emoji-gan
   python image_preprocess.py
   ```
Image preprocessing took about 10 hours on a `gd4n.xlarge` AWS instance. 

Another option is to just ask chosk [a] stanford . edu or chloe [a] stanford . edu for the dataset!

Either way, the processed dataset should be located in a folder called `emoji_challenge_resized_128_faces` for the models to run smoothly. 

## Running simple conditional GAN

   ```
   conda activate emoji-gan
   jupyter notebook
   ```
Open `ip.ip.ip.ip:8888` in your browser (substitude `ip.ip.ip.ip` for the public IPv4 address of AWS instance), and enter password `Stanford!`. 

Open `baseline_conditional_gan.ipynb` and run all cells. 

## Running vanilla SAGAN

  ```
  conda activate emoji-gan
  cd Self-Attention GAN/
  python main.py
  ```
  
Run `tensorboard --log-dir PATH_TO_LOG` to see discriminator and generator loss over time. 

## Running cSAGAN

   ```
   conda activate emoji-gan
   jupyter notebook
   ```
Open `ip.ip.ip.ip:8888` in your browser (substitude `ip.ip.ip.ip` for the public IPv4 address of AWS instance), and enter password `Stanford!`. 

Open `Conditional Self-Attention GAN/Conditional Self-Attention GAN.ipynb` and run all cells. 
