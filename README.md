# Using GANs to generate faces with emojis

## Setup 

We use `conda` to manage our dependencies. Create an environment called `squad` with the required dependencies and activate it with. 
   ```
   conda env create -f environment.yml
   conda activate squad
   ```
Good to go!

## Image preprocessing

To clean the dataset, we crop the original video frames to include *only* faces (using `dlib`'s face detector) and downsize them to 128x128 images for the sake of quicker experimentation. 

Run image preprocessing with the following commands: 
   ```
   conda activate squad
   python image_preprocess.py
   ```
Image preprocessing took about 10 hours on a `gd4n.xlarge` AWS instance. 

## Running simple conditional GAN

   ```
   conda activate squad
   jupyter notebook
   ```
Open `ip.ip.ip.ip:8888` in your browser (substitude `ip.ip.ip.ip` for the public IPv4 address of AWS instance), and enter password `Stanford!`. 

Open `baseline_conditional_gan.ipynb` and run all cells. 
