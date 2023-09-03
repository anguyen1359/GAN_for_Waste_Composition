# GAN_for_Waste_Composition

SampleGAN.py --> GAN Model
SampleGAN_Trained_Discriminator.pth --> Trained Discriminator Model
SampleGAN_Trained_Generator.pth --> Trained Generator Model
waste_example.csv --> Imported CSV File
GANPCA_good.png --> PCA graph of Imported vs Generated Data


Developed a Deep Learning Model, specifically a Generative Adversarial Network, that took in a file containing # number of features, and generated similar data.
- Generated data and visualizations were primarily analyzed using Jupyter Interative Window

Created using two neural networks that compete against one another named the generator and disciminator.
- Discriminator role was to distinguish between real data provided by the input csv file and generated data produced by the generator.
- Generator role was to produce data that would cause the discriminator to label the generated data as real data.

Discriminator Neural Network
- Contained Linear, Dropout as input and hidden Layers
- LeakyReLu was used as activation function
- Sigmoid was used as output layer to produce a binary output (for real or fake data) 

Generator Neural Network
- Contained Linear for all layers
- ReLu was used as activation function


DataLoader was used to extract data from the input file and to generate batches for the training


When Training
- Binary Cross Entropy was used as the loss function
- Adam Optimizer was used as the Optimizer for both Generator and Discriminator


Evaluation
- Calculated a Mean Square Error of 1.0-2.5 and Root Mean Squared Error of 1.0-1.6
- Created Visualizations with Tensorboard and Matplotlib
- Used PCA to help visualize a better comparison and saw how it compared when using Kmeans

After Training and Evaluation, script exported generated data into a csv file
