# Framework description

JADE is a simple framework designed to easily implement deep learning models and log results and architectures using Weights and Biases.

The main design principle consists in separating the definition of the following elements:
- `Dataset`: the data source. All pytorch datasets are supported by default, custom datasets can be easily added to the framework.
- `Trainer`: the training algorithm and which components need to be stored. 
- `Architecture`: the specific structure of the models used in the experiment.
- `Evaluation`:  the metrics to be computed regularly during training

The interaction between the aforementioned components is visualized in the following scheme:

![](figures/JADE.png)

Each component can be defined by writing the code in the respective `modules/` folder.
More details on the framework usage can be found in the **Usage** section

# Installation
The conda and pip dependencies required to run our code are specified in the `environment.yml` environment file and can be installed by running
```shell script
conda env create -f environment.yml
```
Activate the environment
```shell script
conda activate pytorch_and_friends
```

## Weights and Bias setup
Run the initialization command
```shell script
wandb init
```
and enter your username and password as required.

Set two environment variables 
```shell script
export WANDB_USER='<your_username>'
export WANDB_PROJECT='<your_project_name>'
```
before running your scripts.

# Usage

The JADE library allows to implement and run models on multiple datasets and hyper-parameters configurations.
The framework is designed so that new models, datasets and evaluation metrics and hyper-parameters can be added with 
minimal changes to the code by explicitly separating the different components. The implementation of a new model is done
in 6 steps, which are discussed in detail in the following sections:

1) Define the training algorithm. The implementation must extend the `core.trainer.Trainer` class.
  More details can be found in the section `Models and Algorithms`
2) Write the definition of the required neural networks architectures (section `Architectures`)
3) Create the dataset definition if not already supported by pytorch (section `Dataset`).
4) Define the evaluation metrics by extending the `Evaluation` class (section `Evaluation`)
5) Write the model configuration, dataset and evaluation .yaml file to specify the respective hyperparameters. (section `Parameters confituration`)
6) Run the model or define a sweep file for testing hyper-parameter ranges (section `Running the experiments`).

## Models and Algorithms
Each new Model and algorithm must extend the `core.Trainer` class and define the architecture initialization and training.
The models need to override the default `initialize(**params)` and `train_step(data)` as discussed in the following sections.

### Initialization
All arguments of the `initialize(**params)` method can be directly specified in the corresponding configuration file (see section `Parameters confituration/Model and architectures`),
therefore all the parameters that needs to be explored can be added to the method signature.

```python
# Example of the Variational Autoencoder Model https://arxiv.org/abs/1312.6114

# Content of the models/VAE.py file

class VariationalAutoencoder(Trainer):
    def initalize(self, z_dim, encoder_layers, decoder_layers, beta, lr, batch_size, n_workers=0):
        # The value of the z_dim, n_encoder_layers, n_decoder_layers, beta lr, batch_size and n_workers are defined 
        # in the configuration file or specified as additional arguments when running the train python file
        
        # Store the value of the regularization strength beta for the loss computation
        self.beta = beta
```

Within the `initialize` method, the `self.instantiate_architecture(class, params)` method can be called to instantiate 
the corresponding class defined in the `Architecture file` (see section `Architectures`) with the specified parameters.

```python
        # initialize the encoder "Encoder(z_dim, layers)" defined in the architecture python file
        self.encoder = self.instantiate_architecture('Encoder', 
                                                     z_dim=z_dim, layers=encoder_layers)
        # initialize the encoder "Decoder(z_dim, layers, sigma)" defined in the architecture python file
        self.decoder = self.instantiate_architecture('Decoder', 
                                                     z_dim=z_dim, layers=decoder_layers, sigma=0.1)
        # Initialize the Gaussian prior passing the number of dimensions
        self.prior = self.instantiate_architecture('Prior', 
                                                   z_dim=z_dim)
        
        # Initialize the optimizer passing the parameters of encoder, decoder and the specified learning rate
        self.opt = Adam([
            {'params': encoder.parameters(), 'lr':lr},
            {'params': decoder.parameters(), 'lr':lr},
        ])
```
All the attributes (and architectures) that needs to be stored/restored can be specified by calling the 
`self.add_attribute_to_store(attribute_name)` method
```python
        # Store the parameters of encoder, decoder and optimizer
        self.add_attribute_to_store('encoder')
        self.add_attribute_to_store('decoder')
        self.add_attribute_to_store('opt')
```

The attribute `self.train_loader` also needs to be instantiated within the `initialize` method to specify the data loader
used during training. The datasets defined in the `Dataset` configuration files are accessible by calling the 
`self.get_dataset(dataset_name)` method (see section `Parameters confituration/Dataset`)

```python
        # Instantiate a default pytorch DataLoader using the 'train' dataset defined in the dataset confituration file
        self.train_loader = DataLoader(self.get_dataset('train'),
                                       batch_size=batch_size, n_workers=n_workers, shuffle=True)
```

### Train Step
Each model needs to define a train step that will receive a batch of data from the train loader for each new iteration.
```python
    def train_step(self, data):
        # Note that the data is already moved to the device of the trainer and all the architectures are set to train mode
        x, _ = data
        
        # Encode a batch of data
        q_z_given_x = self.encoder(x)
        
        # Sample the representation using the re-parametrization trick
        z = q_z_given_x.rsample()
        
        # Compute the reconstruction distribution
        p_x_given_z = self.decoder(z)
        
        # The reconstruction loss is the expected negative log-likelihood of the input 
        #  - E[log p(X=x|Z=z)]
        rec_loss = - p_x_given_z.log_prob(x).mean()
        
        # The regularization loss is the KL-divergence between posterior and prior
        # KL(q(Z|X=x)||p(Z)) = E[log q(Z=z|X=x) - log p(Z=z)]
        reg_loss = q_z_given_x.log_prob(z) - self.prior().log_prob(z)
        
        loss = rec_loss + self.beta * reg_loss

        self.opt.zero_grad()
        loss.backwards()
        self.opt.step()
        
    # Implementation of a reconstruct method for logging purposes
    def reconstruct(self, x, sample_latents=False):
        # If specified sample the latent distribution
        if sample_latents:
            z = self.encoder(x).sample()
        # Otherwise use the mean of the posterios
        else:
            z = self.encoder(x).mean
            
        # Return the mean of p(X|Z=z)
        x_rec = self.decoder(z).mean
        return x_rec

######## End of VAE.py
```
Different quantities can be easily added to the log by calling the `self.add_loss_item(name, value)`.
The values will be added to the corresponding tensorboard log. The logging frequency can be reduce by setting the 
`log_loss_every` variable (see section `Running the experiments`).
```python
        # Add the two loss components to the log
        # Note that only scalars are currently supported although custom logging can be implemented
        # by extending the implemented methods
        self.add_loss_item('Rec Loss', rec_loss.item())
        self.add_loss_item('Reg Loss', reg_loss.item())
```
The following methods can be extended if required:
- `on_start()`: method called only once after initialization
- `on_iteration_end()`: method called at the end of each training step. Don't forget to increment the iteration count if
this is overwritten
- `on_epoch_end()`: method called once at the end of each training epoch. Don't forget to increment the epoch count if 
this is overwritten

### Architectures
The different architectures that are instantiated in the Trainer described in the previous section
needs to be specified in the 'architectures_file'.
```python

# Content of the architectures/simple_MNIST.py file

# Model for p(Z|X)
class Encoder(nn.Module):
    def __init__(self, z_dim, layers):
        super(Encoder, self).__init__()
        
        # Create a stack of layers with ReLU activations as specified
        nn_layers = make_stack([N_INPUT] + layers)
            
        self.net = nn.Sequential(
            Flatten(),                                      # Layer to flatten the input
            *nn_layers,                                     # The previously created stack
            nn.ReLU(True),                                  # A ReLU activation
            StochasticLinear(n_hidden, z_dim, 'Normal')     # A layer that returns a factorized Normal distribution
        )

    def forward(self, x):
        # Note that the encoder returns a factorized normal distribution and not a vector
        return self.net(x) 

# ... The definition of Decoder are analogous and can be found in the `architectures/simple_MNIST.py` file
```
Note that the parameters can be passed when calling the `instantiate_architecture` in the model definition.

### Dataset
The default torchvision datasets are supported by the framework.
We recommend to create a simple Wrapper to make sure the Dataset object accepts only simple arguments such as float,
strings, list and dictionaries as parameters so that the dataset can be described with a simple yaml configuration file.
```python
# Content of examples/modules/data/MNIST.py

class MNISTWrapper(MNIST):
    def __init__(self, **params):
        # Dataset transforms are objects, this wrapper does bypass the issue
        transforms = ToTensor()
        super(MNISTWrapper, self).__init__(**params, transforms=transforms)

    def __getitem__(self, item):
        x, y = super(MNIST, self).__getitem__(item)
        # Convert into a dictionary for convenience
        return {'x': x, 'y': y}
```
Unless a wrapper is required as in the example reported above to conver PIL images to tensors, torchvision classes 
can be referred directly from the configuration file.

### Evaluation
The Jade framework allows for the definition of arbitrarily evaluators used for logging purposes.
Each evaluator must extend the `jade.eval.Evaluaiton` class and extend the `initialize(**params)` and `evaluate()` methods.
Each evaluator has access to all the dataset definitions and the trainer object that can be easily accessed as attributes.

The `initialize(**params)` method is used for the initialization of the evaluator and receives the parameters form the
evaluation configuration file.
```python
# Code from examples/modules/eval/image_eval.py

class ReconstructionLogger(Evaluation):
    def initialize(self, evaluate_on, n_pictures=10, sample_images=False, sample_latents=False):
        # Consider the dataset labeled with the specified name (names are defined in the dataset configuration file).
        self.dataset = self.datasets[evaluate_on]
        
        self.n_pictures = n_pictures
        self.sample_images = sample_images
        self.sample_latents = sample_latents
        
        # Check that the model has a definition of a method to reconstrut the inputs
        if not hasattr(self.trainer, 'reconstruct'):
            raise Exception('The trainer must implement a reconstruct(x) method with `x` as a picture')
        
    def sample_new_images(self):
        # sample the required number of pictures randomly
        ids = np.random.choice(len(self.dataset), self.n_pictures)
        images_batch = torch.cat([dataset[id]['x'].unsqueeze(0) for id in ids])
        
        return images_batch
        
            
    def evaluate(self):
        # If the images are not sampled dynamically, pick the first n_pictures from the dataset
        if not self.sample_images:
            x = torch.cat([dataset[id]['x'].unsqueeze(0) for id in range(self.n_pictures)])
        # Otherwise pick random ones
        else:
            ids = np.random.choice(len(self.dataset), self.n_pictures)
            x = torch.cat([dataset[id]['x'].unsqueeze(0) for id in ids])
        
        # Move the images to the correct device
        x = x.to(trainer.get_device())
        
        # Compute the reconstructions
        x_rec = trainer.reconstruct(x).to('cpu')
        
        # Concatenate originals and reconstructions
        x_all = torch.cat([x, x_rec], 2)
        
        # Return a dictionary used for logging
        return {
            'type': 'figure',                   # Type of the logged object, to be interpreted by the logger
            'value': x_all,                     # Value to log
            'iteration': trainer.iterations     # Iteration count at the point of logging
        }
```
The frequency at which each evaluation is produced can be also specified from the evaluation configuration file.
At the moment only support for scalars and figures has been added, but the interface can be easily adapted to fit any
other data-type which is supported by tensorboard/wandb.

### Parameters configuration
The value of all the parameters for model training, dataset specification and evaluation needs to be included into 3 
respective configuration files.


### Environment variables
if 'DEVICE' in os.environ:
    device = os.environ['DEVICE']

num_workers = args.num_workers
if 'N_WORKERS' in os.environ:
    num_workers = int(os.environ['N_WORKERS'])

if 'DATA_ROOT' in os.environ:
    data_root = os.environ['DATA_ROOT']

if 'EXPERIMENTS_ROOT' in os.environ:
    experiments_root = os.environ['EXPERIMENTS_ROOT']

