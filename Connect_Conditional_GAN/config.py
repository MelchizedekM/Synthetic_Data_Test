class Config:
    """
    Configuration class to hold the parameters for training.
    """
    
    def __init__(self):
        # Number of epochs of training
        self.n_epochs = 200
        
        # Size of the batches
        self.batch_size = 64
        
        # Adam optimizer: learning rate
        self.lr = 0.0002
        
        # Adam optimizer: decay of first order momentum of gradient
        self.b1 = 0.5
        
        # Adam optimizer: decay of second order momentum of gradient
        self.b2 = 0.999
        
        # Number of CPU threads to use during batch generation
        self.n_cpu = 8
        
        # Dimensionality of the latent space
        self.latent_dim = 100
        
        # Size of each image dimension
        self.img_size = 32
        
        # Number of image channels
        self.channels = 1
        
        # Interval between image sampling
        self.sample_interval = 400
        
        # Whether to use bias in the model layers
        self.bias = True
        
        # The size of the initial condition
        self.condition_size = 4

