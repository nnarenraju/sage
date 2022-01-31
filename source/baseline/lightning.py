# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Filename        = Foobar.py
Description     = Lorem ipsum dolor sit amet

Created on Sat Nov 27 01:34:38 2021

__author__      = nnarenraju
__copyright__   = Copyright 2021, ProjectName
__credits__     = nnarenraju
__license__     = MIT Licence
__version__     = 0.0.1
__maintainer__  = nnarenraju
__email__       = nnarenraju@gmail.com
__status__      = ['inProgress', 'Archived', 'inUsage', 'Debugging']


Github Repository: NULL

Documentation: NULL

"""

# IN-BUILT
import pytorch_lightning as pl


# Work-around to dynamic inheritance
def simple(ModelClass, optimizer, scheduler, loss_function):
    """
    run_lightning excecutes the usual lightning module
    The __init__() and forward() are inherited from the ModelClass passed
    Workaround to dynamic inheritance by passing ParentClass as argument

    Parameters
    ----------
    classifier_name : str
        Name of the Lightning Classifier to initialise and return
    ModelClass : class
        Parent model class with frontend + backend and forward()
    optimizer : class
        Optimizer class provided in configs.py
    scheduler : class
        Scheduler class provided in configs.py

    Returns
    -------
    None.

    """
    
    class SimpleLightningClassifier(pl.LightningModule):
        """
        Simplest possible toy implementation of Lightning Classifier
        
        User-Defined Functions
        ----------------------
        forward - 
        compute_loss - 
        training_step - 
        validation_step - 
        configure_optimizers -
        
        """
        
        def __init__(self):
            super().__init__()
            # Getting attributes from ModelClass
            self.check_batch_idx = 0
            self.average_batch_loss = []
            if hasattr(ModelClass, "backend"):
                self.backend = ModelClass.backend
            self.frontend = ModelClass.frontend
        
        def forward(self, x):
            # Forward from ModelClass inherited
            out = ModelClass.forward(x)
            return out
        
        def compute_loss(self, logits, labels):
            return loss_function(logits, labels)
      
        def training_step(self, train_batch, batch_idx):
            x, y = train_batch
            # Logits are predictions
            logits = ModelClass.forward(x)
            # Temporary display option (sanity check)
            # print("Label = {}".format(y))
            # print("Prediction = {}".format(logits))
            loss = self.compute_loss(logits, y)
            # Store average loss
            self.average_batch_loss.append(loss)
            # Manual progress report
            if batch_idx == self.check_batch_idx:
                print(f"Batch {batch_idx} - Loss={loss}")
                if batch_idx != 0:
                    avg_loss = sum(self.average_batch_loss) / len(self.average_batch_loss)
                    print("Average loss for batch {} = {}".format(batch_idx-1, avg_loss))
                    # Reset avg batch loss for next batch
                    self.average_batch_loss = []
                self.check_batch_idx += 1
                
            self.log("train_loss", loss)
            # print("Training loss = {}".format(loss))
            return loss
        
        """
        def training_epoch_end(self, outs):
            # log epoch metric
            self.log('train_acc_epoch', self.accuracy.compute())
        """
        
        def validation_step(self, val_batch, batch_idx):
            x, y = val_batch
            logits = ModelClass.forward(x)
            loss = self.compute_loss(logits, y)
            # print("Validation loss = {}".format(loss))
            self.log('val_loss', loss)
      
        def configure_optimizers(self):
            # Pass optimizer and scheduler here
            # Multiple optimizer can be used as well
            return [optimizer], [scheduler]
    
    
    """ Create a Lightning Model """
    model = SimpleLightningClassifier()
    
    # Return model to train.py for fitting
    return model
