# Koopman-Training-Pytorch-Tools
This repository holds various tools that might be useful for implementing Koopman training - a la  "A. S. Dogra and W. T. Redman, Optimizing Neural Networks via Koopman Operator Theory, *Advances in Neural Information Processings Systems* **33** (*NeurIPS* 2020)" - in Pytorch. It has three parts: tools for saving weights, tools for predicting where the parameters of the network are headed, and tools for setting the predicted network paramters to their Koopman predicted values. Below is discussion about each set of these tools and how/where to add them. 

## Requirements

The provided function makes use of only Pytorch, Numpy, and Scipy. 

## Saving the weights
This set of functions makes it easy to save all the network parameters. First flattens them all into one vector (indebted to Nicholas Guttenberg for this function) and then adds them to a field of the network model W. 

To implement these tools, first - somewhere before you start saving the weights - you should set self.model.W = []. Then, when you want to save the parameters (for instance, after an optimization step), you should call self.save_weights(). 

## Predicting the weights via Koopman 
This set of tools performs Exact Dynamic Mode Decomposition (DMD) as in Tu et al. 2014 (https://www.aimsciences.org/article/doi/10.3934/jcd.2014.1.391). It outputs the Koopman eigenfunction corresponding to the largest eigenvalue, which should be the fixed point of the parameter dynamics. More eigenfunctions could be incorporated for better approximation. 

This should be called, as [fixed_pt] = ExactDMD(self.model.W), after a decent amount of parameter data has been collected. Here decent is a tricky thing to quantify and has to be determined empircally.  

## Updating the network's parameters to their Koopman predicted values 
This set of tools maps the eigenfunction(s) found by Exact DMD to the network's parameters. 

This should be called after performing ExactDMD and is implemented first as, [new_params] = map_fixed_pt(self.model.parameters(), fixed_pt), and then setting self.model.parameters() = new_params. 

## Questions 

If you have any questions regarding the codebase or the associated NeurIPS paper, don't hesitate to email wredman@ucsb.edu or adogra@nyu.edu 
