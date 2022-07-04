# This provides my answers to the questions from Chapter 1: Your Deep Learning Journey in the Deep Learning for coders with fastai & PyTorch

**1. Do you need these for DL? Lots of math, Lots of data, Lots of expensive computers, A PhD T/F?**

F for all, need to convince myself of all, but I'll take their word for it, for now, except for the last point!

**2. Name 5 areas where DL is now the best tool in the world.**

* Computer vision
* Natural Language Processing
* Medical imaging/Radiology
* Robotics, self-driving cars
* Biology - protein folding
* Recommender system

**3. What was the name of the first device that was based on the principle of the artificial neuron?** 

Mark I Perceptron

**4. Based on the book of the same name, what are the requirements for parallel distributed processing (PDP)?**

PDP - most pivotal work in the past 50y, closer to other frameworks to how the brain works. The requirements include: 
- a set of processing units
- a state of activation
- an output function for each unit 
- a pattern of connectivity among units
- a propagation rule - propagating pattern of activities through the network 
- an activation rule - combines the inputs impinging on a unit with the current state of that unit to produce an output for that unit
- a learning rule - patterns of connectivity are modified by experience, 
- an environment, within which the system must operate!

**6. What is a GPU?**

Graphics Processing Unit, or a graphics card, a special kind of processor in your computer that can handle thousands of single tasks in parallel, 
esp designed for displaying 3D environments on a computer for playing games.

**7.** & **8.** 

Can go over code on page 17 of the book.

**9. Go over the [appendix](https://github.com/fastai/fastbook/blob/master/app_jupyter.ipynb).**

**10. Why is it hard to use a traditional computer program to recognize images in a photo?**

Because we don't know the steps our brain takes to detect that, it happens without us being consciously aware of it.

**11. What did Samuel mean by "weight assignment"?**

Weights are just variables and weight assignment is just choosing a particular set of values for those weights.

**12. What term do we normally use in DL for what Samuel called "weights"?**

Model parameters

**13. Draw a picture that summarizes Samuel's view of a ML model.**

We have inputs and weights (variables that get assigned a set of values), that feed into a model, which is a program to get a result. 
We evaluate the results through a measure of performance which in return influences how we tweak our weights to iterate again through the model.

**14. Why is it hard to understand why a DL model makes a particular prediction?**

Because of their "deep" nature and lots of layers, it's difficult to trace how a particular prediction has been made.

**15. What is the name of the theorem that shows that a neural network can solve any mathematical problem to any level of accuracy?**

If we regard the neural network as a mathematical function, it is a very flexible function depending on its weights. 
A mathematical proof called the universal approximation theorem shows that this function can solve any problem to any level of accuracy, in theory.
So the focus should be on training them i.e. finding good weight assignment.
Stochastic gradient decent (SGD) automatically updates weights for every problem.
NN is a particular kind of ML model that is highly flexible thus solving a large range of problems by finding the right weights.

**16. What do you need in order to train a model?**

- Inputs - images, 
- labels - to assess the results against, 
- weights, 
- our neural net model, 
- our results (whether an image is a dog or a cat in this first chapter example), 
- an automated way of assessing the effectiveness of weight assignment/updating the parameters to improve performance = an optimizer, SDG is our way of updating the weights/
- a loss function to assess the model performance  
- an optimizer

**17. How could a feedback loop impact the rollout of a predictive policing model?**

A predictive model based on where arrests have been made - this measures not where crime happens but rather biases as to where the police has gone to. 
If the police force uses these models to put more resources into these areas there will be even more arrests in these areas and so on, 
leading to a positive feedback loop. 

**18. Do we always have to use 224 x224 pixel images with the cat recognition model?**

No, you can give it any number (old pretrained model used this number)
Larger, better as more features will be learned, but slower and more mem consumption and the opposite is true as well.

**19. What is the difference between classification and regression?**

Image classification - predict the image class/category (e.g., whether there is a cat or dog in the image)
regression - predict one/more numeric categories such as temp, careful not to think of linear regression

**20. What is a validation set? What is a test set? Why do we need them?**

Validation set is a set we measure the model performance against, while the training happens on the training set. 
Generally, the accuracy on the validation set increases, until one point when it decreases because the training will start learning most 
points on the test set. That's when we stop.

**21. What will fastai do if you don’t provide a validation set?**

By default, it will randomly take 20% of your data and make that a validation set.

**22. Can we always use a random sample in a validation set? Why or why not?**

No, the example on time-series with dates. If you pick at random it might be very easy to find the points in between. 
So, you need to pick the last period in the time series that the model hasn't seen.
Or illegal fishing or safe driving models, you don't want to use the same person in different postures, or the same ship in different 
instances in both the training and validation, you want your model to be able to predict on new ppl or new ships.
In RS we need to look into whether the training or validation data come just from geographic region.

**23. What is overfitting, provide an example.**

When we trained on the same data and the model is really accurate for the cases it has seen but does poorly outside of it, see pg 29. 

**24. What is metric? How does it differ from loss?**

They both measure model performance but loss is the general term used in SGD to automatically adjust the weights to improve model 
performance whereas a metric is for human consumption so whatever is easier to interpret, such as error (proportion of validation cases that 
are classified erroneously) or accuracy 1- error.

**25. How can pretrained models help?**

This was one of the biggest revelations for me from this chapter that model architecture, which is sth that is talked about a lot in academic circles 
isn't as important but what is important is using a pretrained model. In various fields there are very few models that are pretrained thus 
it's hard to improve on. Pretrained models make use of existing weights which in first instances identify edges, gradient, contrast, colors. 
They are used for transfer learning.

*There is a lot of work to be done in the transfer learning for time series data as it's not well studied.* 

**26. What is the head of the model?**

In transfer learning when using a pretrained model, we replace the last layer with one or more new layers with randomized weights by updating 
the weights of the last layers faster than earlier layers to make it specific to the data set you are applying it to.

**27. What kind of features do early CNN layers find? How about the later layers?**

First layer has weights that represent edges such as diagonal, horizontal, and vertical edges; Then corners, repeating lines, circles and other 
simple patters, then higher-level semantic components such car wheels, text, flower petals.

**28. Are image models useful only for photos?**

Image models only work on images but lots of data sets can be represented as images for example: sounds can be converted to spectrogram 
(amount of each sound freq at each time in an audio file), creating a time-series into an image (alive detection, like the gramian angular 
difference field), or turning mouse clicks and movements into imgs for fraud detection (or checking if ppl are indeed working from home when 
they say they did lol)

**29. What is an architecture?**

Template for a mathematical function of the model, without weights/ values of the parameters, it doesn't do anything.

**30. What is segmentation?**

Assigning a probability of every pixel in an image belonging to a certain class.

**31. What is y_range used for? When do we need it?**

Pg. 46-47 where we predict a continuous number rather than a category, we need to tell fast.ai what range our target has using the y_range. 
Used in the recommendation system example.

**32. What are hyperparameters?**

Higher level choices that govern the weight parameters so parameters that are not tweaked during model training, such as data augmentation, 
learning rate, network architecture, number of epochs.

**33. What's the best way to avoid failures when using AI in an organization?**

Ensure you understand what validation and test datasets really are -hold out some test data that the vendor doesn’t get to see then YOU check 
their model against that data using a metric that YOU choose, and YOU decide what level of performance is adequate.
Also, good idea for you to try a simple baseline model yourself to understand how well it performs, you may get similar performance to the 
model produced by the 'expert'.

**Other notes from Chapter 1:**

- PyTorch fastest growing lib in DL, low level foundation lib, whereas fast.ai is most popular for adding higher-level functionality on top of PyTorch
- The whole concept of the book is a top-down approach, getting a model to run, dipping into theory as needed.
- What stood out for me the most in this chapter is the fact that transfer learning is the most imp thing rather than network architecture, 
but it's also the least studied!!!
- Terminology - different from what I used to call image classification; in DL this is called image segmentation!!






















