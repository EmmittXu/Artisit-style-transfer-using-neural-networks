# Artisit-style-transfer-using-neural-networks
This project uses convolutional neural networks to generate images intellectually.  
Check our report for more details about this project.  
Credits given to Huafeng Shi and Kexin Pei, Columbia University.  

# A brief intro:  
Gievn a photograph and an artwork, our network draws a content representation from the photo and a style representation from the artwork and mixs them together to generate a new image that looks like the photograph but also conserves the artwork style.  

Two key steps in this style transfer are the content representation and the style representation:  
To do the content representation, we perform gradient descent on a white noise image to find another image that matches the feature responses of the original image. The loss function is defined as the squred-error loss between the two feature representations.  

To do the style representation, we use Gram matrix to evaluate the feature correlations between feature maps from different VGG layers and aim to minimize the mean-squred distance betwwen the entries of the Gram matrix from the original image and the Gram matrix of the image to be generated.  

Finally total loss is defined as the sum of content loss and style loss controller by a weight hyperparameter. We tried SGD and LM-BFGS approach to do the cost function optimization.  

For a detailed explaination, check out the original paper https://arxiv.org/pdf/1508.06576.pdf.


