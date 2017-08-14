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

# Note
1. Directory:
    (1) ArtTransferNaive: the implementation of the paper:
	L. A. Gatys, A. S. Ecker, and M. Bethge. A neural algorithm of artistic style. arXiv preprint arXiv:1508.06576, 2015.

    (2) ArtTransferMRF: the implementation of the paper:
	C. Li and M. Wand. Combining markov random fields and convolutional neural networks for image synthesis. arXiv preprint arXiv:1601.04589, 2016.


2. Note: In each directory, there is another README file that briefly explains
   how to run the code.

