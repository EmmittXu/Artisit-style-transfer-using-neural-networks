import theano
import theano.tensor as T
import numpy
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import pickle

from ImageUtils import ImageHelper
from Layers import InputLayer, ConvLayer, PoolLayer
from utils import floatX, set_all_param_values, get_outputs

import time, datetime

def main(style_weight='1e0', content_file='face_content.jpg', style_file='face_style.jpg'):
    start_time = time.time()
    print('starting time is: ', datetime.datetime.now())
    
    rng = numpy.random.RandomState(62616)
    
    IMAGE_W = 128
    ###############################################################
    # build the model
    ###############################################################
    def build_and_load_model():
        def build_model(theano_input):
            net = {}
            order = ['input', 'conv1_1', 'conv1_2', 'pool1', 'conv2_1', 'conv2_2', 'pool2', 'conv3_1',
                     'conv3_2', 'conv3_3', 'conv3_4', 'pool3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4',
                     'pool4', 'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4']
            net['input'] = InputLayer(theano_input, (1, 3, IMAGE_W, IMAGE_W))
            net['conv1_1'] = ConvLayer(net['input'], 64, 3, rng, flip_filters=False)
            net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, rng, flip_filters=False)
            net['pool1'] = PoolLayer(net['conv1_2'], poolsize=(2, 2), mode='average_exc_pad')
            net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, rng, flip_filters=False)
            net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, rng, flip_filters=False)
            net['pool2'] = PoolLayer(net['conv2_2'], poolsize=(2, 2), mode='average_exc_pad')
            net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, rng, flip_filters=False)
            net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, rng, flip_filters=False)
            net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, rng, flip_filters=False)
            net['conv3_4'] = ConvLayer(net['conv3_3'], 256, 3, rng, flip_filters=False)
            net['pool3'] = PoolLayer(net['conv3_4'], poolsize=(2, 2), mode='average_exc_pad')
            net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, rng, flip_filters=False)
            net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, rng, flip_filters=False)
            net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, rng, flip_filters=False)
            net['conv4_4'] = ConvLayer(net['conv4_3'], 512, 3, rng, flip_filters=False)
            net['pool4'] = PoolLayer(net['conv4_4'], poolsize=(2, 2), mode='average_exc_pad')
            net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, rng, flip_filters=False)
            net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, rng, flip_filters=False)
            net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, rng, flip_filters=False)
            net['conv5_4'] = ConvLayer(net['conv5_3'], 512, 3, rng, flip_filters=False)
            net['pool5'] = PoolLayer(net['conv5_4'], poolsize=(2, 2), mode='average_exc_pad')
            return net, order
        # build it
        net, order = build_model(T.tensor4())
        # load it
        values = pickle.load(open('./data/vgg19_normalized.pkl', 'rb'))['param values']
        set_all_param_values(net, values, order)
        return net
    
    net = build_and_load_model()
    # select the layer to use
    layers = ['conv4_2', 'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
    layers = {k: net[k] for k in layers}
    
    ###############################################################
    # get the images
    ###############################################################
    imageHelper = ImageHelper(IMAGE_W=IMAGE_W)
    photo, art = imageHelper.prep_photo_and_art(photo_path='./data/' + content_file, art_path='./data/' + style_file)
    
    input_im_theano = T.tensor4()
    # compute layer activations for photo and artwork
    outputs = get_outputs(layers, {net['input']:input_im_theano})
    # these features are constant which is the reference for loss
    photo_features = {k: theano.shared(output.eval({input_im_theano: photo}))
                      for k, output in zip(layers.keys(), outputs)}
    art_features = {k: theano.shared(output.eval({input_im_theano: art}))
                    for k, output in zip(layers.keys(), outputs)}
    
    
    ###############################################################
    # calculate loss and grads
    ###############################################################
    def gram_matrix(x):
        x = x.flatten(ndim=3)
        g = T.tensordot(x, x, axes=([2], [2]))
        return g
    
    def content_loss(P, X, layer):
        p = P[layer]
        x = X[layer]
    
        loss = 1. / 2 * ((x - p) ** 2).sum()
        return loss
    
    def style_loss(A, X, layer):
        a = A[layer]
        x = X[layer]
    
        A = gram_matrix(a)
        G = gram_matrix(x)
    
        N = a.shape[1]
        M = a.shape[2] * a.shape[3]
    
        loss = 1. / (4 * N ** 2 * M ** 2) * ((G - A) ** 2).sum()
        return loss
    
    def mrf_loss(A, X, layer):
        a = A[layer]
        x = X[layer]
        
        G = FI(x)
        A = FI(a)
           
        # equation 2
        phi_style = NN(G, A, a)
        # phi_style = A[T.zeros(A.eval().shape[0], dtype='uint8') + 10]
    #     phi_style = A
        
        loss = ((G - phi_style) ** 2).sum()
        return loss
    
    def NN(input_patches, style_patches, style):
        # style shape (1, number of filters (256 or 512), width, height)
        # print(style.eval().shape)
        # style_patches shape (number of patch, number of filters (256 or 512), width(3), height(3))
        # print(style_patches.eval().shape)
        # input_patches shape (number of patch, number of filters (256 or 512), width(3), height(3))
        # print(input_patches.eval().shape)
        
        # upper part of formula (3), namely, numerator, in shape (1, number of patch, width, height)
        numerator = T.nnet.conv2d(
            input=style,
            filters=input_patches,
            border_mode='valid',
            filter_flip=False,
        )
        numerator = numerator.reshape((numerator.shape[1], numerator.shape[2] * numerator.shape[3]))
        
        # lower part of formula (3), namely, denominator
        style_norm = T.sqrt(T.sum(T.square(style_patches), axis=(1, 2, 3)))  # norm of each style patch
        input_norm = T.sqrt(T.sum(T.square(input_patches), axis=(1, 2, 3)))  # norm of each input patch
        
        style_norm = style_norm.dimshuffle('x', 0)
        input_norm = input_norm.dimshuffle(0, 'x')
        denominator = T.dot(input_norm, style_norm)
        print('denominator shape', denominator.eval().shape, 'numerator shape', numerator.eval().shape)
        
        # normalized cross correlation, output shape (1, number of patch, width(148), height(148))
        output = numerator / denominator
        index = output.argmax(1)
        print('index of the best match patch in style feature map', index.eval())
        
        return style_patches[index]
        
        
    def FI(input_tensor):
        # input shape (1,channel,wdith,height)
        channel = input_tensor.shape[1]
        width = input_tensor.shape[2]
        height = input_tensor.shape[3]
        assert width.eval() == height.eval()
        
        numPatch = (width - 3 + 1) ** 2
        neibs = T.nnet.neighbours.images2neibs(input_tensor, neib_shape=(3, 3), neib_step=(1, 1))
        result = T.transpose(neibs.reshape((channel, numPatch, 3, 3)), (1, 0, 2, 3))
        
        return result
        
    
    def total_variation_loss(x):
        # return (((x[:, :, :-1, :-1] - x[:, :, 1:, :-1]) ** 2 + (x[:, :, :-1, :-1] - x[:, :, :-1, 1:]) ** 2) ** 1.25).sum()
        return (((x[:, :, :-1, :-1] - x[:, :, 1:, :-1]) ** 2 + (x[:, :, :-1, :-1] - x[:, :, :-1, 1:]) ** 2)).sum()
    
    
    
    # Get expressions for layer activations for generated image
    #     generated_image = theano.shared(floatX(np.random.uniform(-128, 128, (1, 3, IMAGE_W, IMAGE_W))))
    generated_image = theano.shared(photo.copy())
    gen_features = get_outputs(layers, {net['input']:generated_image})
    gen_features = {k: v for k, v in zip(layers.keys(), gen_features)}
    
    # Define loss function
    losses = []
    
    # content loss
    contentloss = content_loss(photo_features, gen_features, 'conv4_2')
    print('content_loss: ', contentloss.eval())
    losses.append(2e1 * contentloss)
    
    # mrf loss
    # losses.append(weight* mrf_loss(art_features, gen_features, 'conv1_1'))
    # losses.append(weight * mrf_loss(art_features, gen_features, 'conv2_1'))
    conv3_1_loss = mrf_loss(art_features, gen_features, 'conv3_1')
    conv4_1_loss = mrf_loss(art_features, gen_features, 'conv4_1')
    print('conv3_1 loss: ', conv3_1_loss.eval(), 'conv4_1 loss: ', conv4_1_loss.eval())
    losses.append(float(style_weight) * conv3_1_loss)
    losses.append(float(style_weight) * conv4_1_loss)
    # losses.append(weight* mrf_loss(art_features, gen_features, 'conv5_1'))
    
    # style loss
    # losses.append(0.2e6 * style_loss(art_features, gen_features, 'conv1_1'))
    # losses.append(0.2e6 * style_loss(art_features, gen_features, 'conv2_1'))
    # losses.append(0.2e6 * style_loss(art_features, gen_features, 'conv3_1'))
    # losses.append(0.2e6 * style_loss(art_features, gen_features, 'conv4_1'))
    # losses.append(0.2e6 * style_loss(art_features, gen_features, 'conv5_1'))
    
    # total variation penalty
    # losses.append(1e-3 * total_variation_loss(generated_image))
    
    total_loss = sum(losses)
    
    grad = T.grad(total_loss, generated_image)
    # Theano functions to evaluate loss and gradient
    f_loss = theano.function([], total_loss)
    f_grad = theano.function([], grad)
    
    
    ###############################################################
    # start to optimize
    ###############################################################
    # Helper functions to interface with scipy.optimize
    def eval_loss(x0):
        x0 = floatX(x0.reshape((1, 3, IMAGE_W, IMAGE_W)))
        generated_image.set_value(x0)
        return f_loss().astype('float64')
    
    def eval_grad(x0):
        x0 = floatX(x0.reshape((1, 3, IMAGE_W, IMAGE_W)))
        generated_image.set_value(x0)
        return np.array(f_grad()).flatten().astype('float64')
    
    # Optimize, saving the result periodically
    x0 = generated_image.get_value().astype('float64')
    optimize.fmin_l_bfgs_b(eval_loss, x0.flatten(), fprime=eval_grad, maxfun=400)
    x0 = generated_image.get_value().astype('float64')
    transfered = floatX(x0.reshape((1, 3, IMAGE_W, IMAGE_W)))
    
    
    ###############################################################
    # display result
    ###############################################################
    
    plt.figure(figsize=(12, 12))
    
    plt.subplot(1, 3, 1)
    plt.gca().xaxis.set_visible(False)
    plt.gca().yaxis.set_visible(False)
    plt.imshow(imageHelper.deprocess(photo))
    
    plt.subplot(1, 3, 2)
    plt.gca().xaxis.set_visible(False)
    plt.gca().yaxis.set_visible(False)
    plt.imshow(imageHelper.deprocess(art))
    
    plt.subplot(1, 3, 3)
    plt.gca().xaxis.set_visible(False)
    plt.gca().yaxis.set_visible(False)
    plt.imshow(imageHelper.deprocess(transfered))
    plt.tight_layout()
    
    #plt.savefig('data/output/result_' + style_weight + '.jpg')
    plt.show()
    
    
    print("--- running time: %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    main()
    