#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    This function is to Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    # load the graph by its names using vgg tag, and extract each layer by its name defined in this function
    graph = tf.get_default_graph()
    tf.saved_model.loader.load(sess, [vgg_tag],vgg_path)
    
    w1 = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    w3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    w4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    w7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return w1, keep, w3, w4, w7
# Unit test of load vgg model
tests.test_load_vgg(load_vgg, tf)

def conv_1by1(layer_in, num_classes):
    # Construct fully connnected layer using 1 by 1 convolution, and add regulaation
    kernel_size = 1
    stride = 1
    return tf.layers.conv2d(layer_in, num_classes, kernel_size, stride, padding = 'same',
                               kernel_initializer = tf.truncated_normal_initializer(stddev=0.001),
                               kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3))

def upsample(layer_in, kernel_size, stride, num_classes):
    # This function is to upsample the layer, the dimension is scaled by the stride
    return tf.layers.conv2d_transpose(layer_in, num_classes, kernel_size, stride, padding='same',
                                      kernel_initializer = tf.truncated_normal_initializer(stddev=0.01),
                                      kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3))
# upscale by 2, 2, 8

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    This funciton is to create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    # layer 7, conv 1 by 1 and upsample by 2
    layer7_conv_1by1 = conv_1by1(vgg_layer7_out, num_classes)
    layer7_upsample = upsample(layer7_conv_1by1, 4, (2, 2), num_classes)
    
    # layer 4
    layer4_scaled = tf.multiply(vgg_layer4_out, 0.01)
    layer4_conv_1by1 = conv_1by1(layer4_scaled, num_classes)
    
    # Skip layers by adding layer 7 upsample and scaled layer 4, upsample by 2
    skip_layer1 = tf.add(layer7_upsample, layer4_conv_1by1)
    skip_layer1_upsample = upsample(skip_layer1, 4, (2, 2), num_classes)
    
    # layer 3
    layer3_scaled = tf.multiply(vgg_layer3_out, 0.0001)
    layer3_conv_1by1 = conv_1by1(layer3_scaled, num_classes)
    
    # Skip layers by adding layer 3 with skip layer1 unsample
    skip_layer2 = tf.add(skip_layer1_upsample, layer3_conv_1by1)
    
    # Output layer by upsample by factor 8
    output = upsample(skip_layer2, 16, (8, 8), num_classes)
    
    return output
tests.test_layers(layers)

def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    # Using Adam optimizer as standard choice
    # Define loss
    # make logits a 2D tensor where each row represents pixels of an image and each column a class
    # original 4D tensor contains 2D dimension of image + 1D for filter numbers + 1D for batch size
    logits_2D = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label_2D = tf.reshape(correct_label, (-1,num_classes))
    # Define loss
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=correct_label_2D, logits=logits_2D)
    cross_entropy_loss = tf.reduce_mean(cross_entropy)
    # regulization constant, choose an appropriate one.
    # reference https://stackoverflow.com/questions/37107223/how-to-add-regularizations-in-tensorflow
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) 
    reg_constant = 0.01  
    # This is a list of the individual loss values, so we still need to sum them up.
    total_loss = tf.add(cross_entropy_loss,  reg_constant * sum(regularization_losses), name='total_loss') # Using total loss
    
    # Define optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    
    # Define train op to minimize loss. op is the object of operation
    train_op = optimizer.minimize(total_loss)

    return logits_2D, train_op, total_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    iteration = 0
    sess.run(tf.global_variables_initializer())
    
    print("Training of fully convolutional networks (decoder)")
    for epoch in range(epochs):
        for image, label in get_batches_fn(batch_size):
            # Training
            iteration = iteration + 1
            _, loss = sess.run([train_op, cross_entropy_loss], 
                     feed_dict = {input_image:image, correct_label:label, 
                                   keep_prob:0.5, learning_rate: 1e-4})
            print("Epoch: {}, batch: {}, loss: {}".format(epoch+1, iteration, loss))
        print()
    
tests.test_train_nn(train_nn)


def run():
    # This function creates template to 
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)
    
    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        epochs = 50
        batch_size = 10
        # Create placeholder for labels and learning rate
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        # Load vgg and establish decoder
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        layer_output = layers(layer3_out, layer4_out, layer7_out, num_classes)
        # Define label logits, minimization node and loss
        logits, train_op, cross_entropy_loss = optimize(layer_output, correct_label, learning_rate, num_classes)
        sess.run(tf.global_variables_initializer())
        
        # TODO: Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
                 correct_label, keep_prob, learning_rate)
        
        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video, using Term1's video


if __name__ == '__main__':
    run()

    
    
    
    
    
    
    