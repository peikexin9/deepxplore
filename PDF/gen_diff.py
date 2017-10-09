'''
usage: python gen_diff.py -h
'''

from __future__ import print_function

import argparse

from mimicus.tools.featureedit import FeatureDescriptor
from scipy.misc import imsave

from pdf_models import *
from utils import *

# read the parameter
# argument parsing
parser = argparse.ArgumentParser(
    description='Main function for difference-inducing input generation in VirusTotal/Contagio dataset')
parser.add_argument('weight_diff', help="weight hyperparm to control differential behavior", type=float)
parser.add_argument('weight_nc', help="weight hyperparm to control neuron coverage", type=float)
parser.add_argument('step', help="step size of gradient descent", type=float)
parser.add_argument('seeds', help="number of seeds of input", type=int)
parser.add_argument('grad_iterations', help="number of iterations of gradient descent", type=int)
parser.add_argument('threshold', help="threshold for determining neuron activated", type=float)
parser.add_argument('-t', '--target_model', help="target model that we want it predicts differently",
                    choices=[0, 1, 2], default=0, type=int)

args = parser.parse_args()

X_test, _, names = datasets.csv2numpy('./dataset/test.csv')
X_test = X_test.astype('float32')
num_features = X_test.shape[1]
feat_names = FeatureDescriptor.get_feature_names()
incre_idx, incre_decre_idx = init_feature_constraints(feat_names)

# define input tensor as a placeholder
input_tensor = Input(shape=(num_features,))

# load multiple models sharing same input tensor
K.set_learning_phase(0)
model1 = Model1(input_tensor=input_tensor, load_weights=True)
model2 = Model2(input_tensor=input_tensor, load_weights=True)
model3 = Model3(input_tensor=input_tensor, load_weights=True)
# init coverage table
model_layer_dict1, model_layer_dict2, model_layer_dict3 = init_coverage_tables(model1, model2, model3)

# ==============================================================================================
# start gen inputs
for _ in xrange(args.seeds):
    idx = random.randint(0, len(X_test))
    gen_pdf = np.expand_dims(X_test[idx], axis=0)
    orig_pdf = gen_pdf.copy()
    # first check if input already induces differences
    label1, label2, label3 = np.argmax(model1.predict(gen_pdf)[0]), np.argmax(model2.predict(gen_pdf)[0]), np.argmax(
        model3.predict(gen_pdf)[0])
    if not label1 == label2 == label3:
        print(bcolors.OKGREEN + 'input already causes different outputs: {}, {}, {}'.format(label1, label2,
                                                                                            label3) + bcolors.ENDC)

        update_coverage(gen_pdf, model1, model_layer_dict1, args.threshold)
        update_coverage(gen_pdf, model2, model_layer_dict2, args.threshold)
        update_coverage(gen_pdf, model3, model_layer_dict3, args.threshold)

        print(bcolors.OKGREEN + 'covered neurons percentage %d neurons %.3f, %d neurons %.3f, %d neurons %.3f'
              % (len(model_layer_dict1), neuron_covered(model_layer_dict1)[2], len(model_layer_dict2),
                 neuron_covered(model_layer_dict2)[2], len(model_layer_dict3),
                 neuron_covered(model_layer_dict3)[2]) + bcolors.ENDC)
        averaged_nc = (neuron_covered(model_layer_dict1)[0] + neuron_covered(model_layer_dict2)[0] +
                       neuron_covered(model_layer_dict3)[0]) / float(
            neuron_covered(model_layer_dict1)[1] + neuron_covered(model_layer_dict2)[1] +
            neuron_covered(model_layer_dict3)[
                1])
        print(bcolors.OKGREEN + 'averaged covered neurons %.3f' % averaged_nc + bcolors.ENDC)

        # save the result to disk
        with open('generated_inputs', 'a') as f:
            f.write(
                'Already causes differences: name: {}, label1:{}, label2: {}, label3: {}\n'.format(names[idx], label1,
                                                                                                   label2, label3))
        continue

    # if all turning angles roughly the same
    orig_label = label1
    layer_name1, index1 = neuron_to_cover(model_layer_dict1)
    layer_name2, index2 = neuron_to_cover(model_layer_dict2)
    layer_name3, index3 = neuron_to_cover(model_layer_dict3)

    # construct joint loss function
    if args.target_model == 0:
        loss1 = -args.weight_diff * K.mean(model1.get_layer('before_softmax').output[..., orig_label])
        loss2 = K.mean(model2.get_layer('before_softmax').output[..., orig_label])
        loss3 = K.mean(model3.get_layer('before_softmax').output[..., orig_label])
    elif args.target_model == 1:
        loss1 = K.mean(model1.get_layer('before_softmax').output[..., orig_label])
        loss2 = -args.weight_diff * K.mean(model2.get_layer('before_softmax').output[..., orig_label])
        loss3 = K.mean(model3.get_layer('before_softmax').output[..., orig_label])
    elif args.target_model == 2:
        loss1 = K.mean(model1.get_layer('before_softmax').output[..., orig_label])
        loss2 = K.mean(model2.get_layer('before_softmax').output[..., orig_label])
        loss3 = -args.weight_diff * K.mean(model3.get_layer('before_softmax').output[..., orig_label])
    loss1_neuron = K.mean(model1.get_layer(layer_name1).output[..., index1])
    loss2_neuron = K.mean(model2.get_layer(layer_name2).output[..., index2])
    loss3_neuron = K.mean(model3.get_layer(layer_name3).output[..., index3])
    layer_output = (loss1 + loss2 + loss3) + args.weight_nc * (loss1_neuron + loss2_neuron + loss3_neuron)

    # for adversarial image generation
    final_loss = K.mean(layer_output)

    # we compute the gradient of the input picture wrt this loss
    grads = normalize(K.gradients(final_loss, input_tensor)[0])

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_tensor], [loss1, loss2, loss3, loss1_neuron, loss2_neuron, loss3_neuron, grads])

    # we run gradient ascent for 20 steps
    for iters in xrange(args.grad_iterations):
        loss_value1, loss_value2, loss_value3, loss_neuron1, loss_neuron2, loss_neuron3, grads_value = iterate(
            [gen_pdf])
        grads_value = constraint(grads_value, incre_idx, incre_decre_idx)  # constraint the gradients value

        gen_pdf += grads_value * args.step
        label1, label2, label3 = np.argmax(model1.predict(gen_pdf)[0]), np.argmax(
            model2.predict(gen_pdf)[0]), np.argmax(model3.predict(gen_pdf)[0])

        if not label1 == label2 == label3:
            update_coverage(gen_pdf, model1, model_layer_dict1, args.threshold)
            update_coverage(gen_pdf, model2, model_layer_dict2, args.threshold)
            update_coverage(gen_pdf, model3, model_layer_dict3, args.threshold)

            print(bcolors.OKGREEN + 'covered neurons percentage %d neurons %.3f, %d neurons %.3f, %d neurons %.3f'
                  % (len(model_layer_dict1), neuron_covered(model_layer_dict1)[2], len(model_layer_dict2),
                     neuron_covered(model_layer_dict2)[2], len(model_layer_dict3),
                     neuron_covered(model_layer_dict3)[2]) + bcolors.ENDC)
            averaged_nc = (neuron_covered(model_layer_dict1)[0] + neuron_covered(model_layer_dict2)[0] +
                           neuron_covered(model_layer_dict3)[0]) / float(
                neuron_covered(model_layer_dict1)[1] + neuron_covered(model_layer_dict2)[1] +
                neuron_covered(model_layer_dict3)[
                    1])
            print(bcolors.OKGREEN + 'averaged covered neurons %.3f' % averaged_nc + bcolors.ENDC)

            # save the result to disk
            with open('generated_inputs', 'a') as f:
                f.write(
                    'name: {}, label1:{}, label2: {}, label3: {}\n'.format(names[idx], label1, label2, label3))
                f.write('changed features: {}\n\n'.format(features_changed(gen_pdf, orig_pdf, feat_names)))
            break
