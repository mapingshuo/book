# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

from __future__ import print_function

import logging
import os
import argparse
import paddle
import paddle.fluid as fluid
import numpy
import sys
from vgg import vgg_bn_drop
from resnet import resnet_cifar10
import my_optimizer
import cifar_preprocess
import random

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser("image_classification")
    parser.add_argument(
        '--enable_ce',
        action='store_true',
        help='If set, run the task with continuous evaluation logs.')
    parser.add_argument(
        '--use_gpu', type=bool, default=0, help='whether to use gpu')
    parser.add_argument(
        '--num_epochs', type=int, default=1, help='number of epoch')
    parser.add_argument(
        '--learning_rate', type=float, default=0.1, help='learning rate')
    parser.add_argument(
        '--weight_decay', type=float, default=0.001, help='weight decay')
    parser.add_argument(
        '--optimizer_name', type=str, default='SGD', help='optimizer: SGD or Lookahead')
    parser.add_argument(
        '--alpha', type=float, default=0.5, help='alpha in Lookahead Optimizer')
    parser.add_argument(
        '--k', type=int, default=5, help='k in Lookahead Optimizer')
    parser.add_argument(
        '--random_seed', type=int, default=70, help='random seed for shuffle and network initialize')

    args = parser.parse_args()
    return args

def inference_network():
    # The image is 32 * 32 with RGB representation.
    data_shape = [3, 32, 32]
    images = fluid.layers.data(name='pixel', shape=data_shape, dtype='float32')

    predict = resnet_cifar10(images, depth=20)
    # predict = vgg_bn_drop(images) # un-comment to use vgg net
    return predict


def train_network(predict):
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(cost)
    accuracy = fluid.layers.accuracy(input=predict, label=label)
    return [avg_cost, accuracy]


def optimizer_program(name, learning_rate, weight_decay, alpha=0.5, k=10):
    logger.info('optimizer name: ' + name)
    logger.info('weight_decay: ' + str(weight_decay))
    logger.info('learning_rate: ' + str(learning_rate))
  
    boundaries = [39062, 58594]
    values = [learning_rate, learning_rate / 10, learning_rate / 100]     
    #boundaries = [23438, 46875]
    #values = [learning_rate, learning_rate / 5, learning_rate / 25]
    
    logger.info('boundaries: ' + str(boundaries))
    logger.info('boundaries_value: ' + str(values))

    learning_rate=fluid.layers.piecewise_decay(boundaries, values)
    regularization=fluid.regularizer.L2DecayRegularizer(regularization_coeff=weight_decay)

    if name == 'SGD':
        opti = fluid.optimizer.Momentum(
		learning_rate=learning_rate, momentum=0.9, regularization=regularization)
    elif name == 'Lookahead':
        logger.info('alpha: ' + str(alpha))
        logger.info('k: ' + str(k))
        sgd = fluid.optimizer.Momentum(
		learning_rate=learning_rate, momentum=0.9, regularization=regularization)
        opti = my_optimizer.LookaheadOptimizer(sgd, alpha=alpha, k=k, ignore_embed=False)
    else:
	print("No such optimizer: ", name)
    return opti

def train(use_cuda, params_dirname):
    print("CUDA:", use_cuda)
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    BATCH_SIZE = 128
    logger.info("random_seed: " + str(args.random_seed))
    logger.info("batch size: " + str(BATCH_SIZE))
    random.seed(args.random_seed)

    if args.enable_ce:
        print("Enable CE")
        train_reader = paddle.batch(
            cifar_preprocess.preprocess(paddle.dataset.cifar.train10()), 
            batch_size=BATCH_SIZE)
        test_reader = paddle.batch(
            cifar_preprocess.preprocess(paddle.dataset.cifar.test10()), 
	    batch_size=BATCH_SIZE)
    else:
        print("Closed CE")
        #test_reader = paddle.batch(
        #    cifar_preprocess.preprocess(paddle.dataset.cifar.test10()), 
	#    batch_size=BATCH_SIZE)
        train_reader = paddle.batch(
            paddle.reader.shuffle(
                cifar_preprocess.preprocess(paddle.dataset.cifar.train10()), 
		buf_size=128 * 500),
            batch_size=BATCH_SIZE)

    feed_order = ['pixel', 'label']

    main_program = fluid.default_main_program()
    start_program = fluid.default_startup_program()

    if args.enable_ce:
        main_program.random_seed = 90
        start_program.random_seed = 90

    main_program.random_seed = args.random_seed
    start_program.random_seed = args.random_seed
    predict = inference_network()
    avg_cost, acc = train_network(predict)

    # Test program
    test_program = main_program.clone(for_test=True)
    optimizer = optimizer_program(args.optimizer_name, learning_rate=args.learning_rate, alpha=args.alpha, k=args.k, weight_decay=args.weight_decay)
    optimizer.minimize(avg_cost)

    exe = fluid.Executor(place)

    EPOCH_NUM = args.num_epochs

    # For training test cost
    def train_test(program, reader):
        count = 0
        feed_var_list = [
            program.global_block().var(var_name) for var_name in feed_order
        ]
        feeder_test = fluid.DataFeeder(feed_list=feed_var_list, place=place)
        test_exe = fluid.Executor(place)
        accumulated = len([avg_cost, acc]) * [0]
        for tid, test_data in enumerate(reader()):
            avg_cost_np = test_exe.run(
                program=program,
                feed=feeder_test.feed(test_data),
                fetch_list=[avg_cost, acc])
            accumulated = [
                x[0] + x[1][0] for x in zip(accumulated, avg_cost_np)
            ]
            count += 1

        return [x / count for x in accumulated]

    # main train loop.
    def train_loop():
        feed_var_list_loop = [
            main_program.global_block().var(var_name) for var_name in feed_order
        ]
        feeder = fluid.DataFeeder(feed_list=feed_var_list_loop, place=place)
        exe.run(start_program)
    
        with open("main_program", "w") as f:
            f.write(str(main_program))

        step = 0
        for pass_id in range(EPOCH_NUM):
            for step_id, data_train in enumerate(train_reader()):
		lr_var = main_program.global_block().var("learning_rate")
                avg_loss, avg_acc, learning_rate = exe.run(
                    main_program,
                    feed=feeder.feed(data_train),
                    fetch_list=[avg_cost, acc, lr_var])
                if step_id % 100 == 0:
                    print("\nPass %d, Batch %d, Cost %f, Acc %f, learning rate %f" % (
                        step_id, pass_id, avg_loss, avg_acc, learning_rate))
                else:
                    sys.stdout.write('.')
                    sys.stdout.flush()
                step += 1
            avg_cost_test, accuracy_test = train_test(
                test_program, reader=train_reader)
            print('\nTest with Pass %d, Loss %f, Acc %f' % (
                pass_id, avg_cost_test, accuracy_test))

            if params_dirname is not None:
                fluid.io.save_inference_model(params_dirname, ["pixel"],
                                              [predict], exe)

            if args.enable_ce and pass_id == EPOCH_NUM - 1:
                print("kpis\ttrain_cost\t%f" % avg_loss_value[0])
                print("kpis\ttrain_acc\t%f" % avg_loss_value[1])
                print("kpis\ttest_cost\t%f" % avg_cost_test)
                print("kpis\ttest_acc\t%f" % accuracy_test)

    train_loop()


def infer(use_cuda, params_dirname=None):
    from PIL import Image
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    inference_scope = fluid.core.Scope()

    def load_image(infer_file):
        im = Image.open(infer_file)
        im = im.resize((32, 32), Image.ANTIALIAS)

        im = numpy.array(im).astype(numpy.float32)
        # The storage order of the loaded image is W(width),
        # H(height), C(channel). PaddlePaddle requires
        # the CHW order, so transpose them.
        im = im.transpose((2, 0, 1))  # CHW
        im = im / 255.0

        # Add one dimension to mimic the list format.
        im = numpy.expand_dims(im, axis=0)
        return im

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    img = load_image(cur_dir + '/image/dog.png')

    with fluid.scope_guard(inference_scope):
        # Use fluid.io.load_inference_model to obtain the inference program desc,
        # the feed_target_names (the names of variables that will be feeded
        # data using feed operators), and the fetch_targets (variables that
        # we want to obtain data from using fetch operators).
        [inference_program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(params_dirname, exe)

        # The input's dimension of conv should be 4-D or 5-D.
        # Use inference_transpiler to speedup
        inference_transpiler_program = inference_program.clone()
        t = fluid.transpiler.InferenceTranspiler()
        t.transpile(inference_transpiler_program, place)

        # Construct feed as a dictionary of {feed_target_name: feed_target_data}
        # and results will contain a list of data corresponding to fetch_targets.
        results = exe.run(
            inference_program,
            feed={feed_target_names[0]: img},
            fetch_list=fetch_targets)

        transpiler_results = exe.run(
            inference_transpiler_program,
            feed={feed_target_names[0]: img},
            fetch_list=fetch_targets)

        assert len(results[0]) == len(transpiler_results[0])
        for i in range(len(results[0])):
            numpy.testing.assert_almost_equal(
                results[0][i], transpiler_results[0][i], decimal=5)

        # infer label
        label_list = [
            "airplane", "automobile", "bird", "cat", "deer", "dog", "frog",
            "horse", "ship", "truck"
        ]

        print("infer results: %s" % label_list[numpy.argmax(results[0])])


def main(use_cuda):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return
    save_path = "image_classification_resnet.inference.model"

    train(use_cuda=use_cuda, params_dirname=save_path)

    infer(use_cuda=use_cuda, params_dirname=save_path)


if __name__ == '__main__':
    # For demo purpose, the training runs on CPU
    # Please change accordingly.
    args = parse_args()
    use_cuda = args.use_gpu
    main(use_cuda)
