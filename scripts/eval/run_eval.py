import configargparse
import distutils.util
from eval import main

parser = configargparse.ArgumentParser(description='Eval network for few-shot learning')

parser.add_argument('--model.model_path', type=str, metavar='PATH',
                    help="location of pretrained model to evaluate")

parser.add_argument('--data.test_way', type=int, metavar='N', default=0,
                    help="number of classes per episode in test. 0 means same as model's data.test_way")
parser.add_argument('--data.test_shot', type=int, metavar='N', default=0,
                    help="number of support examples per class in test. 0 means same as model's data.shot")
parser.add_argument('--data.test_query', type=int, metavar='N', default=0,
                    help="number of query examples per class in test. 0 means same as model's data.query")

parser.add_argument('--data.test_episodes', type=int, metavar='N', default=1000,
                    help="number of test episodes per epoch")

parser.add('--data.cuda', type=distutils.util.strtobool, metavar='BOOL', help="run in CUDA mode", default=True)

parser.add('--data.gpu', type=int, metavar='N', help="GPU device (starts from 0)", default=0)

opts = vars(parser.parse_args())

main(opts)
