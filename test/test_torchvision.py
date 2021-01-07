
print('testing torchvision...')

import torchvision
print('torchvision version: ' + str(torchvision.__version__) + '\n')

import time
import argparse

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
    
parser = argparse.ArgumentParser()

parser.add_argument('--data', type=str, default='/tmp/ILSVRC2012_img_val_subset_5k')
parser.add_argument('--resolution', default=224, type=int, metavar='N',
                    help='input NxN image resolution of model (default: 224x224) '
                         'note than Inception models should use 299x299')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size (default: 8)') 
parser.add_argument('-p', '--print-freq', default=25, type=int,
                    metavar='N', help='print frequency (default: 10)')                    
parser.add_argument('-t', '--test-threshold', default=-10.0, type=float,
                    metavar='N', help='maximum passing delta between trained model top-1 accuracy  (default is -10%)')
parser.add_argument("--use-cuda", action="store_true", help='use CUDA (otherwise CPU-only)')  
                  
args = parser.parse_args()
    
print('torchvision classification models: ' + ' | '.join(model_names) + '\n')

def load_data(root):
    return torch.utils.data.DataLoader(
            datasets.ImageFolder(root, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(args.resolution),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        
def test_model(model_info, data_loader):

    model_name = model_info[0]
    model_top1 = 100.0 - model_info[1]
    model_top5 = 100.0 - model_info[2]
    
    print("\n")
    print("---------------------------------------------")
    print("-- " + model_name)
    print("---------------------------------------------")
    
    print("loading model '{:s}'".format(model_name))
    model = models.__dict__[model_name](pretrained=True, progress=False).eval()
    
    if args.use_cuda:
        model = model.cuda()
        
    print("loaded model '{:s}'\n".format(model_name))
    
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(data_loader),
        [batch_time, top1, top5],
        prefix=model_name.ljust(9))
        
    with torch.no_grad():
        end = time.time()
        
        for i, (images, target) in enumerate(data_loader):
            if args.use_cuda:
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
            
            # compute output
            output = model(images)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, min(5, len(data_loader.dataset.classes))))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
            
        top1_avg = top1.avg.item()
        top5_avg = top5.avg.item()
        top1_delta = top1_avg - model_top1
        top5_delta = top5_avg - model_top5
        
        passing = (top1_delta >= args.test_threshold)
        images_per_second = 1.0 / batch_time.avg * args.batch_size
        
        results = model_name, top1_avg, model_top1, top1_delta, top5_avg, model_top5, top5_delta, images_per_second, passing
        print_results(results)
        
    return results
    
def print_results(results):
    print(' ')
    print(results[0])
    print('   * Acc@1 {:.3f}  Expected {:.3f}   Delta {:.3f}'.format(results[1], results[2], results[3]))
    print('   * Acc@5 {:.3f}  Expected {:.3f}   Delta {:.3f}'.format(results[4], results[5], results[6]))
    print('   * Images/sec  {:.3f}'.format(results[7]))
    print('   * {:s}'.format('PASS' if results[8] else 'FAIL'))

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('  '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
        
        
if __name__ == '__main__':  
    
    print('using {:s}'.format("CUDA" if args.use_cuda else "CPU"))     
    print('loading dataset from {:s}\n'.format(args.data))
    data_loader = load_data(args.data)
    print('dataset classes: {:d}'.format(len(data_loader.dataset.classes)))
    print('dataset images:  {:d}'.format(len(data_loader.dataset)))
    print('batch size:      {:d}'.format(args.batch_size))
    
    # model name, expected top-1 error, expected top-5 error
    # the trained errors come from:  https://pytorch.org/docs/stable/torchvision/models.html
    model_info = [('alexnet', 43.45, 20.91),
                  ('googlenet', 30.22, 10.47),
                  ('resnet18', 30.24, 10.92),
                  ('resnet50', 23.85, 7.13)]
              
    results = []
    
    for model in model_info:
        results.append(test_model(model, data_loader))
        
    print("\n")
    print("---------------------------------------------")
    print("-- Summary")
    print("---------------------------------------------")
    
    num_passing = 0
    
    for result in results:
        print_results(result)
        
        if result[8] is True:
            num_passing += 1
            
    print("\nModel tests passing:  {:d} / {:d}".format(num_passing, len(model_info)))
    print('torchvision {:s}\n'.format('OK' if num_passing == len(model_info) else 'FAIL'))
