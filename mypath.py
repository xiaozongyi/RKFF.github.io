class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return '/hy-tmp/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return './data/benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return '/hy-tmp/cityscapes/'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return './data/coco/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
