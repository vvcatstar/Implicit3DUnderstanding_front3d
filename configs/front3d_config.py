class Config(object):
    def __init__(self, dataset):
        """
        Configuration of data paths.
        """
        self.dataset = dataset
        self.root_path = '/home/zyw/data/dataset/for_yaowei_front3d_withmodel'
        self.model_path = '/home/zyw/data/dataset/for_yaowei_front3d_withmodel/manifold_remesh_model/'
        self.save_path = '/home/zyw/data/dataset/for_yaowei_front3d_withmodel/ldif'
        
        self.train_split = self.root_path + '/split/train/all_subset.json'
        self.test_split = self.root_path + '/split/test/all_subset.json'

        if dataset == 'front3d':
            # self.metadata_file = self.metadata_path + '/pix3d.json'
            self.classnames = ['desk', 'table', 'cabinet', 'bed', 'chair', 'sofa', 'bookshelf', 'night_stand', 'dresser']
number_pnts_on_template = 2562
neighbors = 30