class Path(object):
    @staticmethod
    def db_dir(database):
        if database == 'DMD-lite-70':
            # folder that contains class labels
            root_dir = './processed_dataset/DMD-clips-70'

            # Save preprocess data into output_dir
            output_dir = './data/DMD-clips-70'

            return root_dir, output_dir

        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def model_dir():
        return './code/models/vit_base_patch16_224_in21k.pth'