'''
Read images practice
1.Read images: input output
2.Get the diff between src and dst, and do the HE
3.Save diff as JPG

ref:
https://www.tensorflow.org/programmers_guide/datasets
https://www.hksilicon.com/articles/1471783

[TODO] Add histogram equalization.
'''
__author__ = 'channai'

import tensorflow as tf
import os
import logging
import argparse
import ntpath
import skimage.io

logging.basicConfig(filename='example.log', filemode='w', level=logging.DEBUG)

def reader(src,dst):
    src_string = tf.read_file(src)
    dst_string = tf.read_file(dst)
    src_decoded = tf.cast(tf.image.decode_image(src_string), tf.int32)
    dst_decoded = tf.cast(tf.image.decode_image(dst_string), tf.int32)
    return src_decoded, dst_decoded

class FileIO:
    def __init__(self, src_dir, dst_dir, output_dir):
        logging.info('__init__ start')
        #check the three dir exist
        if os.path.exists(src_dir) and os.path.exists(dst_dir):
            self.src_dir = src_dir
            self.dst_dir = dst_dir
            self.output_dir = output_dir

            #list dir
            self.src_files = os.listdir(src_dir)
            self.dst_files = os.listdir(dst_dir)
            self.src_files.sort()
            self.dst_files.sort()
            self.src_files = map(lambda s: src_dir+'/'+s, self.src_files)
            self.dst_files = map(lambda s: dst_dir+'/'+s, self.dst_files)

            #dump
            logging.info('src files:')
            for f in self.src_files:
                logging.info(f)

            logging.info('dst files:')
            for f in self.dst_files:
                logging.info(f)

            if len(self.src_files) != len(self.dst_files):
                raise ValueError

            #check output dir
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        else:
            raise ValueError

        logging.info('__init__ end')

    def diff(self):
        logging.info('diff start')
        _src = tf.constant(self.src_files)
        _dst = tf.constant(self.dst_files)

        dataset = tf.data.Dataset.from_tensor_slices((_src,_dst))
        dataset = dataset.map(reader)

        iterator = dataset.make_initializable_iterator()
        next_src, next_dst = iterator.get_next()
        proc_diff = tf.abs(tf.subtract(next_src,next_dst))

        with tf.Session() as sess:
            sess.run(iterator.initializer)
            for f in self.src_files:
                _diff = sess.run(proc_diff)
                output_path = self.output_dir + '/' + ntpath.basename(f)
                skimage.io.imsave(output_path, _diff)
        logging.info('diff end')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('src_dir', default=None, help='path to the src dir')
    parser.add_argument('dst_dir', default=None, help='path to the dst dir')
    parser.add_argument('output_dir', default=None, help='path to the output dir')
    args = parser.parse_args()
    fileIO = FileIO(args.src_dir,args.dst_dir,args.output_dir)
    fileIO.diff()
