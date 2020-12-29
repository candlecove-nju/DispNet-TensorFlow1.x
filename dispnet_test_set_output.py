import tensorflow as tf
import numpy as np
from PIL import Image
import mkdir as get_path
import IO
import matplotlib.pyplot as plt

IMAGE_SIZE_X = 768
IMAGE_SIZE_Y = 384

BATCH_SIZE = 1
TRAINING_ROUNDS = 1
#LEARNING_RATE = 1e-2
TRAIN_NUM = 4160 # 130*32
TEST_NUM = 192 # 6*32

initial_learning_rate = 1e-2
decay_steps = 10000 
decay_rate = 0.1

IMAGE_DIR = './frames_cleanpass'
GT_DIR = './driving__disparity/disparity'
LOGS_DIR = './logs'
RUNNING_LOGS_DIR = './running_logs'
OUTPUT_DIR = './output'

def weight_variable(shape):
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    return tf.Variable(initializer(shape=shape), name='weight')

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name='b')

def conv2d(x, W, strides):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=strides, padding='SAME')

def upconv2d_2x2(x, W, output_shape):
    return tf.nn.conv2d_transpose(x, W, output_shape=output_shape, strides=[1, 2, 2, 1], padding='SAME');

def batch_norm(x):
    return tf.contrib.layers.batch_norm(x, center=True, scale=True, is_training=True)

def upsample(disp):
    m, n = disp.shape[1:3]      
    return tf.image.resize_bilinear(disp, [2*m, 2*n])

def loss(pre, gt):
    loss = tf.sqrt(tf.reduce_mean(tf.square(pre - gt)))
    return loss

def _norm(img):
    return (img - np.mean(img)) / np.std(img)


def model(combine_image, ground_truth):
  # conv1
  # input dims: (BATCH_SIZE)*384*768*6
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([7,7, 6,64]) 
    b_conv1 = bias_variable([64])
    h_conv1 = tf.nn.relu(batch_norm(conv2d(combine_image, W_conv1, [1, 2, 2 ,1]) + b_conv1)) 
    # output dims: 192*384*64

  # conv2
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5,5, 64,128]) 
    b_conv2 = bias_variable([128])
    h_conv2 = tf.nn.relu(batch_norm(conv2d(h_conv1, W_conv2, [ 1, 2, 2, 1]) + b_conv2))
    # output dims: 96*192*128
    
  # conv3a
  with tf.name_scope('conv3a'):
    W_conv3a = weight_variable([5,5, 128,256]) 
    b_conv3a = bias_variable([256])
    h_conv3a = tf.nn.relu(batch_norm(conv2d(h_conv2, W_conv3a, [1, 2, 2, 1]) + b_conv3a)) 
    # output dims: 48*96*256
    
  # conv3b
  with tf.name_scope('conv3b'):
    W_conv3b = weight_variable([3,3, 256,256]) 
    b_conv3b = bias_variable([256])
    h_conv3b = tf.nn.relu(batch_norm(conv2d(h_conv3a, W_conv3b, [1, 1, 1, 1]) + b_conv3b)) 
    # output dims: 48*96*256

  # conv4a
  with tf.name_scope('conv4a'):
    W_conv4a = weight_variable([3,3, 256,512]) 
    b_conv4a = bias_variable([512])
    h_conv4a = tf.nn.relu(batch_norm(conv2d(h_conv3b, W_conv4a, [1, 2, 2, 1]) + b_conv4a)) 
    # output dims: 24*48*512
  # conv4b
  with tf.name_scope('conv4b'):
    W_conv4b = weight_variable([3,3, 512,512]) 
    b_conv4b = bias_variable([512])
    h_conv4b = tf.nn.relu(batch_norm(conv2d(h_conv4a, W_conv4b, [1, 1, 1, 1]) + b_conv4b)) 
    # output dims: 24*48*512

  # conv5a
  with tf.name_scope('conv5a'):
    W_conv5a = weight_variable([3,3, 512,512]) 
    b_conv5a = bias_variable([512])
    h_conv5a = tf.nn.relu(batch_norm(conv2d(h_conv4b, W_conv5a, [1, 2, 2, 1]) + b_conv5a)) 
    # output dims: 12*24*512
  # conv5b
  with tf.name_scope('conv5b'):
    W_conv5b = weight_variable([3,3, 512,512]) 
    b_conv5b = bias_variable([512])
    h_conv5b = tf.nn.relu(batch_norm(conv2d(h_conv5a, W_conv5b, [ 1, 1, 1, 1]) + b_conv5b)) 
    # output dims: 12*24*512

  # conv6a
  with tf.name_scope('conv6a'):
    W_conv6a = weight_variable([3,3, 512,1024]) 
    b_conv6a = bias_variable([1024])
    h_conv6a = tf.nn.relu(batch_norm(conv2d(h_conv5b, W_conv6a, [1, 2, 2, 1]) + b_conv6a)) 
    # output dims: 6*12*1024
    
  # conv6b
  with tf.name_scope('conv6b'):
    W_conv6b = weight_variable([3,3, 1024,1024]) 
    b_conv6b = bias_variable([1024])
    h_conv6b = tf.nn.relu(batch_norm(conv2d(h_conv6a, W_conv6b, [1, 1, 1, 1]) + b_conv6b)) 
    # output dims: 6*12*1024

  # pr6 + loss6
  with tf.name_scope('pr6_loss6'):
    W_pr6 = weight_variable([3,3, 1024,1]) 
    b_pr6 = bias_variable([1])
    pr6 = tf.nn.relu(batch_norm(conv2d(h_conv6b, W_pr6, [1, 1, 1, 1]) + b_pr6)) 
    gt6 = tf.nn.avg_pool(ground_truth, ksize=[1,64,64,1], strides=[1,64,64,1], padding='SAME', name='gt6')
    loss6 = loss(pr6, gt6)
    # pr6 dims: 6*12*1

  # upconv5
  with tf.name_scope('upconv5'):
    W_upconv5 = weight_variable([4,4, 512,1024]) 
    b_upconv5 = bias_variable([512])
    h_upconv5 = tf.nn.relu(batch_norm(upconv2d_2x2(h_conv6b,  W_upconv5, [BATCH_SIZE, np.int32(IMAGE_SIZE_Y/32), np.int32(IMAGE_SIZE_X/32), 512]) + b_upconv5)) 
    # output dims: 12*24*512

  # iconv5
  with tf.name_scope('iconv5'):
    W_iconv5 = weight_variable([3,3, 1025,512]) 
    b_iconv5 = bias_variable([512])
    h_iconv5 = tf.nn.relu(batch_norm(conv2d(tf.concat([h_upconv5, h_conv5b, upsample(pr6)], 3), W_iconv5, [1, 1, 1, 1]) + b_iconv5)) 
    # output dims: 12*24*512

  # pr5 + loss5
  with tf.name_scope('pr5_loss5'):
    W_pr5 = weight_variable([3,3, 512,1]) 
    b_pr5 = bias_variable([1])
    pr5 = tf.nn.relu(batch_norm(conv2d(h_iconv5, W_pr5, [1, 1, 1, 1]) + b_pr5))
    gt5 = tf.nn.avg_pool(ground_truth, ksize=[1,32,32,1], strides=[1,32,32,1], padding='SAME', name='gt5')
    loss5 = loss(pr5, gt5)
    # pr5 dims: 12*24*1

  # upconv4
  with tf.name_scope('upconv4'):
    W_upconv4 = weight_variable([4,4, 256, 512])
    #[height, width, output_channels, in_channels]
    b_upconv4 = bias_variable([256])
    h_upconv4 = tf.nn.relu(batch_norm(upconv2d_2x2(h_iconv5, W_upconv4, [BATCH_SIZE, np.int32(IMAGE_SIZE_Y/16), np.int32(IMAGE_SIZE_X/16), 256]) + b_upconv4))
    # output dims: 24*48*256

  # iconv4
  with tf.name_scope('iconv4'):
    W_iconv4 = weight_variable([3,3, 769,256]) 
    b_iconv4 = bias_variable([256])
    h_iconv4 = tf.nn.relu(batch_norm(conv2d(tf.concat([h_upconv4, h_conv4b, upsample(pr5)], 3), W_iconv4, [ 1, 1, 1, 1]) + b_iconv4))
    # output dims: 24*48*256

  # pr4 + loss4
  with tf.name_scope('pr4_loss4'):
    W_pr4 = weight_variable([3,3, 256,1]) 
    b_pr4 = bias_variable([1])
    pr4 = tf.nn.relu(batch_norm(conv2d(h_iconv4, W_pr4, [1, 1, 1, 1]) + b_pr4))    
    gt4 = tf.nn.avg_pool(ground_truth, ksize=[1,16,16,1], strides=[1,16,16,1], padding='SAME', name='gt4')
    loss4 = loss(pr4, gt4)
    # pr4 dims: 24*48*1
    
  # upconv3
  with tf.name_scope('upconv3'):
    W_upconv3 = weight_variable([4,4,128, 256]) 
    #[height, width, output_channels, in_channels]
    b_upconv3 = bias_variable([128])
    h_upconv3 = tf.nn.relu(batch_norm(upconv2d_2x2(h_iconv4, W_upconv3, [BATCH_SIZE, np.int32(IMAGE_SIZE_Y/8), np.int32(IMAGE_SIZE_X/8), 128]) + b_upconv3)) 
    # output dims: 48*96*128

  # iconv3
  with tf.name_scope('iconv3'):
    W_iconv3 = weight_variable([3,3, 385,128]) 
    b_iconv3 = bias_variable([128])
    h_iconv3 = tf.nn.relu(batch_norm(conv2d(tf.concat([h_upconv3, h_conv3b, upsample(pr4)], 3), W_iconv3, [ 1, 1, 1, 1]) + b_iconv3)) 
    # output dims: 48*96*128

  # pr3 + loss3
  with tf.name_scope('pr3_loss3'):
    W_pr3 = weight_variable([3,3, 128,1]) 
    b_pr3 = bias_variable([1])
    pr3 = tf.nn.relu(batch_norm(conv2d(h_iconv3, W_pr3, [1, 1, 1, 1]) + b_pr3)) 
    gt3 = tf.nn.avg_pool(ground_truth, ksize=[1,8,8,1], strides=[1,8,8,1], padding='SAME', name='gt')
    loss3 = loss(pr3, gt3)
    # pr3 dims: 48*96*1

  # upconv2
  with tf.name_scope('upconv2'):
    W_upconv2 = weight_variable([4,4,64, 128]) 
    #[height, width, output_channels, in_channels]
    b_upconv2 = bias_variable([64])
    h_upconv2 = tf.nn.relu(batch_norm(upconv2d_2x2(h_iconv3, W_upconv2, [BATCH_SIZE, np.int32(IMAGE_SIZE_Y/4), np.int32(IMAGE_SIZE_X/4), 64]) + b_upconv2)) 
    # output dims: 96*192*64

  # iconv2
  with tf.name_scope('iconv2'):
    W_iconv2 = weight_variable([3,3, 193,64]) 
    b_iconv2 = bias_variable([64])
    h_iconv2 = tf.nn.relu(batch_norm(conv2d(tf.concat([h_upconv2, h_conv2, upsample(pr3)], 3), W_iconv2, [1, 1, 1, 1]) + b_iconv2)) 
    # output dims: 96*192*64

  # pr2 + loss2
  with tf.name_scope('pr2_loss2'):
    W_pr2 = weight_variable([3,3, 64,1]) 
    b_pr2 = bias_variable([1])
    pr2 = tf.nn.relu(batch_norm(conv2d(h_iconv2, W_pr2, [1, 1, 1, 1]) + b_pr2)) 
    gt2 = tf.nn.avg_pool(ground_truth, ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME', name='gt')
    loss2 = loss(pr2, gt2)
    # pr2 dims: 96*192*1

  # upconv1
  with tf.name_scope('upconv1'):
    W_upconv1 = weight_variable([4,4,32, 64]) 
    #[height, width, output_channels, in_channels]
    b_upconv1 = bias_variable([32])
    h_upconv1 = tf.nn.relu(batch_norm(upconv2d_2x2(h_iconv2, W_upconv1, [BATCH_SIZE, np.int32(IMAGE_SIZE_Y/2), np.int32(IMAGE_SIZE_X/2), 32]) + b_upconv1)) 
    # output dims: 192*384*32
    
  # iconv1
  with tf.name_scope('iconv1'):
    W_iconv1 = weight_variable([3,3, 97,32]) 
    b_iconv1 = bias_variable([32])
    h_iconv1 = tf.nn.relu(batch_norm(conv2d(tf.concat([h_upconv1, h_conv1, upsample(pr2)], 3), W_iconv1, [ 1, 1, 1, 1]) + b_iconv1)) 
    # output dims: 192*384*32

  # pr1 + loss1
  with tf.name_scope('pr1_loss1'):
    W_pr1 = weight_variable([3,3, 32,1]) 
    b_pr1 = bias_variable([1])
    pr1 = tf.nn.relu(batch_norm(conv2d(h_iconv1, W_pr1, [1, 1, 1, 1]) + b_pr1), name='final_result') 
    gt1 = tf.nn.avg_pool(ground_truth, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='gt')
    loss1 = loss(pr1, gt1)
    # pr1 dims: 192*384*1

  final_output = pr1
  # overall loss
  with tf.name_scope('loss'):
    total_loss = ( 1/2 * loss1 + 1/4 * loss2 + 1/8 * loss3 + 1/16 * loss4 + 1/32 * loss5 + 1/32 * loss6)
  return final_output, total_loss

def main():
 image_left = tf.placeholder(tf.float32, [None, IMAGE_SIZE_Y, IMAGE_SIZE_X, 3], name='image_left')
 image_right = tf.placeholder(tf.float32, [None, IMAGE_SIZE_Y, IMAGE_SIZE_X, 3], name='image_right')
 ground_truth = tf.placeholder(tf.float32, [None, IMAGE_SIZE_Y, IMAGE_SIZE_X, 1], name='ground_truth')
 combine_image = tf.concat([image_left, image_right], 3)
 final_output, total_loss= model(combine_image=combine_image, ground_truth=ground_truth)
    
 tf.summary.scalar('loss', total_loss)
    
 with tf.name_scope('train'):
   global_step = tf.Variable(0, trainable=False, name="global_step") 
   learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, decay_steps, decay_rate)
   optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
   training_op = optimizer.minimize(total_loss, global_step = global_step)
      
 init = tf.global_variables_initializer()
 saver = tf.train.Saver()
 
 with tf.Session() as sess:
  
  saver.restore(sess, "./dispnet_on_driving_model.ckpt")
  #init.run()
     
  left_paths = get_path.get_filelist_left_image(IMAGE_DIR, [])
  right_paths = get_path.get_filelist_right_image(IMAGE_DIR, [])
  disp_paths = get_path.get_filelist_disp(GT_DIR, [])
  
  train_rmse = 0
  
  test_rmse = 0
  
  # test set
  for k in range(TRAIN_NUM, TRAIN_NUM + TEST_NUM):
      print(k)
      left_path = left_paths[k]
      left = Image.open(left_path)
      left = _norm(np.reshape(np.array(left)[:IMAGE_SIZE_Y, :IMAGE_SIZE_X, :], (1, IMAGE_SIZE_Y, IMAGE_SIZE_X, 3)))
      if(k == TRAIN_NUM):
          left_images_test = left
      else:
          left_images_test = np.concatenate((left_images_test, left), axis=0)
                     
      right_path = right_paths[k] 
      right = Image.open(right_path)
      right = _norm(np.reshape(np.array(right)[:IMAGE_SIZE_Y, :IMAGE_SIZE_X, :], (1, IMAGE_SIZE_Y, IMAGE_SIZE_X, 3)))
      if(k == TRAIN_NUM):
          right_images_test = right
      else:
          right_images_test = np.concatenate((right_images_test, right), axis=0)
                    
      disp_path = disp_paths[k]
      disp = IO.read(disp_path)[:IMAGE_SIZE_Y, :IMAGE_SIZE_X]   
      disp = np.reshape(disp, (IMAGE_SIZE_Y, IMAGE_SIZE_X, 1))
      disp = np.reshape(disp, (1, IMAGE_SIZE_Y, IMAGE_SIZE_X, 1))
      if (k == TRAIN_NUM):
          disp_images_test = disp
      else:
          disp_images_test = np.concatenate((disp_images_test, disp), axis=0)
  for k in range(int(TEST_NUM/BATCH_SIZE)):
   test_rmse += total_loss.eval(feed_dict = {image_left:left_images_test[k*BATCH_SIZE:(k+1)*BATCH_SIZE], image_right:right_images_test[k*BATCH_SIZE:(k+1)*BATCH_SIZE], ground_truth:disp_images_test[k*BATCH_SIZE:(k+1)*BATCH_SIZE]})
  test_rmse = test_rmse/(TEST_NUM/BATCH_SIZE)
  print('whole test rmse: {}'.format(test_rmse))
  
  for epoch in range(TRAINING_ROUNDS):
   for i in range(0, TRAIN_NUM, BATCH_SIZE):
    print(i)
    for j in range(BATCH_SIZE):
       
     left_path = left_paths[i+j]
     left = Image.open(left_path)
     left = _norm(np.reshape(np.array(left)[:IMAGE_SIZE_Y, :IMAGE_SIZE_X, :], (1, IMAGE_SIZE_Y, IMAGE_SIZE_X, 3)))
     if(j == 0):
         left_images = left
     else:
         left_images = np.concatenate((left_images, left), axis=0)
                     
     right_path = right_paths[i+j] 
     right = Image.open(right_path)
     right = _norm(np.reshape(np.array(right)[:IMAGE_SIZE_Y, :IMAGE_SIZE_X, :], (1, IMAGE_SIZE_Y, IMAGE_SIZE_X, 3)))
     if(j == 0):
         right_images = right
     else:
         right_images = np.concatenate((right_images, right), axis=0)
                    
     disp_path = disp_paths[i+j]
     disp = IO.read(disp_path)[:IMAGE_SIZE_Y, :IMAGE_SIZE_X]   
     disp = np.reshape(disp, (IMAGE_SIZE_Y, IMAGE_SIZE_X, 1))
     disp = np.reshape(disp, (1, IMAGE_SIZE_Y, IMAGE_SIZE_X, 1))
     if(j == 0):
         disp_images = disp
     else:
         disp_images = np.concatenate((disp_images, disp), axis=0)
    
    train_rmse += total_loss.eval(feed_dict = {image_left:left_images, image_right:right_images, ground_truth:disp_images})
  train_rmse = train_rmse/(TRAIN_NUM/BATCH_SIZE)
  
  print('whole train rmse: {} whole test rmse: {}'.format(train_rmse, test_rmse))
     
  # output disparity image of test set 
  print('Start Outputing left, right images and corresponding disparity images of test set.')
  for k in list(range(32))+list(range(TRAIN_NUM, TRAIN_NUM+32)):
      
      left_path = left_paths[k]
      left = Image.open(left_path)
      left = np.array(left)[:IMAGE_SIZE_Y, :IMAGE_SIZE_X, :]
      if (k<32):
          plt.imsave(OUTPUT_DIR + '/' + 'train_left' +str(k) + '.png', left, format='png')
          print('left image {} is already saved.'.format(k))
      else:
          plt.imsave(OUTPUT_DIR + '/' + 'test_left' +str(k) + '.png', left, format='png')
          print('left image {} is already saved.'.format(k))
      left_images = _norm(np.reshape(left, (1, IMAGE_SIZE_Y, IMAGE_SIZE_X, 3)))
      
      right_path = right_paths[k]
      right = Image.open(right_path)
      right = np.array(right)[:IMAGE_SIZE_Y, :IMAGE_SIZE_X, :]
      if (k<32):
          plt.imsave(OUTPUT_DIR + '/' + 'train_right' +str(k) + '.png', right, format='png')
          print('right image {} is already saved.'.format(k))
      else:
          plt.imsave(OUTPUT_DIR + '/' + 'test_right' +str(k) + '.png', right, format='png')
          print('right image {} is already saved.'.format(k))
      right_images = _norm(np.reshape(right, (1, IMAGE_SIZE_Y, IMAGE_SIZE_X, 3)))
      
      disp_path = disp_paths[k]
      disp = IO.read(disp_path)[:IMAGE_SIZE_Y, :IMAGE_SIZE_X]
      disp_int = disp.astype(np.int8)
      plt.imsave(OUTPUT_DIR + '/' + 'groundtruth' +str(k) + '.png', disp_int, cmap='gray', format='png')
      print('ground truth {} is already saved.'.format(k))
          
      disp = np.reshape(disp, (IMAGE_SIZE_Y, IMAGE_SIZE_X, 1))    
      disp = np.reshape(disp, (1, IMAGE_SIZE_Y, IMAGE_SIZE_X, 1))
      
      disp_pre = final_output.eval(feed_dict = {image_left:left_images, image_right:right_images, ground_truth:disp})
      disp_pre = get_path.bilinear(disp_pre[0], disp_pre[0,:,:,0].shape, [IMAGE_SIZE_Y, IMAGE_SIZE_X])
      disp_pre = disp_pre[:,:,0].astype(np.int8)
      plt.imsave(OUTPUT_DIR + '/' + 'disp_pre' +str(k) + '.png', disp_pre, cmap='gray', format='png')
      print('disparity prediction {} is already saved.'.format(k))


if __name__ == '__main__': 
    main()







