from __future__ import division

# Required to run the script on Slurm
import sys
import os
os.chdir('/ubc/cs/research/shield/projects/aditya10/RefSeg/')
sys.path.append('/ubc/cs/research/shield/projects/aditya10/python-packages/')
sys.path.append('/ubc/cs/research/shield/projects/aditya10/RefSeg/external/tensorflow-deeplab-resnet/')

import argparse
import tensorflow as tf
import skimage
from skimage import io as sio
import time
import matplotlib.pyplot as plt
from get_model import get_segmentation_model
from pydensecrf import densecrf
import h5py

from util import data_reader
from util.processing_tools import *
from util import im_processing, eval_tools, MovingAverage

def train(max_iter, snapshot, dataset, setname, mu, lr, bs, tfmodel_folder,
          conv5, model_name, stop_iter, pre_emb=False, visualize=False):


    iters_per_log = 100
    vcount_thresh = 5000
    vcount = 0
    data_folder = '/ubc/cs/research/shield/datasets/refer/data/' + dataset + '/'
    data_prefix = dataset + '_' + setname
    #model_folder = '/ubc/cs/research/shield/projects/aditya10/RefSeg/results'
    
    snapshot_file = os.path.join(tfmodel_folder, dataset + '_iter_%d.tfmodel')
    if not os.path.isdir(tfmodel_folder):
        os.makedirs(tfmodel_folder)

    

    cls_loss_avg = 0
    avg_accuracy_all, avg_accuracy_pos, avg_accuracy_neg = 0, 0, 0
    decay = 0.99
    vocab_size = 8803 if dataset == 'referit' else 12112
    emb_name = 'referit' if dataset == 'referit' else 'Gref'

    if pre_emb:
        print("Use pretrained Embeddings.")
        model = get_segmentation_model(model_name, mode='train',
                                       vocab_size=vocab_size, start_lr=lr,
                                       batch_size=bs, conv5=conv5, emb_name=emb_name)
    else:
        model = get_segmentation_model(model_name, mode='train',
                                       vocab_size=vocab_size, start_lr=lr,
                                       batch_size=bs, conv5=conv5)

    weights = './data/weights/deeplab_resnet_init.ckpt'
    print("Loading pretrained weights from {}".format(weights))
    load_var = {var.op.name: var for var in tf.global_variables()
                if var.name.startswith('res') or var.name.startswith('bn') or var.name.startswith('conv1')}

    snapshot_loader = tf.train.Saver(load_var)
    snapshot_saver = tf.train.Saver(max_to_keep=4)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    snapshot_loader.restore(sess, weights)

    im_h, im_w, num_steps = model.H, model.W, model.num_steps
    text_batch = np.zeros((bs, num_steps), dtype=np.float32)
    image_batch = np.zeros((bs, im_h, im_w, 3), dtype=np.float32)
    mask_batch = np.zeros((bs, im_h, im_w, 1), dtype=np.float32)
    valid_idx_batch = np.zeros((bs, 1), dtype=np.int32)

    reader = data_reader.DataReader(data_folder, data_prefix)

    # for time calculate
    last_time = time.time()
    time_avg = MovingAverage()
    for n_iter in range(max_iter):

        for n_batch in range(bs):
            batch = reader.read_batch(is_log=(n_batch == 0 and n_iter % iters_per_log == 0))
            text = batch['text_batch']
            im = batch['im_batch'].astype(np.float32)
            mask = np.expand_dims(batch['mask_batch'].astype(np.float32), axis=2)

            im = im[:, :, ::-1]
            im -= mu

            text_batch[n_batch, ...] = text
            image_batch[n_batch, ...] = im
            mask_batch[n_batch, ...] = mask

            for idx in range(text.shape[0]):
                if text[idx] != 0:
                    valid_idx_batch[n_batch, :] = idx
                    break

        _, cls_loss_val, lr_val, scores_val, label_val, labelfine_val, up_val = sess.run([model.train_step,
                                                                   model.cls_loss,
                                                                   model.learning_rate,
                                                                   model.pred_graph,
                                                                   model.target,
                                                                   model.target_fine,
                                                                   model.up],
                                                                  feed_dict={
                                                                      model.words: text_batch,
                                                                      # np.expand_dims(text, axis=0),
                                                                      model.im: image_batch,
                                                                      # np.expand_dims(im, axis=0),
                                                                      model.target_fine: mask_batch,
                                                                      # np.expand_dims(mask, axis=0)
                                                                      model.valid_idx: valid_idx_batch
                                                                  })

        print(np.shape(label_val))
        cls_loss_avg = decay * cls_loss_avg + (1 - decay) * cls_loss_val

        # Accuracy
        accuracy_all, accuracy_pos, accuracy_neg = compute_accuracy(scores_val, label_val)
        # accuracy_all, accuracy_pos, accuracy_neg = compute_accuracy(up_val, labelfine_val)
        avg_accuracy_all = decay * avg_accuracy_all + (1 - decay) * accuracy_all
        avg_accuracy_pos = decay * avg_accuracy_pos + (1 - decay) * accuracy_pos
        avg_accuracy_neg = decay * avg_accuracy_neg + (1 - decay) * accuracy_neg

        # timing
        cur_time = time.time()
        elapsed = cur_time - last_time
        last_time = cur_time

        # Print log
        if n_iter % iters_per_log == 0:
            print('iter = %d, loss (cur) = %f, loss (avg) = %f, lr = %f'
                  % (n_iter, cls_loss_val, cls_loss_avg, lr_val))
            print('iter = %d, accuracy (cur) = %f (all), %f (pos), %f (neg)'
                  % (n_iter, accuracy_all, accuracy_pos, accuracy_neg))
            print('iter = %d, accuracy (avg) = %f (all), %f (pos), %f (neg)'
                  % (n_iter, avg_accuracy_all, avg_accuracy_pos, avg_accuracy_neg))
            time_avg.add(elapsed)
            print('iter = %d, cur time = %.5f, avg time = %.5f, model_name: %s' % (n_iter, elapsed, time_avg.get_avg(), model_name))
            with open(tfmodel_folder+"/avgloss.txt", "a") as myfile:
                myfile.write("%f\n"%(cls_loss_avg))
            with open(tfmodel_folder+"/avgacc.txt", "a") as myfile:
                myfile.write("%f\n"%(avg_accuracy_all))

        # Save snapshot
        if (n_iter + 1) % snapshot == 0 or (n_iter + 1) >= max_iter:
            snapshot_saver.save(sess, snapshot_file % (n_iter + 1))
            print('snapshot saved to ' + snapshot_file % (n_iter + 1))
        if (n_iter + 1) >= stop_iter:
            print('stop training at iter ' + str(stop_iter))
            break

    print('Optimization done.')


def test(iter, dataset, visualize, setname, dcrf, mu, tfmodel_folder, model_name, pre_emb=False):
    
    data_folder = '/ubc/cs/research/shield/datasets/refer/data/' + dataset + '/'
    data_prefix = dataset + '_' + setname
    if visualize:
        save_dir = './visualize/' + dataset + "/test_final_"+ str(iter) + '/'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
    weights = os.path.join(tfmodel_folder, dataset + '_iter_' + str(iter) + '.tfmodel')
    print("Loading trained weights from {}".format(weights))

    score_thresh = 1e-9
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    cum_I, cum_U = 0, 0
    cum_I_base, cum_U_base = 0, 0
    cum_I_graph, cum_U_graph = 0, 0
    mean_IoU, mean_IoU_base, mean_IoU_graph, mean_dcrf_IoU = 0, 0, 0, 0
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    if dcrf:
        cum_I_dcrf, cum_U_dcrf = 0, 0
        seg_correct_dcrf = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0.
    H, W = 320, 320
    vocab_size = 8803 if dataset == 'referit' else 12112
    emb_name = 'referit' if dataset == 'referit' else 'Gref'

    IU_result = list()

    if pre_emb:
        # use pretrained embbeding
        print("Use pretrained Embeddings.")
        model = get_segmentation_model(model_name, H=H, W=W,
                                       mode='eval', vocab_size=vocab_size, emb_name=emb_name)
    else:
        model = get_segmentation_model(model_name, H=H, W=W,
                                       mode='eval', vocab_size=vocab_size)

    # Load pretrained model
    snapshot_restorer = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    snapshot_restorer.restore(sess, weights)
    reader = data_reader.DataReader(data_folder, data_prefix, shuffle=False)

    NN = reader.num_batch
    vcount = 0
    for n_iter in range(reader.num_batch):

        if n_iter % (NN // 50) == 0:
            if n_iter / (NN // 50) % 5 == 0:
                sys.stdout.write(str(n_iter / (NN // 50) // 5))
            else:
                sys.stdout.write('.')
            sys.stdout.flush()

        batch = reader.read_batch(is_log=False)
        text = batch['text_batch']
        im = batch['im_batch']
        mask = batch['mask_batch'].astype(np.float32)
        valid_idx = np.zeros([1], dtype=np.int32)
        for idx in range(text.shape[0]):
            if text[idx] != 0:
                valid_idx[0] = idx
                break

        proc_im = skimage.img_as_ubyte(im_processing.resize_and_pad(im, H, W))
        proc_im_ = proc_im.astype(np.float32)
        proc_im_ = proc_im_[:, :, ::-1]
        proc_im_ -= mu

        
        if visualize: 
            scores_val, up_val, sigm_val, pred_base, up_base, pred_graph, up_graph, adj_base, adj_pred, adj_refine = sess.run([model.pred, model.up, model.sigm, model.pred_base, model.up_base, model.pred_graph, model.up_graph, model.adj_base, model.adj_pred, model.adj_refine],
                                                feed_dict={
                                                    model.words: np.expand_dims(text, axis=0),
                                                    model.im: np.expand_dims(proc_im_, axis=0),
                                                    model.valid_idx: np.expand_dims(valid_idx, axis=0)
                                                })
            
            pred_base = pred_base[0].sum(axis=2)
            pred_graph = pred_graph[0].sum(axis=2)
            pred_final = scores_val[0].sum(axis=2)
            adj_base = adj_base[0]
            adj_pred = adj_pred[0]
            adj_refine = adj_refine[0]
            pred_diff = np.subtract(pred_graph, pred_base)
            vmin = min(np.min(pred_base), np.min(pred_graph), np.min(pred_final), np.min(pred_diff))
            vmax = max(np.max(pred_base), np.max(pred_graph), np.max(pred_final), np.max(pred_diff))
            
        else: 
            scores_val, up_val, sigm_val, up_base, up_graph = sess.run([model.pred, model.up, model.sigm, model.up_base, model.up_graph],
                                                    feed_dict={
                                                        model.words: np.expand_dims(text, axis=0),
                                                        model.im: np.expand_dims(proc_im_, axis=0),
                                                        model.valid_idx: np.expand_dims(valid_idx, axis=0)
                                                    })

        up_val_squeezed = np.squeeze(up_val)
        pred_raw = (up_val_squeezed >= score_thresh).astype(np.float32)
        predicts = im_processing.resize_and_crop(pred_raw, mask.shape[0], mask.shape[1])

        up_graph_squeezed = np.squeeze(up_graph)
        pred_graph_raw = (up_graph_squeezed >= score_thresh).astype(np.float32)
        predicts_graph = im_processing.resize_and_crop(pred_graph_raw, mask.shape[0], mask.shape[1])

        up_base_squeezed = np.squeeze(up_base)
        pred_base_raw = (up_base_squeezed >= score_thresh).astype(np.float32)
        predicts_base = im_processing.resize_and_crop(pred_base_raw, mask.shape[0], mask.shape[1])

        if dcrf:
            # Dense CRF post-processing
            sigm_val = np.squeeze(sigm_val)
            d = densecrf.DenseCRF2D(W, H, 2)
            U = np.expand_dims(-np.log(sigm_val), axis=0)
            U_ = np.expand_dims(-np.log(1 - sigm_val), axis=0)
            unary = np.concatenate((U_, U), axis=0)
            unary = unary.reshape((2, -1))
            d.setUnaryEnergy(unary)
            d.addPairwiseGaussian(sxy=3, compat=3)
            d.addPairwiseBilateral(sxy=20, srgb=3, rgbim=proc_im, compat=10)
            Q = d.inference(5)
            pred_raw_dcrf = np.argmax(Q, axis=0).reshape((H, W)).astype(np.float32)
            predicts_dcrf = im_processing.resize_and_crop(pred_raw_dcrf, mask.shape[0], mask.shape[1])


        I, U = eval_tools.compute_mask_IU(predicts, mask)
        I_graph, U_graph = eval_tools.compute_mask_IU(predicts_graph, mask)
        I_base, U_base = eval_tools.compute_mask_IU(predicts_base, mask)
        IoU = float(I)/U
        IoU_graph = float(I_graph)/U_graph
        IoU_base = float(I_base)/U_base

        #visualize if the IoU is smaller than this, i.e. the images where the segmentation results are worse
        IoU_VisThresh = 0.9
        
        if visualize & (IoU<=IoU_VisThresh):
            sent = batch['sent_batch'][()]
            hmaps = [(pred_base, "2-pred_base"), (pred_graph, "5-pred_graph"), (pred_final, "6-pred_final"), (pred_diff, "7-pred_diff")]
            adjmaps = [(adj_base, "1-adj_base"), (adj_pred, "3-adj_pred"), (adj_refine, "4-adj_refine")]
            if dcrf:
                visualize_seg(im, mask, predicts_dcrf, sent, IoU, IoU_base=IoU_base, hmaps=hmaps, adjmaps=adjmaps, vmin=vmin, vmax=vmax)
            else:
                visualize_seg(im, mask, predicts, sent, IoU, IoU_base=IoU_base, hmaps=hmaps, adjmaps=adjmaps, vmin=vmin, vmax=vmax)

        IU_result.append({'batch_no': n_iter, 'I': I, 'U': U})
        mean_IoU += float(I) / U
        cum_I += I
        cum_U += U

        mean_IoU_base += float(I_base) / U_base
        cum_I_base += I_base
        cum_U_base += U_base

        mean_IoU_graph += float(I_graph) / U_graph
        cum_I_graph += I_graph
        cum_U_graph += U_graph

        msg = 'cumulative IoU = %f' % (cum_I / cum_U)
        for n_eval_iou in range(len(eval_seg_iou_list)):
            eval_seg_iou = eval_seg_iou_list[n_eval_iou]
            seg_correct[n_eval_iou] += (I / U >= eval_seg_iou)
        if dcrf:
            I_dcrf, U_dcrf = eval_tools.compute_mask_IU(predicts_dcrf, mask)
            mean_dcrf_IoU += float(I_dcrf) / U_dcrf
            cum_I_dcrf += I_dcrf
            cum_U_dcrf += U_dcrf
            msg += '\tcumulative IoU (dcrf) = %f' % (cum_I_dcrf / cum_U_dcrf)
            for n_eval_iou in range(len(eval_seg_iou_list)):
                eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                seg_correct_dcrf[n_eval_iou] += (I_dcrf / U_dcrf >= eval_seg_iou)
        # print(msg)
        seg_total += 1

    # Print results
    print('Segmentation evaluation (without DenseCRF):')
    result_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        result_str += 'precision@%s = %f\n' % \
                      (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] / seg_total)
    result_str += 'overall IoU = %f; mean IoU = %f\n' % (cum_I / cum_U, mean_IoU / seg_total)
    print(result_str)
    if dcrf:
        print('Segmentation evaluation (with DenseCRF):')
        result_str = ''
        for n_eval_iou in range(len(eval_seg_iou_list)):
            result_str += 'precision@%s = %f\n' % \
                          (str(eval_seg_iou_list[n_eval_iou]), seg_correct_dcrf[n_eval_iou] / seg_total)
        result_str += 'overall IoU = %f; mean IoU = %f\n' % (cum_I_dcrf / cum_U_dcrf, mean_dcrf_IoU / seg_total)
        print(result_str)
    
    print('overall base IoU = %f; mean IoU = %f\n' % (cum_I_base / cum_U_base, mean_IoU_base / seg_total))
    print('overall graph IoU = %f; mean IoU = %f\n' % (cum_I_graph / cum_U_graph, mean_IoU_graph / seg_total))


def visualize_seg(im, mask, predicts, sent, IoU, interim=None, IoU_base=None, hmaps=None, adjmaps=None, vmin=-20, vmax=20):
    # Saves visualizations as image files
    vis_dir = "./visualize/unc/test_final_500"
    mask_alpha = 0.4
    IoU_str = str("%.3f" % round(IoU,3))[2:]
    IoU_sent = IoU_str[0]+"/"+IoU_str+ " - "+sent
    sent_dir = os.path.join(vis_dir, IoU_sent)
    if not os.path.exists(sent_dir):
        os.makedirs(sent_dir)

    # Ignore sio warnings of low-contrast image.
    import warnings
    warnings.filterwarnings('ignore')

    # write IoU score to the text file
    f = open(os.path.join(sent_dir, 'result.txt'), 'w')
    f.write(str(IoU)+" \n")
    if IoU_base is not None:
        f.write(str(IoU_base)+" \n")
        f.write("BETTER?   "+str(IoU_base<IoU)+" \n")
    f.close()

    # save original image
    sio.imsave(os.path.join(sent_dir, "im.png"), im)
    
    # save ground truth mask
    im_gt = np.zeros_like(im)
    im_gt[:, :, 2] = 170
    im_gt[:, :, 0] += mask.astype('uint8') * 170
    im_gt = im_gt.astype('int16')
    im_gt[:, :, 2] += mask.astype('int16') * (-170)
    im_gt = im_gt.astype('uint8')
    im_gt = np.ubyte(((1-mask_alpha)*im)+(mask_alpha*im_gt))
    sio.imsave(os.path.join(sent_dir, "gt.png"), im_gt)
    
    # save base mask
    if interim is not None:
        im_interim = np.zeros_like(im)
        im_interim[:, :, 2] = 170
        im_interim[:, :, 0] += interim.astype('uint8') * 170
        im_interim = im_interim.astype('int16')
        im_interim[:, :, 2] += interim.astype('int16') * (-170)
        im_interim = im_interim.astype('uint8')
        im_interim = np.ubyte(((1-mask_alpha)*im)+(mask_alpha*im_interim))
        sio.imsave(os.path.join(sent_dir, "base_mask.png"), im_interim)

    # save predicted mask
    im_pred = np.zeros_like(im)
    im_pred[:, :, 2] = 170
    im_pred[:, :, 0] += predicts.astype('uint8') * 170
    im_pred = im_pred.astype('int16')
    im_pred[:, :, 2] += predicts.astype('int16') * (-170)
    im_pred = im_pred.astype('uint8')
    im_pred = np.ubyte(((1-mask_alpha)*im)+(mask_alpha*im_pred))
    sio.imsave(os.path.join(sent_dir, "mask.png"), im_pred)

    # Save heatmap for scene graph
    if hmaps is not None:
        for hmap in hmaps:
            plt.imsave(os.path.join(sent_dir, str(hmap[1])+".png"), hmap[0], cmap='hot', vmin=vmin, vmax=vmax)
    if adjmaps is not None:
        for adjmap in adjmaps:
            plt.imsave(os.path.join(sent_dir, str(adjmap[1])+".png"), adjmap[0], cmap='hot')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', type=str, default='0')
    parser.add_argument('-i', type=int, default=800000)
    parser.add_argument('-s', type=int, default=100000)
    parser.add_argument('-st', type=int, default=800000)  # stop training when get st iters
    parser.add_argument('-m', type=str)  # 'train' 'test'
    parser.add_argument('-d', type=str, default='referit')  # 'Gref' 'unc' 'unc+' 'referit'
    parser.add_argument('-t', type=str)  # 'train' 'trainval' 'val' 'test' 'testA' 'testB'
    parser.add_argument('-f', type=str)  # directory to save models
    parser.add_argument('-lr', type=float, default=0.00025)  # start learning rate
    parser.add_argument('-bs', type=int, default=1)  # batch size
    parser.add_argument('-v', default=False, action='store_true')  # visualization
    parser.add_argument('-c', default=False, action='store_true')  # whether or not apply DenseCRF
    parser.add_argument('-emb', default=False, action='store_true')  # whether or not use Pretrained Embeddings
    parser.add_argument('-n', type=str, default='')  # select model
    parser.add_argument('-conv5', default=False, action='store_true')  # finetune conv layers

    args = parser.parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.g
    mu = np.array((104.00698793, 116.66876762, 122.67891434))

    if args.m == 'train':
        train(max_iter=args.i,
              snapshot=args.s,
              dataset=args.d,
              setname=args.t,
              mu=mu,
              lr=args.lr,
              bs=args.bs,
              tfmodel_folder=args.f,
              conv5=args.conv5,
              model_name=args.n,
              stop_iter=args.st,
              pre_emb=args.emb,
              visualize=args.v)
    elif args.m == 'test':
        test(iter=args.i,
             dataset=args.d,
             visualize=args.v,
             setname=args.t,
             dcrf=args.c,
             mu=mu,
             tfmodel_folder=args.f,
             model_name=args.n,
             pre_emb=args.emb)