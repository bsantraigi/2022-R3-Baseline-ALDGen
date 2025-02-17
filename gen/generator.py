from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import pickle
import heapq
import tensorflow.python.platform

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import utils.data_utils as data_utils
import utils.conf as conf
import gen.gen_model as seq2seq_model
from tensorflow.python.platform import gfile
sys.path.append('../utils')

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.


def read_data(config, source_path, target_path, max_size=None):
    data_set = [[] for _ in config.buckets]
    with gfile.GFile(source_path, mode="r") as source_file:
        with gfile.GFile(target_path, mode="r") as target_file:
            source, target = source_file.readline(), target_file.readline()
            counter = 0
            while source and target and (not max_size or counter < max_size):
                counter += 1
                if counter % 100000 == 0:
                    print("  reading disc_data line %d" % counter)
                    sys.stdout.flush()
                source_ids = [int(x) for x in source.split()]
                target_ids = [int(x) for x in target.split()]
                target_ids.append(data_utils.EOS_ID)
                for bucket_id, (source_size, target_size) in enumerate(config.buckets): #[bucket_id, (source_size, target_size)]
                    if len(source_ids) < source_size and len(target_ids) < target_size:
                        data_set[bucket_id].append([source_ids, target_ids])
                        break
                source, target = source_file.readline(), target_file.readline()
    return data_set


def create_model(session, gen_config, forward_only, name_scope, initializer=None):
    """Create translation model and initialize or load parameters in session."""
    with tf.variable_scope(name_or_scope=name_scope, initializer=initializer):
        model = seq2seq_model.Seq2SeqModel(gen_config,  name_scope=name_scope, forward_only=forward_only)
        # Try adv-learned gen model 
        gen_ckpt_dir = os.path.abspath(os.path.join(gen_config.train_dir, "al_checkpoints"))
        ckpt = tf.train.get_checkpoint_state(gen_ckpt_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading AL-Gen model parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(session, ckpt.model_checkpoint_path)
        else:
            # Try the pretrained gen model (i.e. without adv-training)
            gen_ckpt_dir = os.path.abspath(os.path.join(gen_config.train_dir, "checkpoints"))
            ckpt = tf.train.get_checkpoint_state(gen_ckpt_dir)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                print("Reading Gen model parameters from %s" % ckpt.model_checkpoint_path)
                model.saver.restore(session, ckpt.model_checkpoint_path)
            else:
                print("Created Gen model with fresh parameters.")
                gen_global_variables = [gv for gv in tf.global_variables() if name_scope in gv.name]
                session.run(tf.variables_initializer(gen_global_variables))
        return model


def prepare_data(gen_config):
    train_path = os.path.join(gen_config.train_dir, "chitchat.train")
    voc_file_path = [train_path+".answer", train_path+".query"]
    vocab_path = os.path.join(gen_config.train_dir, "vocab%d.all" % gen_config.vocab_size)
    data_utils.create_vocabulary(vocab_path, voc_file_path, gen_config.vocab_size)
    vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)

    print("Preparing Chitchat gen_data in %s" % gen_config.train_dir)
    train_query, train_answer, dev_query, dev_answer = data_utils.prepare_chitchat_data(
        gen_config.train_dir, vocab, gen_config.vocab_size)

    # Read disc_data into buckets and compute their sizes.
    print ("Reading development and training gen_data (limit: %d)."
               % gen_config.max_train_data_size)
    dev_set = read_data(gen_config, dev_query, dev_answer)
    train_set = read_data(gen_config, train_query, train_answer, gen_config.max_train_data_size)

    return vocab, rev_vocab, dev_set, train_set


def softmax(x):
    prob = np.exp(x) / np.sum(np.exp(x), axis=0)
    return prob


def train(gen_config):
    vocab, rev_vocab, dev_set, train_set = prepare_data(gen_config)
    for b_set in train_set:
        print("b_set: ", len(b_set))

    with tf.Session() as sess:
    #with tf.device("/gpu:1"):
        # Create model.
        print("Creating %d layers of %d units." % (gen_config.num_layers, gen_config.emb_dim))
        model = create_model(sess, gen_config, forward_only=False, name_scope=gen_config.name_model)

        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(gen_config.buckets))]
        train_total_size = float(sum(train_bucket_sizes))
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]
        print("Train bucket sizes: ", train_bucket_sizes)
        print("Train total size: ", train_total_size)

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []

        gen_loss_summary = tf.Summary()
        gen_writer = tf.summary.FileWriter(gen_config.tensorboard_dir, sess.graph)
        
        epoch_progress_equivalent = 0.0
        
        while True:
            # Choose a bucket according to disc_data distribution. We pick a random number
            # in [0, 1] and use the corresponding interval in train_buckets_scale.
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale)) if train_buckets_scale[i] > random_number_01])

            # Get a batch and make a step.
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights, batch_source_encoder, batch_source_decoder = model.get_batch(
                train_set, bucket_id, gen_config.batch_size)

            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, forward_only=False)
            
            # Count progress in terms of epoch
            epoch_progress_equivalent += gen_config.batch_size/train_total_size

            step_time += (time.time() - start_time) / gen_config.steps_per_checkpoint
            loss += step_loss / gen_config.steps_per_checkpoint
            current_step += 1

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % gen_config.steps_per_checkpoint == 0:

                bucket_value = gen_loss_summary.value.add()
                bucket_value.tag = gen_config.name_loss
                bucket_value.simple_value = float(loss)
                gen_writer.add_summary(gen_loss_summary, int(model.global_step.eval()))

                # Print statistics for the previous epoch.
                perplexity = math.exp(loss) if loss < 300 else float('inf')
                print ("[E: %.2f] global step %d learning rate %.4f step-time %.2f perplexity "
                       "%.2f" % (epoch_progress_equivalent, model.global_step.eval(), model.learning_rate.eval(),
                                 step_time, perplexity))
                # Decrease learning rate if no improvement was seen over last 3 times.
                # if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                #     sess.run(model.learning_rate_decay_op)
                
                # Save checkpoint and zero timer and loss.

                if current_step % (gen_config.steps_per_checkpoint * 2) == 0:
                    if len(previous_losses) > 2:
                        # update best model
                        if loss < min(previous_losses):
                            print("current_step: %d, save model" %(current_step))
                            gen_ckpt_dir = os.path.abspath(os.path.join(gen_config.train_dir, "checkpoints"))
                            if not os.path.exists(gen_ckpt_dir):
                                os.makedirs(gen_ckpt_dir)
                            checkpoint_path = os.path.join(gen_ckpt_dir, "chitchat.model")
                            model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                        else:
                            print("Not saving model. Loss hasn't improved.")
                    
                    PATIENCE=15
                    if len(previous_losses) > PATIENCE:
                        # Early stop - if min loss has not changed in last 15 checks                        
                        if min(previous_losses[-PATIENCE:]) > min(previous_losses):
                            break
                
                    if epoch_progress_equivalent > gen_config.max_epochs:
                        print("Stopping gen training due to max_epochs setting. E=%d" % epoch_progress_equivalent)
                        break 
                    previous_losses.append(loss)


                step_time, loss = 0.0, 0.0
                # Run evals on development set and print their perplexity.
                # for bucket_id in xrange(len(gen_config.buckets)):
                #   encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                #       dev_set, bucket_id)
                #   _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                #                                target_weights, bucket_id, True)
                #   eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
                #   print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
                sys.stdout.flush()


def test_decoder(gen_config, disc_config, output_file_extension):
    with tf.Session() as sess:
        model = create_model(sess, gen_config, forward_only=True, name_scope=gen_config.name_model)
        model.batch_size = 1
	
        # for split in ['dev', 'test', 'train']:
        for split in ['test', 'dev']:
            train_path = os.path.join(gen_config.train_dir, "chitchat.%s" % split)
            voc_file_path = [train_path + ".answer", train_path + ".query"]
            vocab_path = os.path.join(gen_config.train_dir, "vocab%d.all" % gen_config.vocab_size)
            data_utils.create_vocabulary(vocab_path, voc_file_path, gen_config.vocab_size)
            vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)
            
            # Initialize the array to store the generated outputs
            outputs_list = []
            
            # Iterate through the sentences in the train_path file
            with open(train_path + ".query", 'r') as f:
                for xi, sentence in enumerate(f):
                    if xi % 100 == 0:
                        print("Progress %d"% xi)
                    token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), vocab)
                    # print("token_id: ", token_ids)
                    bucket_id = len(gen_config.buckets) - 1
                    for i, bucket in enumerate(gen_config.buckets):
                        if bucket[0] >= len(token_ids):
                            bucket_id = i
                            break
                    # else:
                        # print("Sentence truncated: %s"% sentence)

                    encoder_inputs, decoder_inputs, target_weights, _, _ = model.get_batch({bucket_id: [(token_ids, [1])]},
                                                                 bucket_id, model.batch_size, type=0)

                    # print("bucket_id: ", bucket_id)
                    # print("encoder_inputs:", encoder_inputs)
                    # print("decoder_inputs:", decoder_inputs)
                    # print("target_weights:", target_weights)

                    _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)

                    # print("output_logits", np.shape(output_logits))

                    outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]

                    if data_utils.EOS_ID in outputs:
                        outputs = outputs[:outputs.index(data_utils.EOS_ID)]
                    # Convert the outputs to strings and store them in the outputs list
                    outputs_list.append(" ".join([tf.compat.as_str(rev_vocab[output]) for output in outputs]))

                    # print(" ".join([tf.compat.as_str(rev_vocab[output]) for output in outputs]))
                    # print("> ", end="")
                    # sys.stdout.flush()
                    # sentence = sys.stdin.readline()

                # Dump the generated outputs to the file ({split}.gen)
                out_path = os.path.join(disc_config.train_dir, split)
                with open(out_path + output_file_extension, 'w') as f:
                    for output in outputs_list:
                        f.write(output + '\n')

def decoder(gen_config):
    vocab, rev_vocab, dev_set, train_set = prepare_data(gen_config)

    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(gen_config.buckets))]
    train_total_size = float(sum(train_bucket_sizes))
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]

    with tf.Session() as sess:
        model = create_model(sess, gen_config, forward_only=True, name_scope=gen_config.name_model)
        
        disc_data_path = gen_config.train_dir.replace("gen_data", "disc_data")
        print("Writing disc training files to %s" % (disc_data_path,))
        disc_train_query = open(os.path.join(disc_data_path, "train.query"), "w")
        disc_train_answer = open(os.path.join(disc_data_path, "train.answer"), "w")
        disc_train_gen = open(os.path.join(disc_data_path, "train.gen"), "w")

        num_step = 0
        while num_step < 10000:
            print("generating num_step: ", num_step)
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])

            encoder_inputs, decoder_inputs, target_weights, batch_source_encoder, batch_source_decoder = \
                model.get_batch(train_set, bucket_id, gen_config.batch_size)

            _, _, out_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id,
                                          forward_only=True)

            tokens = []
            resps = []
            for seq in out_logits:
                token = []
                for t in seq:
                    token.append(int(np.argmax(t, axis=0)))
                tokens.append(token)
            tokens_t = []
            for col in range(len(tokens[0])):
                tokens_t.append([tokens[row][col] for row in range(len(tokens))])

            for seq in tokens_t:
                if data_utils.EOS_ID in seq:
                    resps.append(seq[:seq.index(data_utils.EOS_ID)][:gen_config.buckets[bucket_id][1]])
                else:
                    resps.append(seq[:gen_config.buckets[bucket_id][1]])

            for query, answer, resp in zip(batch_source_encoder, batch_source_decoder, resps):

                answer_str = " ".join([str(rev_vocab[an]) for an in answer][:-1])
                disc_train_answer.write(answer_str)
                disc_train_answer.write("\n")

                query_str = " ".join([str(rev_vocab[qu]) for qu in query])
                disc_train_query.write(query_str)
                disc_train_query.write("\n")

                resp_str = " ".join([tf.compat.as_str(rev_vocab[output]) for output in resp])

                disc_train_gen.write(resp_str)
                disc_train_gen.write("\n")
            num_step += 1

        disc_train_gen.close()
        disc_train_query.close()
        disc_train_answer.close()
    pass




def decoder_bk(gen_config):
    vocab, rev_vocab, dev_set, train_set = prepare_data(gen_config)

    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(gen_config.buckets))]
    train_total_size = float(sum(train_bucket_sizes))
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]

    with tf.Session() as sess:
        model = create_model(sess, gen_config, forward_only=True, name_scope=gen_config.name_model)

        disc_train_query = open("train.query", "w")
        disc_train_answer = open("train.answer", "w")
        disc_train_gen = open("train.gen", "w")

        num_step = 30000
        while num_step > 0:
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])

            encoder_inputs, decoder_inputs, target_weights, batch_source_encoder, batch_source_decoder = model.get_batch(
                train_set, bucket_id, gen_config.batch_size)

            _, _, out_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, forward_only=True)

            tokens = []
            resps = []
            for seq in out_logits:
                # print("seq: %s" %seq)
                token = []
                for t in seq:
                    token.append(int(np.argmax(t, axis=0)))
                tokens.append(token)
            tokens_t = []
            for col in range(len(tokens[0])):
                tokens_t.append([tokens[row][col] for row in range(len(tokens))])

            for seq in tokens_t:
                if data_utils.EOS_ID in seq:
                    resps.append(seq[:seq.index(data_utils.EOS_ID)][:gen_config.buckets[bucket_id][1]])
                else:
                    resps.append(seq[:gen_config.buckets[bucket_id][1]])

            for query, answer, resp in zip(batch_source_encoder, batch_source_decoder, resps):
                disc_train_answer.write(answer)
                disc_train_answer.write("\n")

                disc_train_query.write(query)
                disc_train_query.write("\n")

                disc_train_gen.write(resp)
                disc_train_gen.write("\n")
            num_step -= 1

        disc_train_gen.close()
        disc_train_query.close()
        disc_train_answer.close()
    pass


def get_predicted_sentence(sess, input_token_ids, vocab, model,
                           beam_size, buckets, mc_search=True,debug=False):
    def model_step(enc_inp, dec_inp, dptr, target_weights, bucket_id):
        #model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
        _, _, logits = model.step(sess, enc_inp, dec_inp, target_weights, bucket_id, True)
        prob = softmax(logits[dptr][0])
        # print("model_step @ %s" % (datetime.now()))
        return prob

    def greedy_dec(output_logits):
        selected_token_ids = [int(np.argmax(logit, axis=1)) for logit in output_logits]
        return selected_token_ids

    #input_token_ids = data_utils.sentence_to_token_ids(input_sentence, vocab)
    # Which bucket does it belong to?
    bucket_id = min([b for b in range(len(buckets)) if buckets[b][0] > len(input_token_ids)])
    outputs = []
    feed_data = {bucket_id: [(input_token_ids, outputs)]}

    # Get a 1-element batch to feed the sentence to the model.   None,bucket_id, True
    encoder_inputs, decoder_inputs, target_weights, _, _ = model.get_batch(feed_data, bucket_id, 1)
    if debug: print("\n[get_batch]\n", encoder_inputs, decoder_inputs, target_weights)

    ### Original greedy decoding
    if beam_size == 1 or (not mc_search):
        _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
        return [{"dec_inp": greedy_dec(output_logits), 'prob': 1}]

    # Get output logits for the sentence. # initialize beams as (log_prob, empty_string, eos)
    beams, new_beams, results = [(1, {'eos': 0, 'dec_inp': decoder_inputs, 'prob': 1, 'prob_ts': 1, 'prob_t': 1})], [], []

    for dptr in range(len(decoder_inputs)-1):
      if dptr > 0:
        target_weights[dptr] = [1.]
        beams, new_beams = new_beams[:beam_size], []
      if debug: print("=====[beams]=====", beams)
      heapq.heapify(beams)  # since we will srot and remove something to keep N elements
      for prob, cand in beams:
        if cand['eos']:
          results += [(prob, cand)]
          continue

        all_prob_ts = model_step(encoder_inputs, cand['dec_inp'], dptr, target_weights, bucket_id)
        all_prob_t  = [0]*len(all_prob_ts)
        all_prob    = all_prob_ts

        # suppress copy-cat (respond the same as input)
        if dptr < len(input_token_ids):
          all_prob[input_token_ids[dptr]] = all_prob[input_token_ids[dptr]] * 0.01

        # beam search
        for c in np.argsort(all_prob)[::-1][:beam_size]:
          new_cand = {
            'eos'     : (c == data_utils.EOS_ID),
            'dec_inp' : [(np.array([c]) if i == (dptr+1) else k) for i, k in enumerate(cand['dec_inp'])],
            'prob_ts' : cand['prob_ts'] * all_prob_ts[c],
            'prob_t'  : cand['prob_t'] * all_prob_t[c],
            'prob'    : cand['prob'] * all_prob[c],
          }
          new_cand = (new_cand['prob'], new_cand) # for heapq can only sort according to list[0]

          if (len(new_beams) < beam_size):
            heapq.heappush(new_beams, new_cand)
          elif (new_cand[0] > new_beams[0][0]):
            heapq.heapreplace(new_beams, new_cand)

    results += new_beams  # flush last cands

    # post-process results
    res_cands = []
    for prob, cand in sorted(results, reverse=True):
      res_cands.append(cand)
    return res_cands


def gen_sample(sess ,gen_config, model, vocab, source_inputs, source_outputs, mc_search=True):
    sample_inputs = []
    sample_labels =[]
    rep = []

    for source_query, source_answer in zip(source_inputs, source_outputs):
        sample_inputs.append(source_query+source_answer)
        sample_labels.append(1)
        responses = get_predicted_sentence(sess, source_query, vocab,
                                           model, gen_config.beam_size, gen_config.buckets, mc_search)

        for resp in responses:
            if gen_config.beam_size == 1 or (not mc_search):
                dec_inp = [dec for dec in resp['dec_inp']]
                rep.append(dec_inp)
                dec_inp = dec_inp[:]
            else:
                dec_inp = [dec.tolist()[0] for dec in resp['dec_inp'][:]]
                rep.append(dec_inp)
                dec_inp = dec_inp[1:]
            print("  (%s) -> %s" % (resp['prob'], dec_inp))
            sample_neg = source_query + dec_inp
            sample_inputs.append(sample_neg)
            sample_labels.append(0)

    return sample_inputs, sample_labels, rep
    pass
