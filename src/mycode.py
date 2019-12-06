import json
import model, encoder, sample
import tensorflow as tf
import os
import numpy as np

enc = encoder.get_encoder("124M", "models")
hparams = model.default_hparams()
with open("models/124M/hparams.json") as file:
    hparams.override_from_dict(json.load(file))
seed = 131193
texto = "I love pizza"
texto1 = "I don't love pizza"

with tf.Session(graph=tf.Graph()) as sess:
    np.random.seed(seed)
    tf.set_random_seed(seed)
    #context = tf.fill([1, 1], enc.encoder["<|endoftext|>"])
    context = tf.placeholder(tf.int32, [1, None])
    modelo = model.model(hparams=hparams, X=context)
    #variable = sample.sample_sequence(hparams=hparams,
    #                                  start_token=enc.encoder["<|endoftext|>"],
    #                                  length=10,
    #                                  batch_size=1,
    #                                  temperature=1,
    #                                  top_k=0,
    #                                  top_p=1)[:,1:]
    #sess.run(init)
    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(os.path.join("models", "124M"))
    saver.restore(sess, ckpt)
    #var = sess.run(variable)
    valor = sess.run(modelo, feed_dict={context: [enc.encode(texto)]})
    print(valor["wte"][enc.encode(texto)])
    #valor1 = sess.run(modelo, feed_dict={context: [enc.encode(texto)]})
    #valor2 = sess.run(modelo, feed_dict={context: [enc.encode(texto1)]})
    #print(enc.decode(var[0]))
