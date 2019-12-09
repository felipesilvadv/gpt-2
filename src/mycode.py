import json
import model, encoder, sample
import tensorflow as tf
import os
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt


model_name = "124M"
enc = encoder.get_encoder(model_name, "models")
hparams = model.default_hparams()
with open(f"models/{model_name}/hparams.json") as file:
    hparams.override_from_dict(json.load(file))
seed = 131193
texto = "I love pizza"
texto1 = "I hate pizza"
texto2 = "I like pizza"
texto3 = "I am stupid"

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
    ckpt = tf.train.latest_checkpoint(os.path.join("models", model_name))
    saver.restore(sess, ckpt)
    #var = sess.run(variable)
    valor = sess.run(modelo, feed_dict={context: [enc.encode(texto)]})
    valor1 = sess.run(modelo, feed_dict={context: [enc.encode(texto1)]})
    valor2 = sess.run(modelo, feed_dict={context: [enc.encode(texto2)]})
    valor3 = sess.run(modelo, feed_dict={context: [enc.encode(texto3)]})
    dim = valor["present"].shape
    total = 1
    for elem in dim:
        total *= elem
    pca = PCA(n_components=2)
    sequence = [valor["present"].reshape(total),
                valor1["present"].reshape(total),
                valor2["present"].reshape(total),
                valor3["present"].reshape(total)]
    pca.fit(sequence)
    puntos = pca.transform(sequence)
    labels = [texto, texto1, texto2, texto3]
    for i in range(len(sequence)):
        plt.scatter(puntos[i][0], puntos[i][1], label=labels[i])
    plt.title("PCA for encoder example")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.savefig("ejemplo_encoder3.png")
    #print(enc.decode(var[0]))
