# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import extraction as extract
import sys, getopt

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def init_bias(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#Opérations du réseau
def model(X, w_h, w_h2, w_o,b_h,b_h2,b_o, p_drop_input, p_drop_hidden): 
    X = tf.nn.dropout(X, p_drop_input)
    h = tf.nn.relu(tf.matmul(X, w_h) + b_h)
    

    h = tf.nn.dropout(h, p_drop_hidden)
    h2 = tf.nn.relu(tf.matmul(h, w_h2) + b_h2)

    h2 = tf.nn.dropout(h2, p_drop_hidden)

    return tf.nn.relu(tf.matmul(h2, w_o)+b_o)




#Apprentissage du réseau
def train(train_data,sess,saver,accuracy,X,Y,p_keep_input,p_keep_hidden,train_op,nbTours=5000, nbLocs=1000):

    print("Initialisation de l'apprentissage")

    # Restauration du réseau
    #saver.restore(sess,"sauvegarde_reseau.ckpt")
    #print("Modèle restauré")
    
    print("Initialisation terminée")
    print("Démarrage de l'apprentissage")

    for i in range(nbTours):
        
        batch_xs, batch_ys = train_data.next_batch(20000)
        
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                X: batch_xs, Y: batch_ys, p_keep_input: 1.0, p_keep_hidden: 1.0})
            
            print("Précision du réseau : %f" % train_accuracy)
            
        sess.run(train_op, feed_dict={X: batch_xs, Y: batch_ys,
                                    p_keep_input: 0.8, p_keep_hidden: 0.5})

        if i%1000 == 0 :
            #Sauvegarde du réseau
            save_path = saver.save(sess,"sauvegarde_reseau.ckpt")
            print("Modèle sauvegardé dans : %s" % save_path)

    #Sauvegarde du réseau
    save_path = saver.save(sess,"sauvegarde_reseau.ckpt")
    print("Modèle sauvegardé dans : %s" % save_path)




#Evaluation du réseau
def evaluate(data,sess,saver,accuracy,X,Y,p_keep_input,p_keep_hidden):
    
    #Restauration du modele
    saver.restore(sess,"sauvegarde_reseau.ckpt")
    print("Modèle restauré\n")
    
    print("Evaluating test data\n")
        
    e_accuracy = sess.run(accuracy, feed_dict = {X: data.features, Y: data.labels,
                                        p_keep_input: 1.0, p_keep_hidden: 1.0})
        
    print("Accuracy : %f" % e_accuracy)



def main(argv):
    
    try:
        opts, args = getopt.getopt(argv,"t:e:s")
    except:
        print("wrong arguments")
        sys.exit(2)

    data = extract.load_dataset('../Data/Caida/data_caida_original.arff')

    #Normalization
    data.normalize()

    
    
    X = tf.placeholder("float", [None, data.features.shape[1]])
    Y = tf.placeholder("float", [None, data.labels.shape[1]])

    #Architecture du réseau
    w_h = init_weights([data.features.shape[1],800])
    b_h = init_bias([800])
    w_h2 = init_weights([800, 800])
    b_h2 = init_bias([800])
    w_o = init_weights([800, data.labels.shape[1]])
    b_o = init_bias([data.labels.shape[1]])

    p_keep_input = tf.placeholder("float")
    p_keep_hidden = tf.placeholder("float")
    py_x = model(X, w_h, w_h2, w_o,b_h, b_h2, b_o, p_keep_input, p_keep_hidden)

    correct_prediction = tf.equal(tf.argmax(py_x,1),tf.argmax(Y,1))

    classes = tf.argmax(py_x,1)+1
    loc_max = tf.reduce_max(classes)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

    train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)

    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
    init = tf.initialize_all_variables()

    #Initialisation des opération de sauvegarde du modèle
    saver = tf.train.Saver()

    sess.run(init)

    # Gestion des arguments
    for opt,arg in opts:

        print(opts)
        #Entrainement du réseau
        if opt == '-t':
            if ('-s', '') in opts:
                train_data, test_data = data.split()
                train(train_data,sess,saver,accuracy,X,Y,p_keep_input,p_keep_hidden,train_op,2000)
            else :
                train(data,sess,saver,accuracy,X,Y,p_keep_input,p_keep_hidden,train_op,2000)

        #Evaluation du réseau
        if opt == '-e':
            if ('-s', '') in opts:
                evaluate(test_data,sess,saver,accuracy,X,Y,p_keep_input,p_keep_hidden)
            else:
                evaluate(data,sess,saver,accuracy,X,Y,p_keep_input,p_keep_hidden)
        

if __name__ == "__main__":
    main(sys.argv[1:])
