import tensorflow as tf
import numpy as np
import os
import time
import cv2
import numpy as np
from tqdm import tqdm



def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def, 
            input_map=None, 
            return_elements=None, 
            name="prefix", 
            op_dict=None, 
            producer_op_list=None
        )

    input_name = graph.get_operations()[0].name+':0'
    output_name = graph.get_operations()[-1].name+':0'
    return graph, input_name, output_name




def prepare_data_x(img_dir,label=False,invert_color=False):
    inicio = time.time()
    tipos = ['cruzamento', 'curva', 'parada', 'reta', 'zigzag']
    aux = []
#     resultado = []

#     print(img_dir)
    frame = cv2.imread(img_dir,0)
    img = frame
#     img = cv2.resize(frame, (150,150))
    if invert_color:
        img = (255-img)
    aux.append(img)
    img = np.array(aux)
    img = img/255
    img = img.reshape(img.shape[0], img.shape[1], img.shape[2], 1).astype('float32')
#     print("tempo antes pred:",time.time()-inicio)
#     resultado.append(img)
#     return np.array(resultado)
    return img
#     pred = model.predict(img)
#     print("tempo pos pred:",time.time()-inicio)
#     pred = np.around(pred, decimals = 2, out = None) 
#     print("tempo TOTAL:",time.time()-inicio)
#     return tipos[np.argmax(pred)],pred,frame

def predict(model_path,img_path):
    # load tf graph
    tf_model,tf_input,tf_output = load_graph(model_path)
    print(tf_input,tf_output)
    # Create tensors for model input and output
    x = tf_model.get_tensor_by_name(tf_input)
    y = tf_model.get_tensor_by_name(tf_output) 
    num_outputs = y.shape.as_list()[0]
    
    fotos = (os.listdir(folder))
    predictions = np.zeros((len(fotos),len(tipos)))
    
    with tf.Session(graph=tf_model) as sess:
        for foto in tqdm(fotos):    
            tempo1 = time.time()
            y_out = sess.run(y, feed_dict={x: prepare_data_x(folder+foto)})
            print("tempo pred:",time.time()-tempo1)
            predictions = y_out[0]
            pred = np.around(predictions, decimals = 2, out = None) 
            print(tipos[np.argmax(pred)])
        #     return tipos[np.argmax(pred)],pred,input_data
#     return predictions


if __name__ == '__main__':
    model_path = "/home/junior/Documents/seguidor_linha/modelo_9935.pb"
    load_graph(model_path)
    predict(model_path,)

    main()