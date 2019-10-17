import tensorflow as tf
import os
import cv2

global tfgraph

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_ocr_graph():
    global tfgraph
    return tfgraph

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph

def get_model_api():
    global tfgraph
    my_path = os.path.dirname(__file__)
    graph = load_graph("{}/exported-model-frozen/frozen_graph.pb".format(my_path))

    input = graph.get_tensor_by_name('prefix/input_image_as_bytes:0')
    prediction = graph.get_tensor_by_name('prefix/prediction:0')

    session_conf = tf.ConfigProto(
        device_count={'GPU' : 1},
        allow_soft_placement=True,
        log_device_placement=False
    )

    # We launch a Session
    sess = tf.Session(graph=graph, config=session_conf)
    #image = open(directory + "/" + filename, "rb")
    tfgraph = tf.get_default_graph()
    def model_api(image):
        image_string = cv2.imencode('.jpg',image)[1].tostring()
        out = sess.run(prediction, feed_dict={
            input: [image_string]
        })
        result = out.decode('utf-8')
        return result
    return model_api
