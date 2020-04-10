import tensorflow as tf
import modeling
import tokenization
pathname = "chinese_L-12_H-768_A-12/bert_model.ckpt"  # 模型地址
bert_config = modeling.BertConfig.from_json_file("chinese_L-12_H-768_A-12/bert_config.json")  # 配置文件地址。
configsession = tf.ConfigProto()
configsession.gpu_options.allow_growth = True
sess = tf.Session(config=configsession)
input_ids = tf.placeholder(shape=[None, None], dtype=tf.int32, name="input_ids")
input_mask = tf.placeholder(shape=[None, None], dtype=tf.int32, name="input_mask")
segment_ids = tf.placeholder(shape=[None, None], dtype=tf.int32, name="segment_ids")

with sess.as_default():
    model = modeling.BertModel(
        config=bert_config,
        is_training=True,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=False)
    encoder_last_layer = model.get_sequence_output()
    encoder_last2_layer = model.all_encoder_layers[-2]
    #saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())  # 这里尤其注意，先初始化，在加载参数，否者会把bert的参数重新初始化。这里和demo1是有区别的
    #saver.restore(sess, pathname)
    #print(1)
    token = tokenization.CharTokenizer(vocab_file="chinese_L-12_H-768_A-12/vocab.txt")
    query = u'美国大选，特朗普到底是咋想的，难道美国人民眼睛有问题吗？'
    split_tokens = token.tokenize(query)
    print(split_tokens)
    word_ids = token.convert_tokens_to_ids(split_tokens)
    print(word_ids)
    word_mask = [1] * len(word_ids)
    word_segment_ids = [0] * len(word_ids)
    fd = {input_ids: [word_ids], input_mask: [word_mask], segment_ids: [word_segment_ids]}
    last, last2 = sess.run([encoder_last_layer, encoder_last2_layer], feed_dict=fd)
    print(last)
    print(last2)
    print('last shape:{}, last2 shape: {}'.format(last.shape, last2.shape))
    pass


