import os
from jieba import lcut, add_word
import tensorflow as tf

from Seq2Seq import Encoder, Decoder
from flask import Flask, render_template, request,jsonify

# 代码11-18 调用Flask前端
# 设置参数
data_path = '../data/ids'  # 数据路径
embedding_dim = 256  # 词嵌入维度
hidden_dim = 512  # 隐层神经元个数
checkpoint_path = '../tmp/model'  # 模型参数保存的路径
MAX_LENGTH = 50  # 句子的最大词长
CONST = {'_BOS': 0, '_EOS': 1, '_PAD': 2, '_UNK': 3}

# 聊天预测
def chat(sentence='你好'):
    # 初始化所有词语的哈希表
    table = tf.lookup.StaticHashTable(  # 初始化后即不可变的通用哈希表。
                initializer=tf.lookup.TextFileInitializer(
                    os.path.join(data_path, 'all_dict.txt'),
                    tf.string,
                    tf.lookup.TextFileIndex.WHOLE_LINE,
                    tf.int64,
                    tf.lookup.TextFileIndex.LINE_NUMBER
                ),  # 要使用的表初始化程序。有关支持的键和值类型，请参见HashTable内核。
                default_value=CONST['_UNK'] - len(CONST)  # 表中缺少键时使用的值。
            )

    # 实例化编码器和解码器
    encoder = Encoder(table.size().numpy() + len(CONST), embedding_dim, hidden_dim)
    decoder = Decoder(table.size().numpy() + len(CONST), embedding_dim, hidden_dim)
    optimizer = tf.keras.optimizers.Adam()  # 优化器
    # 模型保存路径
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
    # 导入训练参数
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))
    # 给句子添加开始和结束标记
    sentence = '_BOS' + sentence + '_EOS'
    # 读取字段
    with open(os.path.join(data_path, 'all_dict.txt'), 'r', encoding='utf-8') as f:
        all_dict = f.read().split()
    # 构建: 词-->id的映射字典
    word2id = {j: i+len(CONST) for i, j in enumerate(all_dict)}
    word2id.update(CONST)
    # 构建: id-->词的映射字典
    id2word = dict(zip(word2id.values(), word2id.keys()))
    # 分词时保留_EOS 和 _BOS
    for i in ['_EOS', '_BOS']:
        add_word(i)
    # 添加识别不到的词，用_UNK表示
    inputs = [word2id.get(i, CONST['_UNK']) for i in lcut(sentence)]
    # 长度填充
    inputs = tf.keras.preprocessing.sequence.pad_sequences(
        [inputs], maxlen=MAX_LENGTH, padding='post', value=CONST['_PAD'])
    # 将数据转为tensorflow的数据类型
    inputs = tf.convert_to_tensor(inputs)
    # 空字符串，用于保留预测结果
    result = ''

    # 编码
    enc_out, enc_hidden = encoder(inputs)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([word2id['_BOS']], 0)

    for t in range(MAX_LENGTH):
        # 解码
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)
        # 预测出词语对应的id
        predicted_id = tf.argmax(predictions[0]).numpy()
        # 通过字典的映射，用id寻找词，遇到_EOS停止输出
        if id2word.get(predicted_id, '_UNK') == '_EOS': 
            break
        # 未预测出来的词用_UNK替代
        result += id2word.get(predicted_id, '_UNK')
        dec_input = tf.expand_dims([predicted_id], 0)
    return result # 返回预测结果

# 实例化APP
app = Flask(__name__, static_url_path='/static')
@app.route('/message', methods=['POST'])

# 定义应答函数，用于获取输入信息并返回相应的答案
def reply():
    # 从请求中获取参数信息
    req_msg = request.form['msg']
    # 将语句使用结巴分词进行分词
    # req_msg = " ".join(jieba.cut(req_msg))
    # 调用decode_line对生成回答信息
    res_msg = chat(req_msg)
    # 将unk值的词用微笑符号代替
    res_msg = res_msg.replace('_UNK', '^_^')
    res_msg = res_msg.strip()
    # 如果接受到的内容为空，则给出相应的回复
    if res_msg == ' ': 
        res_msg = '我们来聊聊天吧'
    return jsonify({'text': res_msg})
@app.route("/")
# 在网页上展示对话
def index(): 
    return render_template('index.html')
# 启动APP
if (__name__ == '__main__'): 
    app.run(host='127.0.0.1', port=8808)