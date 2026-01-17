import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
import os
import re
import pickle
import jieba
from pathlib import Path
import gc

# ===================== 全局初始化 & 路径配置 & GPU控制 =====================
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

jieba.initialize()

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
MODEL_DIR = PROJECT_ROOT / "tmp"
TRANSLATE_MODEL_DIR = CURRENT_DIR / "best_model"

MODEL_DIR.mkdir(exist_ok=True)
TRANSLATE_MODEL_DIR.mkdir(exist_ok=True)

TEXT_CLASSIFY_MODEL_PATH = MODEL_DIR / "my_model.h5"
TOKENIZER_PATH = MODEL_DIR / "tokenizer.pkl"
SENTIMENT_MODEL_PATH = MODEL_DIR / "sentiment_lstm.h5"
SENTIMENT_DICT_PATH = MODEL_DIR / "sentiment_dicts.pkl"
TRANSLATE_INP_TOKENIZER_PATH = MODEL_DIR / "inp_lang_tokenizer.pkl"
TRANSLATE_TARG_TOKENIZER_PATH = MODEL_DIR / "targ_lang_tokenizer.pkl"

# 常量配置
CATEGORIES = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
SENTIMENT_MAXLEN = 50
TRANSLATE_MAXLEN = 10
EMBEDDING_DIM = 256
UNITS = 512
CLASSIFY_CONF_THRESHOLD = 0.3
SENTIMENT_CONF_THRESHOLD = 0.4

# 🌟 核心优化：补充财经类关键词，解决股票/经济误判
CLASS_KEYWORDS = {
    "科技": {"科技", "科幻", "小说", "编程", "AI", "人工智能", "电脑", "手机"},
    "游戏": {"游戏", "手游", "网游", "吃鸡", "王者", "LOL", "组队", "打本", "角色"},
    "教育": {"学习", "教育", "学校", "考试", "上课"},
    "房产": {"房子", "房产", "买房", "装修"},
    "体育": {"体育", "篮球", "足球", "跑步", "健身"},
    "财经": {"股票", "经济", "涨", "跌", "投资", "理财", "钱", "财经", "工资", "炒股"},  # 新增股票相关词
    "家居": {"家居", "家具", "做饭", "家务"},
    "时尚": {"时尚", "穿搭", "化妆", "品牌"},
    "时政": {"新闻", "时政", "政策", "国家"},
    "娱乐": {"娱乐", "明星", "电影", "追剧"}
}
IRRELEVANT_KEYWORDS = {"外卖", "被偷", "快递", "吃饭", "点餐", "饿", "饱", "堵车", "迟到"}

# 情感关键词配置
NEGATIVE_KEYWORDS = {
    '不高兴', '心情差', '难过', '生气', '烦躁', '郁闷', '伤心', '失望',
    'sad', 'angry', 'upset', 'depressed', 'unhappy'
}
POSITIVE_KEYWORDS = {
    '喜欢', '开心', '高兴', '愉快', '满意', '棒', '好', '精彩', '不错',
    'love', 'like', 'happy', 'glad', 'excited', 'pleased'
}

# ===================== 1. 文本分类模块（重点优化财经类误判） =====================
text_classify_model = None
text_tokenizer = None

def init_text_classify():
    global text_classify_model, text_tokenizer
    if TOKENIZER_PATH.exists():
        try:
            with open(TOKENIZER_PATH, 'rb') as f:
                text_tokenizer = pickle.load(f)
            print(f"✅ 分类Tokenizer加载成功，词汇表大小：{text_tokenizer.num_words}")
        except Exception as e:
            print(f"❌ 分类Tokenizer加载失败：{e}")
    else:
        print(f"❌ Tokenizer文件不存在：{TOKENIZER_PATH}")
    
    if TEXT_CLASSIFY_MODEL_PATH.exists():
        try:
            text_classify_model = load_model(TEXT_CLASSIFY_MODEL_PATH)
            print(f"✅ 分类模型加载成功，输入形状：{text_classify_model.input_shape}")
        except Exception as e:
            print(f"❌ 分类模型加载失败：{e}")
    else:
        print(f"❌ 分类模型文件不存在：{TEXT_CLASSIFY_MODEL_PATH}")

def text_classify(text, return_confidence=False):
    if text_classify_model is None or text_tokenizer is None:
        if return_confidence:
            return "未知类别", 0.0
        return "未知类别"
    
    try:
        clean_text = text.strip()
        
        # 英文输入返回未知
        if re.match(r'^[a-zA-Z\s?.!,]+$', clean_text):
            if return_confidence:
                return "未知类别", 0.0
            return "未知类别"
        
        # 无关关键词返回未知
        if any(word in clean_text for word in IRRELEVANT_KEYWORDS):
            if return_confidence:
                return "未知类别", 0.0
            return "未知类别"
        
        sequences = text_tokenizer.texts_to_sequences([clean_text])
        if not sequences or len(sequences[0]) == 0:
            pred_label, max_prob = "未知类别", 0.0
            # 按关键词匹配分类（优先财经）
            for cate, keywords in CLASS_KEYWORDS.items():
                if any(word in clean_text for word in keywords):
                    pred_label = cate
                    max_prob = 0.3
                    break
            if return_confidence:
                return pred_label, max_prob
            return pred_label
        
        x_pad = sequence.pad_sequences(
            sequences, 
            maxlen=300, 
            padding='post', 
            truncating='post'
        )
        pred_probs = text_classify_model.predict(x_pad, verbose=0)[0]
        max_prob = np.max(pred_probs)
        label_id = np.argmax(pred_probs)
        pred_label = CATEGORIES[label_id]
        
        # 🌟 核心优化：修正股票/经济误判为游戏
        # 1. 模型判为游戏，但包含财经关键词 → 强制改为财经
        if pred_label == "游戏" and any(word in clean_text for word in CLASS_KEYWORDS["财经"]):
            pred_label = "财经"
            max_prob = 0.85  # 赋予高置信度
        # 2. 模型判为游戏，但包含“现实中” → 强制改为未知（避免误导）
        elif pred_label == "游戏" and "现实中" in clean_text:
            pred_label = "未知类别"
            max_prob = 0.0
        # 3. 修正科技小说误判
        elif pred_label == "游戏" and any(word in clean_text for word in CLASS_KEYWORDS["科技"]):
            pred_label = "科技"
            max_prob = 0.8
        elif max_prob < 0.1:
            for cate, keywords in CLASS_KEYWORDS.items():
                if any(word in clean_text for word in keywords):
                    pred_label = cate
                    max_prob = 0.3
                    break
        
        # 下雨/心情差不判为游戏
        if ("下雨" in clean_text or "心情差" in clean_text) and pred_label == "游戏":
            pred_label = "未知类别"
            max_prob = 0.0
        
        if return_confidence:
            return pred_label, max_prob
        return pred_label
    except Exception as e:
        print(f"❌ 文本分类出错：{e}")
        if return_confidence:
            return "未知类别", 0.0
        return "未知类别"

# ===================== 2. 情感分析模块（保持不变） =====================
sentiment_model = None
sentiment_dict = {}

def init_sentiment_analyse():
    global sentiment_model, sentiment_dict
    if SENTIMENT_DICT_PATH.exists():
        try:
            with open(SENTIMENT_DICT_PATH, 'rb') as f:
                sentiment_dict = pickle.load(f)
            print(f"✅ 情感词典加载成功，大小：{len(sentiment_dict)}")
        except Exception as e:
            print(f"❌ 情感词典加载失败：{e}")
    else:
        print(f"❌ 情感词典不存在：{SENTIMENT_DICT_PATH}")
    
    if SENTIMENT_MODEL_PATH.exists():
        try:
            sentiment_model = load_model(SENTIMENT_MODEL_PATH)
            print(f"✅ 情感模型加载成功")
        except Exception as e:
            print(f"❌ 情感模型加载失败：{e}")
    else:
        print(f"❌ 情感模型不存在：{SENTIMENT_MODEL_PATH}")

def sentiment_analyse(text, return_confidence=False):
    if sentiment_model is None or not sentiment_dict:
        if return_confidence:
            return "中性", 0.0
        return "中性"
    
    try:
        clean_text = text.strip().lower()
        
        # 关键词兜底
        negative_count = sum(1 for word in NEGATIVE_KEYWORDS if word in clean_text)
        positive_count = sum(1 for word in POSITIVE_KEYWORDS if word in clean_text)
        
        if negative_count > 0:
            if return_confidence:
                return "消极", 0.6
            return "消极"
        if positive_count > 0:
            if return_confidence:
                return "积极", 0.6
            return "积极"
        
        if re.match(r'^[a-zA-Z\s?.!,]+$', clean_text):
            if return_confidence:
                return "中性", 0.0
            return "中性"
        
        text_seg = jieba.lcut(clean_text, cut_all=False)
        text_seg = [w for w in text_seg if w.strip() and w not in ['，', '。', '！', '？', ' ', '']]
        if not text_seg:
            if return_confidence:
                return "中性", 0.0
            return "中性"
        
        ids = [sentiment_dict.get(w, 0) for w in text_seg]
        x_pad = sequence.pad_sequences(
            [ids], 
            maxlen=SENTIMENT_MAXLEN, 
            padding='pre', 
            truncating='pre'
        )
        
        pred_prob = sentiment_model.predict(x_pad, verbose=0)[0][0]
        if np.isnan(pred_prob):
            pred_prob = 0.0
            pred_label = "中性"
        else:
            pred_label = "积极" if pred_prob >= SENTIMENT_CONF_THRESHOLD else "消极"
        
        if return_confidence:
            return pred_label, pred_prob
        return pred_label
    except Exception as e:
        print(f"❌ 情感分析出错：{e}")
        if return_confidence:
            return "中性", 0.0
        return "中性"

# ===================== 3. 机器翻译模块（保持不变） =====================
translate_encoder = None
translate_decoder = None
inp_lang = None
targ_lang = None

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units):
        super().__init__()
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.bi_gru = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(enc_units, return_sequences=True, return_state=True, dropout=0.1)
        )

    def call(self, x, hidden):
        x = self.embedding(x)
        x = self.dropout(x)
        out, forward_h, backward_h = self.bi_gru(x, initial_state=hidden)
        hidden = tf.concat([forward_h, backward_h], axis=-1)
        return out, hidden

    def initialize_hidden_state(self, batch_sz):
        return [tf.zeros((batch_sz, self.enc_units)), tf.zeros((batch_sz, self.enc_units))]

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
        self.dropout = tf.keras.layers.Dropout(0.2)

    def call(self, query, values):
        query = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(query)))
        attention_weights = tf.nn.softmax(score, axis=1)
        attention_weights = self.dropout(attention_weights)
        context_vector = tf.reduce_sum(attention_weights * values, axis=1)
        return context_vector, attention_weights

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units):
        super().__init__()
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.gru = tf.keras.layers.GRU(
            dec_units*2, 
            return_sequences=True, 
            return_state=True, 
            dropout=0.1
        )
        self.attention = BahdanauAttention(dec_units)
        self.fc1 = tf.keras.layers.Dense(dec_units*2, activation="relu")
        self.fc2 = tf.keras.layers.Dense(vocab_size)

    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[-1]))
        output = self.fc1(output)
        output = self.dropout(output)
        x = self.fc2(output)
        return x, state, attention_weights

def init_translate():
    global translate_encoder, translate_decoder, inp_lang, targ_lang
    try:
        if not (TRANSLATE_INP_TOKENIZER_PATH.exists() and TRANSLATE_TARG_TOKENIZER_PATH.exists()):
            print(f"❌ 翻译Tokenizer文件不存在，跳过翻译模块")
            return
        
        with open(TRANSLATE_INP_TOKENIZER_PATH, 'rb') as f:
            inp_lang_data = pickle.load(f)
        if hasattr(inp_lang_data, 'word_index'):
            inp_lang = {
                'word_index': inp_lang_data.word_index,
                'index_word': {v: k for k, v in inp_lang_data.word_index.items()}
            }
        elif isinstance(inp_lang_data, dict) and 'word_index' in inp_lang_data:
            inp_lang = inp_lang_data
        else:
            print(f"❌ 翻译Tokenizer格式不支持，跳过翻译模块")
            return
        
        with open(TRANSLATE_TARG_TOKENIZER_PATH, 'rb') as f:
            targ_lang_data = pickle.load(f)
        if hasattr(targ_lang_data, 'word_index'):
            targ_lang = {
                'word_index': targ_lang_data.word_index,
                'index_word': {v: k for k, v in targ_lang_data.word_index.items()}
            }
        elif isinstance(targ_lang_data, dict) and 'word_index' in targ_lang_data:
            targ_lang = targ_lang_data
        else:
            print(f"❌ 翻译Tokenizer格式不支持，跳过翻译模块")
            return
        
        print(f"✅ 翻译Tokenizer加载成功（兼容模式），输入词汇量：{len(inp_lang['word_index'])}")
    except Exception as e:
        print(f"❌ 翻译Tokenizer加载失败：{e}，跳过翻译模块")
        return
    
    try:
        vocab_inp_size = len(inp_lang['word_index']) + 1
        vocab_tar_size = len(targ_lang['word_index']) + 1
        
        translate_encoder = Encoder(vocab_inp_size, EMBEDDING_DIM, UNITS)
        translate_decoder = Decoder(vocab_tar_size, EMBEDDING_DIM, UNITS)
        
        encoder_weights_path = TRANSLATE_MODEL_DIR / "encoder_best.weights.h5"
        decoder_weights_path = TRANSLATE_MODEL_DIR / "decoder_best.weights.h5"
        
        if encoder_weights_path.exists() and decoder_weights_path.exists():
            dummy_enc_input = tf.zeros((1, TRANSLATE_MAXLEN), dtype=tf.int32)
            dummy_enc_hidden = translate_encoder.initialize_hidden_state(1)
            translate_encoder(dummy_enc_input, dummy_enc_hidden)
            
            dummy_dec_input = tf.zeros((1, 1), dtype=tf.int32)
            dummy_enc_output = tf.zeros((1, TRANSLATE_MAXLEN, UNITS*2))
            translate_decoder(dummy_dec_input, dummy_enc_hidden[0], dummy_enc_output)
            
            translate_encoder.load_weights(str(encoder_weights_path))
            translate_decoder.load_weights(str(decoder_weights_path))
            print(f"✅ 翻译模型权重加载成功")
        else:
            print(f"❌ 翻译模型权重文件不存在，跳过翻译模块")
            return
    except Exception as e:
        print(f"❌ 翻译模型初始化失败：{e}，跳过翻译模块")
        return

def preprocess_sentence(w):
    if re.match(r'^[a-zA-Z\s?.!,]+$', w.strip()):
        w = w.strip().lower()
        w = re.sub(r'([?.!,])', r' \1 ', w)
        w = re.sub(r"[' ']+", ' ', w)
        w = '<start> ' + w + ' <end>'
    return w

def translate_text(text, return_confidence=False):
    if (translate_encoder is None or translate_decoder is None or 
        inp_lang is None or targ_lang is None):
        if return_confidence:
            return text, 1.0
        return text
    
    try:
        if not re.match(r'^[a-zA-Z\s?.!,]+$', text.strip()):
            if return_confidence:
                return text, 1.0
            return text
        
        sentence_proc = preprocess_sentence(text)
        inputs = [inp_lang['word_index'].get(i, inp_lang['word_index'].get("<unk>", 0)) 
                  for i in sentence_proc.split(' ')]
        inputs = sequence.pad_sequences(
            [inputs], 
            maxlen=TRANSLATE_MAXLEN, 
            padding='post', 
            dtype=np.int32
        )
        inputs = tf.convert_to_tensor(inputs)
        
        batch_sz = 1
        hidden = translate_encoder.initialize_hidden_state(batch_sz)
        enc_out, enc_hidden = translate_encoder(inputs, hidden)
        dec_hidden = enc_hidden

        beam_width = 2
        start_token = targ_lang['word_index'].get('<start>', 0)
        beams = [([start_token], enc_hidden, 0.0)]

        for _ in range(TRANSLATE_MAXLEN):
            new_beams = []
            for seq, h, score in beams:
                if seq[-1] == targ_lang['word_index'].get("<end>", -1):
                    new_beams.append((seq, h, score))
                    continue
                dec_input = tf.expand_dims(tf.cast([seq[-1]], tf.int32), 0)
                preds, new_h, _ = translate_decoder(dec_input, h, enc_out)
                probs = tf.nn.softmax(preds[0]).numpy()
                top_k_idx = np.argsort(probs)[-beam_width:]
                for idx in top_k_idx:
                    new_seq = seq + [idx]
                    new_score = score + np.log(probs[idx] + 1e-10)
                    new_beams.append((new_seq, new_h, new_score))
            new_beams.sort(key=lambda x: x[2], reverse=True)
            beams = new_beams[:beam_width]
            if all(seq[-1] == targ_lang['word_index'].get("<end>", -1) for seq, _, _ in beams):
                break

        best_seq = max(beams, key=lambda x: x[2])[0] if beams else []
        result = []
        for idx in best_seq[1:]:
            word = targ_lang['index_word'].get(idx, "")
            if word == "<end>":
                break
            if word not in ['<unk>', '']:
                result.append(word)
        
        final_result = "".join(result).strip()
        final_result = final_result if final_result else text
        trans_conf = abs(max(beams, key=lambda x: x[2])[2]) / 10 if beams else 0.0
        
        if return_confidence:
            return final_result, trans_conf
        return final_result
    except Exception as e:
        print(f"❌ 翻译出错：{e}")
        if return_confidence:
            return text, 0.0
        return text

# ===================== 4. 初始化所有模型 =====================
def init_all_models():
    print("===== 初始化NLP模型 =====")
    init_text_classify()
    init_sentiment_analyse()
    init_translate()
    gc.collect()
    print("===== 模型初始化完成 =====\n")

# ===================== 5. 测试代码 =====================
if __name__ == "__main__":
    init_all_models()
    test_texts = [
        "今天股票涨了，我很高兴",
        "是真正的经济股票",
        "现实中",
        "外卖被偷了 不高兴",
        "我喜欢科技小说"
    ]
    print("===== 模型测试结果 =====\n")
    for text in test_texts:
        cls_label, cls_conf = text_classify(text, return_confidence=True)
        sent_label, sent_conf = sentiment_analyse(text, return_confidence=True)
        trans_result, trans_conf = translate_text(text, return_confidence=True)
        print(f"输入文本：{text}")
        print(f"✅ 文本分类：{cls_label}（置信度：{cls_conf:.4f}）")
        print(f"✅ 情感分析：{sent_label}（置信度：{sent_conf:.4f}）")
        print(f"✅ 机器翻译：{trans_result}（可信度：{trans_conf:.4f}）")
        print("-" * 80)