import os
import re
import jieba
import numpy as np
import pandas as pd
from multiprocessing import freeze_support
import gensim
from gensim.models.ldamodel import LdaModel
import pyLDAvis.sklearn
from gensim.models import CoherenceModel
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from opencc import OpenCC


# 现在df中的'text'列中的中文繁体字已经被转为简体字


def top_words_data_frame(model: LatentDirichletAllocation,
                         tf_idf_vectorizer: TfidfVectorizer,
                         n_top_words: int) -> pd.DataFrame:
    '''
    求出每个主题的前 n_top_words 个词

    Parameters
    ----------
    model : sklearn 的 LatentDirichletAllocation
    tf_idf_vectorizer : sklearn 的 TfidfVectorizer
    n_top_words :前 n_top_words 个主题词

    Return
    ------
    DataFrame: 包含主题词分布情况
    '''
    rows = []
    feature_names = tf_idf_vectorizer.get_feature_names_out()
    for topic in model.components_:
        top_words = [feature_names[i]
                     for i in topic.argsort()[:-n_top_words - 1:-1]]
        rows.append(top_words)
    columns = [f'topic word {i + 1}' for i in range(n_top_words)]
    df = pd.DataFrame(rows, columns=columns)

    return df


def predict_to_data_frame(model: LatentDirichletAllocation, X: np.ndarray) -> pd.DataFrame:
    '''
    求出文档主题概率分布情况

    Parameters
    ----------
    model : sklearn 的 LatentDirichletAllocation
    X : 词向量矩阵

    Return
    ------
    DataFrame: 包含主题词分布情况
    '''
    matrix = model.transform(X)
    columns = [f'P(topic {i + 1})' for i in range(len(model.components_))]
    df = pd.DataFrame(matrix, columns=columns)
    return df


def calculate_coherence_score(dictionary, corpus, texts, limit, start=2, step=1):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        # 使用 LdaModel 替代 LatentDirichletAllocation
        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            iterations=50  # 可根据需要调整
        )
        model_list.append(lda_model)
        coherence_model = CoherenceModel(
            model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherence_model.get_coherence())

    return model_list, coherence_values


def main():
    # 待做 LDA 的文本 csv 文件，可以是本地文件，也可以是远程文件，一定要保证它是存在的！！！！
    # source_csv_path = 'comment.csv'
    source_excel_path = 'data/2022 lda.xlsx'
    # 文本 csv 文件里面文本所处的列名,注意这里一定要填对，要不然会报错的！！！
    document_column_name = 'content'
    # 输出主题词的文件路径
    top_words_csv_path = 'res/2022/top-topic-words.csv'
    # 输出各文档所属主题的文件路径
    predict_topic_csv_path = 'res/2022/document-distribution.csv'
    # 可视化 html 文件路径
    html_path = 'res/2022/document-lda-visualization.html'
    # 选定的主题数
    # n_topics = 3
    # 要输出的每个主题的前 n_top_words 个主题词数
    n_top_words = 15
    # 去除无意义字符的正则表达式
    pattern = u'[\\s\\d,.<>/?:;\'\"[\\]{}()\\|~!\t"@#$%^&*\\-_=+，。\n《》、？：；“”‘’｛｝【】（）…￥！—┄－]+'

    # 选择合适的转换配置，"t2s.json"表示繁体字到简体字的转换
    cc = OpenCC('t2s.json')

    # 读取Excel文件到DataFrame
    df = (
        pd.read_excel(
            source_excel_path)
            .drop_duplicates()
            .rename(columns={
            document_column_name: 'text'
        }))

    # 对'text'列中的文本进行繁体字到简体字的转换
    df['text'] = df['text'].apply(lambda x: cc.convert(x) if isinstance(x, str) else x)
    df['text'] = df['text'].apply(lambda x: re.sub(r'[a-zA-Z]', '', x) if isinstance(x, str) else x)
    # 设置停用词集合
    stop_words_set = {'你', '我', 'nan', '的', '什么', '怎么', '微博', '转发', '可以', '就是', '自己', '我们', '他们', '你们', '这个', '应该',
                      '现在', '核子', '真的', '一边', '如何', '不到', '决定', '意思', '看待', '试试', '各位', '查看', '一个', '没有', '因为', '但是',
                      '如果', '已经', '这样', '这么', '最后', '时候', '别人', '知道', '还是', '觉得', '不是', '很多', '开始', '一样', '一天', '还有',
                      '继续', '终于', '每天', '今天', '不能', '四个', '只有', '一些', '那么', '为什么', '所以', '一直', '其实', '为了', '每个', '然后',
                      '以后', '有点', '不会', '可能', '需要', '看到', '一起', '一定', '不要', '还要', '一句', '一樣', '这部分', '便是', '之一', '样子',
                      '以及', '一条', '感觉', '结果', '人待', '没订', '计较', '想待', '连话都', '拿到', '八点', '人比', '同意', '气到', '我少', '带来',
                      '好爽', '一个月', '他們', '很多很多', '花着', '中塞', '万多', '哈哈哈', '哈哈', '不得不', '能过', '学个', '一场', '只配', '再战',
                      '小吕', '优秀', '十岁', '无论', '状态', '拥有', '一路', '每次', '家里', '多久', '没法', '只为', '个钟', '卡巴', '巴斯', '截路',
                      '实苦', '绝不', '当成', '莫名', '来接', '不写', '这伙', '跑掉', '碰过', '店门', '混在', '嗷嗷', '托尼', '已到', '一手', '小日',
                      '替儿', '直起', '电五', '电一', '电二', '工师', '改了', ',周内', '铁打', '吃不上', '小齐', '齐齐', '确幸', '阿巴', '呜呼', '小鲜',
                      '停歇', '体打', '无情', '出没', '一瓶', '坐过', '一女', '一儿', '空亡', '石奶', '挨刀', '日主旺', '已失', '失水', '岛村', '财多,',

                      }
    # 去重、去缺失、分词
    df['cut'] = (
        df['text']
            .apply(lambda x: str(x))
            .apply(lambda x: re.sub(pattern, ' ', x))
            .apply(lambda x: " ".join([word for word in jieba.lcut(x) if word not in stop_words_set]))
    )

    # 构造 tf-idf
    tf_idf_vectorizer = TfidfVectorizer()
    tf_idf = tf_idf_vectorizer.fit_transform(df['cut'])

    # # 准备文本数据
    # text_data = df['cut'].apply(lambda x: x.split())
    #
    # # 创建字典和语料库
    # dictionary = gensim.corpora.Dictionary(text_data)
    # corpus = [dictionary.doc2bow(text) for text in text_data]
    #
    # # 设置计算一致性分数的范围
    # start, limit, step = 2, 10, 1
    # # 计算一致性分数
    # freeze_support()  # 添加这行代码
    # model_list, coherence_values = calculate_coherence_score(
    #    dictionary, corpus, text_data, limit, start, step)
    #
    # # 选择具有最高一致性分数的模型
    # best_model_index = np.argmax(coherence_values)
    # best_model = model_list[best_model_index]
    best_num_topics = 6

    # 输出最佳主题数
    print(f"Best Number of Topics: {best_num_topics}")

    # 使用最佳主题数建立模型
    lda = LatentDirichletAllocation(
        n_components=best_num_topics,
        max_iter=200,
        learning_method='online',
        learning_offset=50,
        random_state=0)

    # 使用 tf_idf 语料训练 lda 模型
    lda.fit(tf_idf)

    # 计算 n_top_words 个主题词
    top_words_df = top_words_data_frame(lda, tf_idf_vectorizer, n_top_words)

    # 保存 n_top_words 个主题词到 csv 文件中
    top_words_df.to_csv(top_words_csv_path, encoding='utf-8-sig', index=None)

    # 转 tf_idf 为数组，以便后面使用它来对文本主题概率分布进行计算
    X = tf_idf.toarray()

    # 计算完毕主题概率分布情况
    predict_df = predict_to_data_frame(lda, X)

    # 保存文本主题概率分布到 csv 文件中
    predict_df.to_csv(predict_topic_csv_path, encoding='utf-8-sig', index=None)

    # 使用 pyLDAvis 进行可视化
    data = pyLDAvis.sklearn.prepare(lda, tf_idf, tf_idf_vectorizer)
    pyLDAvis.save_html(data, html_path)
    # 清屏
    os.system('clear')
    # 浏览器打开 html 文件以查看可视化结果
    os.system(f'start {html_path}')

    print('本次生成了文件：',
          top_words_csv_path,
          predict_topic_csv_path,
          html_path)


if __name__ == '__main__':
    main()
