"""
词袋
"""
import config.corpus_config as config
import jieba
import os

def txt2sentence():
    no_word = [',','.','?','!','，','”',"？","！",":","；","：",";","……","——",
               '1','2','3','4','5','6','7','8','9','0','(','（',')','）',' ',"“","⑵",
               '⒊','⒉']
    sentences = []
    for i in range(1,5):
        in_file_path = config.corpus_path+"\\file"+str(i)+".txt"
        for sentence in open(in_file_path,'r',encoding='utf-8').readlines():
            text = ''.join([word.strip() for word in sentence.strip() if word.strip() not in no_word])
            for string in text.strip().split("。"):
                if string != '' and len(string)>8:
                    sentences.append(string)
    print(len(sentences))
    return sentences

def filp_sentence(sentences:list,user_stop = False)->list:
    sentences_array = []
    if os.path.exists(config.realnamed_path):
        jieba.load_userdict(config.realnamed_path)
    if user_stop:
        if not os.path.exists(config.stop_word_path):
            raise ValueError("停用词位置错误")
        stop_dict = [word.strip() for word in open(config.stop_word_path, 'r', encoding='utf-8').readlines()]
        for sentence in sentences:
            text = [word for word in jieba.lcut(sentence) if word not in stop_dict]
            sentences_array.append(text)
            print(text)
    else:
        for sentence in sentences:
            text = jieba.lcut(sentence)
            sentences_array.append(text)
            print(text)
    return sentences_array
if __name__ == '__main__':
    filp_sentence(txt2sentence())
    pass