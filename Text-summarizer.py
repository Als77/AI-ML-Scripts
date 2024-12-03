"""
Text Summarization using TF-IDF

This script performs text summarization by analyzing the frequency of words in sentences using Term Frequency (TF) 
and Inverse Document Frequency (IDF) to identify and rank important sentences. The most relevant sentences 
are selected based on a calculated threshold to generate a concise summary of the input text.


Dependencies:
- nltk: Used for tokenization and stop word removal.
  Ensure you have the necessary NLTK resources downloaded:
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')

"""

import math

from nltk import sent_tokenize, word_tokenize, PorterStemmer
from nltk.corpus import stopwords

text_str = '''
Artificial Intelligence has been one of the most impactful technologies that has helped in shaping the modern world. It refers to the ability of machines to mimic human intelligence by performing tasks that typically require human cognition, such as learning, reasoning, problem-solving, and decision-making. AI is a branch of computer science that seeks to create machines capable of thinking and acting like humans. This technology is revolutionizing many industries and transforming our daily lives, making it a crucial topic of study, especially for school students.

Artificial Intelligence (AI) is one of the most impactful technologies shaping the modern world. It refers to the ability of machines to mimic human intelligence by performing tasks that typically require human cognition, such as learning, reasoning, problem-solving, and decision-making. AI is a branch of computer science that seeks to create machines capable of thinking and acting like humans. This technology is revolutionizing many industries and transforming our daily lives, making it a crucial topic of study, especially for school students. 

AI is used across many industries to improve efficiency and effectiveness. Let us take a look at some of the most common applications of AI. 

AI has made significant contributions to healthcare by helping doctors diagnose diseases faster and more accurately. AI-powered systems detect early signs of illnesses, analyse medical images, and even assist in surgeries. It also plays a crucial role in drug discovery and personalised medicine. Artificial intelligence has also made significant contributions to transportation. Companies like Tesla, Google, and Uber are using AI algorithms to improve the safety and reliability of autonomous vehicles. Additionally, AI is used in traffic management systems to reduce congestion and improve road safety.

AI is used to personalise recommendations on platforms like Netflix, YouTube, and Spotify, ensuring users have access to content that is according to their preferences. Even AI-driven gaming engines are also making video games more interactive and lifelike. AI-based tools are used in education to help students with personalised learning. Virtual tutors and AI-powered educational software are becoming popular to enhance student learning.

AI is also being used in financial institutions to detect fraudulent activities and automate trading processes. It also provides customer service through AI-powered chatbots. If we talk about the advantages of AI, then it is vast. It can process large amounts of data quickly and accurately, reducing human error and saving time. In industries like healthcare, it improves diagnosis and treatment, potentially saving lives. In business, AI helps companies become more efficient and profitable by taking up all the important tasks and completing them on time. In education, AI can provide personalised learning experiences, improving student outcomes.

Despite its many benefits, AI also raises several ethical concerns. One major issue is job displacement, as automation could potentially replace many jobs currently performed by humans. For example, self-driving cars could replace drivers, and automated systems in factories could reduce the need for human workers. There are concerns about data privacy too as AI systems often rely on large datasets, which can be misused. In conclusion, we can say that Artificial Intelligence is a rapidly evolving field that holds immense potential to change the way we live and work. While it offers numerous benefits, such as improving healthcare, education, and efficiency in industries, it also presents challenges that need to be addressed, including ethical concerns and job displacement. By using AI responsibly, society can make developments for the greater good, ensuring a better and more equitable future for everyone.
'''


def _create_frequency_table(text_string) -> dict:
    """
    we create a dictionary for the word frequency table.
    For this, we should only use the words that are not part of the stopWords array.

    Removing stop words and making frequency table
    Stemmer - an algorithm to bring words to its root word.
    :rtype: dict
    """
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text_string)
    ps = PorterStemmer()

    freqTable = dict()
    for word in words:
        word = ps.stem(word)
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    return freqTable


def _create_frequency_matrix(sentences):
    frequency_matrix = {}
    stopWords = set(stopwords.words("english"))
    ps = PorterStemmer()

    for sent in sentences:
        freq_table = {}
        words = word_tokenize(sent)
        for word in words:
            word = word.lower()
            word = ps.stem(word)
            if word in stopWords:
                continue

            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1

        frequency_matrix[sent[:15]] = freq_table

    return frequency_matrix


def _create_tf_matrix(freq_matrix):
    tf_matrix = {}

    for sent, f_table in freq_matrix.items():
        tf_table = {}

        count_words_in_sentence = len(f_table)
        for word, count in f_table.items():
            tf_table[word] = count / count_words_in_sentence

        tf_matrix[sent] = tf_table

    return tf_matrix


def _create_documents_per_words(freq_matrix):
    word_per_doc_table = {}

    for sent, f_table in freq_matrix.items():
        for word, count in f_table.items():
            if word in word_per_doc_table:
                word_per_doc_table[word] += 1
            else:
                word_per_doc_table[word] = 1

    return word_per_doc_table


def _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents):
    idf_matrix = {}

    for sent, f_table in freq_matrix.items():
        idf_table = {}

        for word in f_table.keys():
            idf_table[word] = math.log10(total_documents / float(count_doc_per_words[word]))

        idf_matrix[sent] = idf_table

    return idf_matrix


def _create_tf_idf_matrix(tf_matrix, idf_matrix):
    tf_idf_matrix = {}

    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):

        tf_idf_table = {}

        for (word1, value1), (word2, value2) in zip(f_table1.items(),
                                                    f_table2.items()):  # here, keys are the same in both the table
            tf_idf_table[word1] = float(value1 * value2)

        tf_idf_matrix[sent1] = tf_idf_table

    return tf_idf_matrix


def _score_sentences(tf_idf_matrix) -> dict:
    """
    score a sentence by its word's TF
    Basic algorithm: adding the TF frequency of every non-stop word in a sentence divided by total no of words in a sentence.
    :rtype: dict
    """

    sentenceValue = {}

    for sent, f_table in tf_idf_matrix.items():
        total_score_per_sentence = 0

        count_words_in_sentence = len(f_table)
        for word, score in f_table.items():
            total_score_per_sentence += score

        sentenceValue[sent] = total_score_per_sentence / count_words_in_sentence

    return sentenceValue


def _find_average_score(sentenceValue) -> int:
    """
    Find the average score from the sentence value dictionary
    :rtype: int
    """
    sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]

    # Average value of a sentence from original summary_text
    average = (sumValues / len(sentenceValue))

    return average


def _generate_summary(sentences, sentenceValue, threshold):
    sentence_count = 0
    summary = ''

    for sentence in sentences:
        if sentence[:15] in sentenceValue and sentenceValue[sentence[:15]] >= (threshold):
            summary += " " + sentence
            sentence_count += 1

    return summary


def run_summarization(text):
    """
    :param text: Plain summary_text of long article
    :return: summarized summary_text
    """

    '''
    We already have a sentence tokenizer, so we just need 
    to run the sent_tokenize() method to create the array of sentences.
    '''
    # 1 Sentence Tokenize
    sentences = sent_tokenize(text)
    total_documents = len(sentences)
    #print(sentences)

    # 2 Create the Frequency matrix of the words in each sentence.
    freq_matrix = _create_frequency_matrix(sentences)
    #print(freq_matrix)

    '''
    Term frequency (TF) is how often a word appears in a document, divided by how many words are there in a document.
    '''
    # 3 Calculate TermFrequency and generate a matrix
    tf_matrix = _create_tf_matrix(freq_matrix)
    #print(tf_matrix)

    # 4 creating table for documents per words
    count_doc_per_words = _create_documents_per_words(freq_matrix)
    #print(count_doc_per_words)

    '''
    Inverse document frequency (IDF) is how unique or rare a word is.
    '''
    # 5 Calculate IDF and generate a matrix
    idf_matrix = _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents)
    #print(idf_matrix)

    # 6 Calculate TF-IDF and generate a matrix
    tf_idf_matrix = _create_tf_idf_matrix(tf_matrix, idf_matrix)
    #print(tf_idf_matrix)

    # 7 Important Algorithm: score the sentences
    sentence_scores = _score_sentences(tf_idf_matrix)
    #print(sentence_scores)

    # 8 Find the threshold
    threshold = _find_average_score(sentence_scores)
    #print(threshold)

    # 9 Important Algorithm: Generate the summary
    summary = _generate_summary(sentences, sentence_scores, 1.3 * threshold)
    return summary


if __name__ == '__main__':
    result = run_summarization(text_str)
    print(result)