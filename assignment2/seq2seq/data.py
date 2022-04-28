import argparse
import pandas as pd
import warnings
import re
import tensorflow as tf
import tensorflow_text as tf_text
from sklearn.model_selection import train_test_split


#################################
# CONSTANTS
EOS = "[end]"
SOS = "[start]"
test_size = 0.2
BATCH_SIZE = 64
Max_Vocab_Size = 5000
embedding_dim = 256
units = 1024
#################################

warnings.filterwarnings("ignore")


class Train_Data:

    """
    Train_Data class Loads the data from multiple Training JSON files in Pandas Dataframes
    """

    def __init__(self, path, filenames):
        """
        Parameters
        ------------
        path: directory where English to SQL translation JSON are placed
        filenames: Json file names containing Train data
        """
        self.path = path
        self.filenames = filenames
        self.df_train = pd.DataFrame()
        for f in self.filenames:
            print("Reading file at path", self.path + f)
            try:
                df = pd.read_json(self.path + f)
                if len(self.df_train) == 0:
                    self.df_train = df
                else:
                    self.df_train = self.df_train.append(df)
                print("{} Rows in Total".format(len(self.df_train)))
            except Exception as e:
                print("Got error while Reading file : ", e)
        print("Filter Easy Queries")
        df = self.df_train
        df2 = df[
            ~df["query"].str.contains("join", case=False)
        ]  # filter queries with join
        df2 = df2[
            ~df2["query"].str.contains("group by", case=False)
        ]  # filter group by queries
        df2 = df2[
            ~df2["query"].str.contains("\);$", case=False)
        ]  # filter nested queries

        print("Splittin the Train and Test data")
        train, test = train_test_split(
            df2, test_size=0.2, random_state=50, stratify=df2["db_id"]
        )

        print(test.shape)
        self.df_train = train
        self.df_test = test

        print("Data for Training", self.df_train.shape)
        print("Data for Testing", self.df_test.shape)

    @property
    def questions(self):
        """
        Returns
        ------------
        Returns English Questions in Dataframe Rows as List
        """
        return self.df_train.question.values.tolist()

    @property
    def sql(self):
        """
        Returns
        ------------
        Returns SQL in Dataframe Rows as List
        """
        return self.df_train["query"].values.tolist()

    @property
    def test_questions(self):
        """
        Returns
        ------------
        Returns English Questions in Dataframe Rows as List
        """
        return self.df_test.question.values.tolist()

    @property
    def test_sql(self):
        """
        Returns
        ------------
        Returns SQL in Dataframe Rows as List
        """
        return self.df_test["query"].values.tolist()

    @property
    def question_tokens(self):
        """
        Returns
        ------------
        Returns English Question Tokens in Dataframe Rows as List
        """

        return self.df_train["question_toks"].values.tolist()

    @property
    def sql_tokens(self):
        """
        Returns
        ------------
        Returns SQL Query Tokens in Dataframe Rows as List
        """
        return self.df_train["query_toks"].values.tolist()

    def get_special_characters(self, list_of_text):
        """
        Parameters
        ------------
        list_of_text: Input List of Text
        Returns
        ------------
        Provides list of Special Characters in the text
        """
        return list(
            set(
                Preprocess.special_char("".join(["".join(ele) for ele in list_of_text]))
            )
        )

    def get_vocab_size(self, list_of_text):
        """
        Parameters
        ------------
        list_of_text: Input List of Text

        Returns
        ------------
        Vocabulary size or unique words in the corpus
        """
        word_list = []
        for sentence in list_of_text:
            for word in sentence.split():
                word = word.lower().strip()
                if word not in word_list:
                    word_list.append(word)
        return len(word_list), word_list


class Preprocess:
    ''' Class to prepare data before training'''
    def __init__(self, text):
        """
        Parameters
        ------------
        text : Input string
        Runs the text processing steps

        """
        self.processed_text = self.run_pipeline(text)

    def text_standardize(self, text):
        """
        Parameters
        ------------
        text : Input string

        Returns
        ------------
        -   Unicode normalization using NFKD method
        -   Lower Case text

        """
        text = tf_text.normalize_utf8(text, "NFKD")
        text = tf.strings.lower(text)
        return text

    def text_whitespace(self, text):
        """
        Parameters
        ------------
        text : Input string

        Returns
        ------------
        -   Remove $ and \\ special characters
        -   Add space around punctations
        -   Remove spaces around sentences

        """
        text = tf.strings.regex_replace(text, "[$\\\\]", "")
        text = tf.strings.regex_replace(text, "[.?!,¿()*:@><]", r" \0 ")
        text = tf.strings.strip(text)
        return text

    def add_SOS_EOS(self, text):
        """
        Parameters
        ------------
        text : Input string

        Returns
        ------------
        -   Add <SOS> and <EOS> tags to each sentence

        """
        text = tf.strings.join([SOS, text, EOS], separator=" ")
        return text

    @classmethod
    def special_char(cls, text):
        """
        Parameters
        ------------
        text : Input string

        Returns
        ------------
        -   Special Characters found in Text using Regular Expression
        """
        return re.findall(r"[\W]", text.replace(" ", ""))

    def run_pipeline(self, text):
        """
        Parameters
        ------------
        text : Input string

        Returns
        ------------
        Executes series of Text pre processing functions

        """
        text = self.text_standardize(text)
        text = self.text_whitespace(text)
        text = self.add_SOS_EOS(text)
        self.text = text
        return self.text


class Features:
    """
    Extracts text features from data
    """

    def tf_lower_and_split_punct(self, text):
        """
        Parameters
        ------------
        text : Input string

        Returns
        ------------
        Standardized Text

        """
        return Preprocess(text).processed_text

    def vectorizor(self, document, max_vocab_size):
        """
        Parameters
        ------------
        document : Collection of sentences
        max_vocab_size : No of words in document used for TextVectorization

        Returns
        ------------
        TextVectorization object

        """
        text_processor = tf.keras.layers.TextVectorization(
            standardize=self.tf_lower_and_split_punct, max_tokens=max_vocab_size
        )
        text_processor.adapt(document)
        print("Sample Vocabulary", text_processor.get_vocabulary()[:10])
        return text_processor


if __name__ == "__main__":
    sel = Train_Data(
        "D:/DS/Learnin/Essex MS/CE888/git/CE888/assignment2/seq2seq/spider/",
        ["train_spider.json", "train_others.json"],
    )
    print(sel.df_test.shape)
    print(sel.df_train.shape)
