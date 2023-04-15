class Word2Vec_Bi_SVM_Update_class:

    def __init__(self, pretrainedDir, entireCorpus, trainFile, testFile, outFile, epoch):
        self.pretrainedDir = pretrainedDir
        self.entireCorpus = entireCorpus
        self.trainFile = trainFile
        self.testFile = testFile
        self.outFile = outFile
        self.epoch = epoch

    def processing(self):


        entireCorpus = self.entireCorpus
        entireFr = open(entireCorpus, 'r')
        entireContents = entireFr.readlines()
        entireFr.close()

        import re

        entireList = []
        entireset = set()

        for entireContent in entireContents:
            entireContent = re.sub('[^가-힣]', ' ', entireContent)
            entireContent = re.sub('[\s]+', ' ', entireContent)
            entireContent = entireContent.strip()
            entireSplitList = entireContent.split(" ")
            entireList.append(entireSplitList)
            for eachset in entireSplitList:
                entireset.add(eachset)


        print(len(entireList))
        print(len(entireset))

        # entireDict = dict()
        # for eachset in entireset:
        #     entireDict[eachset] = 0
        #
        # for entireContent in entireContents:
        #     entireContent = re.sub('[^가-힣]', ' ', entireContent)
        #     entireContent = re.sub('[\s]+', ' ', entireContent)
        #     entireContent = entireContent.strip()
        #     entireSplitList = entireContent.split(" ")
        #     for eachset in entireSplitList:
        #         entireDict[eachset] = entireDict.get(eachset) + 1
        #
        # entireDictSorted = dict(sorted(entireDict.items(), key=lambda x: x[1], reverse=True))
        #
        # DictNum = 0
        # for key, value in entireDictSorted.items():
        #     # print(key, value)
        #     if value > 355:
        #         # print(key, value)
        #         DictNum = DictNum + 1
        # print(DictNum)



        #ko
        import pandas as pd
        import gensim

        pretrainedDir = self.pretrainedDir
        model = gensim.models.Word2Vec.load(pretrainedDir)

        print(len(list(model.wv.vocab.keys())))

        model.build_vocab(entireList, update=True)


        model.train(entireList, total_examples=model.corpus_count,  word_count=0, epochs=self.epoch)


        words = list(model.wv.vocab.keys())
        print(len(words))
        # vectors = []
        #
        # pretainedNum = 0
        # for word in words:
        #     vectors.append(model[word])
        #
        #
        # embedded_matrix = pd.DataFrame(vectors, index=words)
        # #
        # # print(embedded_matrix)
        # # #
        # from sklearn.manifold import TSNE
        # tsne = TSNE(n_components=1, random_state=0)
        # tsne_embedded_matrix = tsne.fit_transform(embedded_matrix)
        # #
        # # print(tsne_embedded_matrix)
        # #
        # embededDic = {}
        # for i in range(0, len(tsne_embedded_matrix)):
        #     embededDic[words[i]] = tsne_embedded_matrix[i][0]
        # print("embed done!")
        #
        #
        #
        #
        #
        #
        # fileDir = self.trainFile
        # fr = open(fileDir, 'r')
        # contents = fr.readlines()
        # fr.close()
        #
        # train = pd.DataFrame(columns=('Label', 'Sentence'))
        # i = 0
        # label = ""
        # sentence = ""
        # for content in contents:
        #     if i == 0:
        #         pass
        #     else:
        #         infos = content.split(",")
        #         label = int(infos[2])
        #         sentence = infos[3].replace("\n", "")
        #         train.loc[i] = [label, sentence]
        #     i = i + 1
        #
        # train['Sentence'] = train['Sentence'].str.replace(r'[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》\\n\t]+', " ",regex=True)
        # train['Sentence'] = train['Sentence'].str.replace(r'\t+', " ", regex=True)
        # train['Sentence'] = train['Sentence'].str.replace(r'[\\n]+', " ", regex=True)
        # trainList = train['Sentence'].tolist()
        # trainLabel = train['Label'].tolist()
        #
        # #문장 정제
        # trainlines = []
        # from konlpy.tag import Kkma
        # kkma = Kkma()
        #
        # for trainEach in trainList:
        #     posEachList = kkma.pos(trainEach)
        #     eachString = ""
        #     for posEach in posEachList:
        #         eachString = eachString + " " + posEach[0]
        #     eachString = eachString.strip()
        #     trainlines.append(eachString)
        #
        # # test
        # testDir = self.testFile
        # testfr = open(testDir, 'r')
        # testContents = testfr.readlines()
        # testfr.close()
        #
        # test = pd.DataFrame(columns=('Label1', 'Label2', 'Sentence'))
        # i = 0
        # label = ""
        # sentence = ""
        # for content in testContents:
        #     if i == 0:
        #         pass
        #     else:
        #         infos = content.split(",")
        #         label1 = int(infos[0])
        #         label2 = int(infos[1])
        #         sentence = infos[2].replace("\n", "")
        #         test.loc[i] = [label1, label2, sentence]
        #     i = i + 1
        #
        # test['Sentence'] = test['Sentence'].str.replace(r'[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》\\n\t]+', " ", regex=True)
        # test['Sentence'] = test['Sentence'].str.replace(r'\t+', " ", regex=True)
        # test['Sentence'] = test['Sentence'].str.replace(r'[\\n]+', " ", regex=True)
        # testList = test['Sentence'].tolist()
        # testLabel = test['Label1'].tolist()
        #
        # # 문장 정제
        # testLines = []
        # from konlpy.tag import Kkma
        # kkma = Kkma()
        #
        # for testEach in testList:
        #     posEachList = kkma.pos(testEach)
        #     eachString = ""
        #     for posEach in posEachList:
        #         eachString = eachString + " " + posEach[0]
        #     eachString = eachString.strip()
        #     testLines.append(eachString)
        #
        #
        # # https://www.kaggle.com/gabrielaltay/word-vectors-from-pmi-matrix
        #
        # # https://www.kaggle.com/gabrielaltay/word-vectors-from-pmi-matrix
        #
        # # install packages
        # from collections import Counter
        #
        # headlines = trainlines + testLines
        #
        # headlines = [[tok for tok in headline.split()] for headline in headlines]
        # # remove single word headlines
        # # 하나의 단어로 구성된 문장은 제거한다.
        # headlines = [hl for hl in headlines if len(hl) > 1]
        # # show results
        # # 결과 확인
        # print(headlines[0:20])
        #
        # # calculate a unigram vocabulary
        # # 단일 단어 사전 생성 (얼마나 많은 타입의 단어를 포함하는가?)
        # tok2indx = dict()
        # unigram_counts = Counter()
        # for ii, headline in enumerate(headlines):
        #     if ii % 200000 == 0:
        #         print(f'finished {ii / len(headlines):.2%} of headlines')
        #     for token in headline:
        #         unigram_counts[token] += 1
        #         if token not in tok2indx:
        #             tok2indx[token] = len(tok2indx)
        # indx2tok = {indx: tok for tok, indx in tok2indx.items()}
        # print('done')
        # print('vocabulary size: {}'.format(len(unigram_counts)))
        # print('most common: {}'.format(unigram_counts.most_common(10)))
        #
        # # 단어의 타입 수
        # wordType = len(unigram_counts)
        # print(wordType)
        #
        # ##############################결과구문으로출력###################################
        # def outreault(guess):
        #     guess = int(guess)
        #     outConstruction = ""
        #     if guess == 0:
        #         outConstruction = "agent-first"
        #     elif guess == 1:
        #         outConstruction = "theme-first"
        #
        #     return outConstruction
        #
        #
        #
        #
        #
        #
        # f = open(self.outFile + "word2vecBi_SVM_Update_epoch_"+str(self.epoch)+".csv", 'w')
        # f.write("window,sentence,originalLabel,predictedLabel,predictedConstruction,result" + "\n")
        #
        #
        #
        # totalVecList = []
        # for eachSentence in trainlines:
        #     eachVecList = []
        #     # print(eachSentence)
        #     for typeeach in indx2tok:
        #         # print(indx2tok[typeeach])
        #         if indx2tok[typeeach] in eachSentence:
        #             try:
        #                 eachVecList.append(embededDic[indx2tok[typeeach]])
        #             except KeyError:
        #                 eachVecList.append(0)
        #         else:
        #             eachVecList.append(0)
        #     # eachVecList = tuple(eachVecList)
        #     totalVecList.append(eachVecList)
        #
        # print(totalVecList)
        # print(trainLabel)
        #
        #
        #
        #
        # #https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=slykid&logNo=221630584607
        #
        # from sklearn.pipeline import Pipeline
        # from sklearn.preprocessing import StandardScaler
        # from sklearn.svm import LinearSVC
        #
        # svm_clf = Pipeline([("scaler", StandardScaler()), ("linear_svc", LinearSVC(C=1, loss="hinge"))])
        # svm_clf = svm_clf.fit(totalVecList, trainLabel)
        #
        # # Train the model using the training sets]
        #
        # totalNum = 0
        # correctNum = 0
        # for each in range(0, len(testLines)):
        #     print(test['Label1'][each + 1], test['Label2'][each + 1], test['Sentence'][each + 1])
        #     eachSentence = testLines[each]
        #     eachVecList = []
        #     # print(eachSentence)
        #     for typeeach in indx2tok:
        #         # print(indx2tok[typeeach])
        #         if indx2tok[typeeach] in eachSentence:
        #             try:
        #                 eachVecList.append(embededDic[indx2tok[typeeach]])
        #             except KeyError:
        #                 eachVecList.append(0)
        #         else:
        #             eachVecList.append(0)
        #
        #     sentence = str(test['Sentence'][each + 1])
        #     guess = svm_clf.predict([eachVecList])
        #     guess = str(guess[0])
        #
        #     if guess == str(test['Label1'][each + 1]) or guess == str(test['Label2'][each + 1]):
        #         print("input: ", sentence, ", predict: ", guess, "(O)")
        #         f.write(sentence + "," + str(test['Label1'][each + 1]) + "or" + str(
        #             test['Label2'][each + 1]) + "," + guess + "," + outreault(guess) + ",1" + "\n")
        #         correctNum = correctNum + 1
        #     else:
        #         f.write(sentence + "," + str(test['Label1'][each + 1]) + "or" + str(
        #             test['Label2'][each + 1]) + "," + guess + "," + outreault(guess) + ",0" + "\n")
        #         print("input: ", sentence, ", predict: ", guess, "(X)")
        #     totalNum = totalNum + 1
        #
        # print("totalNum: ", totalNum, " correctNum: ", correctNum, " accuracy: ", (correctNum / totalNum))
        #
        # f.close()









