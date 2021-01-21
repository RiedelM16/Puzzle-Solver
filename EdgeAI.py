
import numpy as np

def standardzie(traind, testd):
    max = 0
    for d in traind:
        if max < d.shape[0]:
            max = d.shape[0]
    count = 0
    for d in traind:
        while d.shape[0] < max:
            d = np.append(d, [0])
        traind[count] = np.reshape(d, (1, max))
        count += 1
    count = 0
    for t in testd:
        while t.shape[0] < max:
            t = np.append(t, [0])
        testd[count] = np.reshape(t, (1, max))
        count += 1
    return max


def gettraindata():
    traintarget = list()
    traindata = list()
    efile = open("EdgeTrain.txt", "r")
    lines = efile.readlines()
    for line in lines:
        linelist = line.split()
        linelist = linelist[26:]
        linelist = [int(i) for i in linelist]
        row = np.array(linelist, dtype=int)
        traindata.append(row)
        target = np.zeros((1, 2))
        target[0, 1] = 1
        traintarget.append(target)
    efile.close()

    nefile = open("Non-EdgeTrain.txt", "r")
    lines = nefile.readlines()
    for line in lines:
        linelist = line.split()
        linelist = linelist[26:]
        linelist = [int(i) for i in linelist]
        row = np.array(linelist, dtype=int)
        traindata.append(row)
        target = np.zeros((1, 2))
        target[0, 0] = 1
        traintarget.append(target)
    nefile.close()
    return traindata, traintarget

def gettestdata():
    testtarget = list()
    testdata = list()

    efile = open("EdgeTrain.txt", "r")
    lines = efile.readlines()
    lcount = 0
    while lcount < 26:
        linelist = lines[lcount].split()
        linelist = [int(i) for i in linelist]
        row = np.array(linelist, dtype=int)
        testdata.append(row)
        target = np.zeros((1, 2))
        target[0, 1] = 1
        testtarget.append(target)
        lcount += 1
    efile.close()

    efile = open("Non-EdgeTrain.txt", "r")
    lines = efile.readlines()
    lcount = 0
    while lcount < 26:
        linelist = lines[lcount].split()
        linelist = [int(i) for i in linelist]
        row = np.array(linelist, dtype=int)
        testdata.append(row)
        target = np.zeros((1, 2))
        target[0, 0] = 1
        testtarget.append(target)
        lcount += 1
    efile.close()
    return testdata, testtarget

def softmax(z):
    if z[0, 0] > z[0, 1]:
        z[0, 0] = 1.
        z[0, 1] = 0.
    else:
        z[0, 0] = 0.
        z[0, 1] = 1.
    return z
    """
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div
    """





def WTestW(hiddenlayer, outlayer, td, tr):
    correct = 0
    inputv = td
    hiddenout = np.dot(inputv, hiddenlayer)
    y5 = np.maximum(hiddenout, 0)
    out = np.dot(y5, outlayer)
    inum = 0
    #print("Batch")
    for i in out:
        #print("Gusse: ", np.argmax(i), "   Actual: ", np.argmax(tr[inum]))
        if (np.argmax(tr[inum]) == np.argmax(i)):

            correct += 1
        inum += 1
    return correct / len(td)

def train(outlayer, hiddenlayer, epoc, hnodes, datasize, trdata, trtest, tedata, tetest):
    bsize = 2
    bnum = 0
    alpha = .0000005
    epochs = epoc
    ctar = 0
    bestcorrect = 0
    # print("Getting Train data")
    traind = trdata
    traint = trtest
    # print("Getting Test data")
    testd = tedata
    testt = tetest
    besthidden = hiddenlayer
    bestoutlayer = outlayer
    BEST = 0
    for e in range(epochs):

        rng_state = np.random.get_state()

        shuffledata = traind.copy()
        shuffletarget = traint.copy()
        np.random.shuffle(shuffledata)
        np.random.set_state(rng_state)
        np.random.shuffle(shuffletarget)
        ebest = 0

        print("epoch: ", e)
        for j in range(len(trdata) // bsize):
            td = shuffledata[j*bsize:j*bsize+bsize]
            tr = shuffletarget[j*bsize:j*bsize+bsize]
            # print("Batch Number: ", bnum)
            dw5 = np.zeros((datasize, hnodes), dtype=np.longlong)
            dw0 = np.zeros((hnodes, 2), dtype=np.longlong)
            bcorrect = 0

            for i in range(bsize):  # batch


                inputv = td[i]

                hiddenout = np.dot(inputv, hiddenlayer)

                y5 = np.maximum(hiddenout, 0)

                out = np.dot(y5, outlayer)

                y = softmax(out)

                # print(np.argmax(out))

                # BACK PROPAGATION
                delta = np.subtract(tr[i], y)  # tr = 1X10 y = 1x10

                wo = np.transpose(outlayer)  # out' = 10X100

                e5 = np.dot(delta, wo)

                delta5 = np.maximum(e5, 0)

                dw5 = np.add(dw5, np.dot(np.transpose(inputv), delta5))

                dw0 = np.add(dw0, np.dot(np.transpose(y5), delta))

            dw5 = np.divide(dw5, bsize)
            dw0 = np.divide(dw0, bsize)
            hiddenlayer = np.add(hiddenlayer, np.multiply(alpha, dw5))
            outlayer = np.add(outlayer, np.multiply(alpha, dw0))

            testcorrect = WTestW(hiddenlayer, outlayer, testd, testt)
            if (testcorrect > BEST):
                besthidden = hiddenlayer
                bestoutlayer = outlayer
                BEST = testcorrect
            if testcorrect > ebest:
                ebest = testcorrect

            bnum += 1

        # print("Best Precent at Epoch", e, ":", bestcorrect / 100 )
        print("Best Test Acurracy:", ebest)
        # test best W on test set

    print("Best Accuracy: ", WTestW(besthidden, bestoutlayer, testd, testt))
    print()
    return besthidden, bestoutlayer


def main():


    #nuralnet
    traind, traint = gettraindata()
    print("Got Train Data")
    testd, testt = gettestdata()
    print("Got Test Data")
    insize = standardzie(traind, testd)


    hiddenNodes = 10
    epochs = 1000

    print("Runing with", hiddenNodes, "Hidden Layer Neurons")
    hidden_layer = (np.random.rand(insize, hiddenNodes)) * 0.1 - 0.05
    out_layer = (np.random.rand(hiddenNodes, 2)) * 0.1 - 0.05
    hidden_layer.astype(np.longlong)
    out_layer.astype(np.longlong)
    bh, bo = train(out_layer, hidden_layer, epochs, hiddenNodes, insize, traind, traint, testd, testt)




main()


