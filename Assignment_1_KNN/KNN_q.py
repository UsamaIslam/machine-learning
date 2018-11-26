import numpy as np
import operator


def load_dataset(trainData, trainLabel, testData, testLabel):
    with open(trainData) as train, open(trainLabel)as trainlabel, open(testData)as test, open(testLabel)as testlabel:
        trainingData = []
        testingData = []
        trainingSet = [list(map(int, line.strip().split(' '))) for line in train]
        #        print(trainingSet)
        trainingLabel = [line.rstrip('\n') for line in trainlabel]
        testingSet = [list(map(int, line.strip().split(' '))) for line in test]
        testingLabel = [line.rstrip('\n') for line in testlabel]
        [trainingData.append(np.asarray(contentTrain, dtype=np.uint8)) for contentTrain in trainingSet]
        [testingData.append(np.asarray(contentTest, dtype=np.uint8)) for contentTest in testingSet]

    return trainingData, testingData, trainingLabel, testingLabel


def calculateDistance(i1, i2):
    return np.linalg.norm(np.asarray(i1, dtype='float64') - np.asarray(i2, dtype='float64'))


def getNieghbors(trainingSet, testInstance, trainingLabel, k):
    distances = []
    for x in range(len(trainingSet)):
        dist = calculateDistance(testInstance, trainingSet[x])
        distances.append((trainingSet[x], trainingLabel[x], dist))
    distances.sort(key=operator.itemgetter(2))
    neighbors = []
    [neighbors.append((distances[x][0], distances[x][1])) for x in range(k)]
    return neighbors


def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        response = str(response)
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def getAccuracy(originalValues, predictions):
    correct = 0
    for x in range(len(originalValues)):
        correct += 1 if originalValues[x] == predictions[x] else False

    return (correct / float(len(originalValues))) * 100.0


def question_no_1(k:int):
    trainingData = r'data/trainX.txt'
    trainingLabel = r'data/trainY.txt'
    testingData = r'data/testX.txt'
    testingLabel = r'data/testY.txt'

    dataset = load_dataset(trainingData, trainingLabel, testingData, testingLabel)

    trainingDataset = dataset[0]
    testingDataset = dataset[1]
    trainingLabel = dataset[2]
    testingLabel = dataset[3]

    training_testing_dataset = trainingDataset + testingDataset
    training_testing_labels = trainingLabel + testingLabel

    predictions = []
    [predictions.append(getResponse(getNieghbors(trainingDataset, testInstance, trainingLabel, k))) for testInstance in training_testing_dataset]

    accuracy = getAccuracy(training_testing_labels, predictions)
    print('Overall Accuracy (using Training and Testing dataset): ', accuracy)
    print('Training Dataset: ', len(trainingDataset), 'Test Dataset: ', len(training_testing_dataset))

    training_dataset_ = list(zip(trainingDataset, trainingLabel))
    two_training_lable = []
    four_training_lable = []
    predictions_two = []
    predictions_four = []
    for i in range(len(training_dataset_)):
        if training_dataset_[i][1] == ' 2':
            neighbor_two = getNieghbors(trainingDataset, training_dataset_[i][0], trainingLabel, k)
            predictions_two.append(getResponse(neighbor_two))
            two_training_lable.append(training_dataset_[i][1])
        else:
            neighbor_four = getNieghbors(trainingDataset, training_dataset_[i][0], trainingLabel, k)
            predictions_four.append(getResponse(neighbor_four))
            four_training_lable.append(training_dataset_[i][1])

    accuracy_two = getAccuracy(two_training_lable, predictions_two)
    accuracy_four = getAccuracy(four_training_lable, predictions_four)

    print('training accuracy for class 2 is: ', accuracy_two)
    print('training accuracy for class 4 is: ', accuracy_four)

    testing_dataset_ = list(zip(testingDataset, testingLabel))
    two_testing_lable = []
    four_testing_lable = []
    predictions_two = []
    predictions_four = []
    for i in range(len(testing_dataset_)):
        if testing_dataset_[i][1] == ' 2':
            neighbor_two = getNieghbors(trainingDataset, testing_dataset_[i][0], trainingLabel, k)
            predictions_two.append(getResponse(neighbor_two))
            two_testing_lable.append(testing_dataset_[i][1])
        else:
            neighbor_four = getNieghbors(trainingDataset, testing_dataset_[i][0], trainingLabel, k)
            predictions_four.append(getResponse(neighbor_four))
            four_testing_lable.append(testing_dataset_[i][1])

    accuracy_two = getAccuracy(two_testing_lable, predictions_two)
    accuracy_four = getAccuracy(four_testing_lable, predictions_four)

    print('testing accuracy for class 2 is: ', accuracy_two)
    print('testing accuracy for class 4 is: ', accuracy_four)


if __name__ == '__main__':
    print('for k = 5')
    question_no_1(5)
    print('for k = 21')
    question_no_1(21)
    print('for k = 11')
    question_no_1(11)
