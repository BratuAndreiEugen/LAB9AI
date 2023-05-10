import csv

def loadInput(fileName, inputVariabName):
    data = []
    dataNames = []
    with open(fileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                dataNames = row
            else:
                data.append(row)
            line_count += 1
    selectedVariable = dataNames.index(inputVariabName)
    inputs = []
    for i in range(0, len(data)):
        if data[i][selectedVariable] == '':
            inputs.append(0.0)
        else:
            inputs.append(float(data[i][selectedVariable]))

    return inputs

def loadOutput(fileName, outputVariabName):
    data = []
    dataNames = []
    with open(fileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                dataNames = row
            else:
                data.append(row)
            line_count += 1

    selectedOutput = dataNames.index(outputVariabName)
    outputs = []
    for i in range(0, len(data)):
        outputs.append(data[i][selectedOutput])

    return outputs