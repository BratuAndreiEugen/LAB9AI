import math

# actual = list of class names
# predicted = list of triplet probabilities [p1,p2,p3] p1+p2+p3 = 1.0
# map_to_int = dicitionary to map classes with positions in a triplet
def my_cross_entropy_loss(actual, predicted, map_to_int = None):
    # print("\nLoss pentru multi class : ")
    avg_loss = 0
    if map_to_int != None:
        for i in range(0, len(actual)):
            actual[i] = map_to_int[actual[i]]

    for i in range(len(actual)):
        # start with the first class
        ce = 0 # initialize entropy
        for j in range(len(predicted[i])):
            if actual[i] == j:
                loss_for_j = (-1)*1*math.log(predicted[i][j])
                ce += loss_for_j
            else:
                loss_for_j = 0 # nu se adauga nimic
        avg_loss += ce
    return avg_loss/len(actual)

# MULTI CLASS
# actual = ["Daisy", "Tulip", "Rose", "Daisy", "Tulip", "Rose"]
# map_to_int = {"Daisy" : 0, "Tulip" : 1, "Rose" : 2}
# predicted = [[0.8, 0.1, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6],
#                    [0.9, 0.05, 0.05], [0.3, 0.4, 0.3], [0.1, 0.7, 0.2]]
# #print(multiclass_cross_entropy_loss(actual, predicted, map_to_int))
# print(my_cross_entropy_loss(actual, predicted, map_to_int))