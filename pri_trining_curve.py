import re
from seetings import settings
s = settings()
with open(s.log_path+"/training.log") as f:
    training_loss = []
    for line in f:
        match = re.search(r"([0-9]+\.[0-9]+)", line)
        if match:
            training_loss.append(float(match.group(1)))


with open(s.log_path+"/validation.log") as f:
    val_loss = []
    for line in f:
        match = re.search(r"([0-9]+\.[0-9]+)", line)
        if match:
            val_loss.append(float(match.group(1)))


import matplotlib.pyplot as plt
#next i want to draw the training_loss curve and the val_loss curve in only one graph with different color and add the legend
plt.plot(training_loss, label='training_loss', color='red')
plt.plot(val_loss, label='val_loss', color='blue')
plt.legend()
# we need to set the x and y label
plt.xlabel('epoch')
plt.ylabel('loss')
#we need to set the title
plt.title('training curve')
'''
#we need to set the x and y axis range
plt.xlim(0, 300)
plt.ylim(0, 0.5)
#we need to set the x and y axis ticks
plt.xticks(np.arange(0, 300, 50))
plt.yticks(np.arange(0, 0.5, 0.1))
#we need to set the grid
plt.grid(True)
#we need to show the graph
plt.show()
'''
# we need to save the graph
plt.savefig(s.log_path+"/training_curve.png")



