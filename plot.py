import pickle
import matplotlib.pyplot as plt

result = pickle.load(open('train_validation_result.pkl', 'rb'))
savefig = False

fig = plt.figure(figsize=(8, 6))
fig.set_facecolor('white')
ax = fig.add_subplot(111)
ln1 = plt.plot(result['epoch'], result['train_err'], color='blue', label='Train loss')
ln2 = plt.plot(result['epoch'], result['valid_err'], color='red', label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

ax2 = ax.twinx()
ln3 = plt.plot(result['epoch'], result['train_acc'], color='black', label='Train accuracy')
ln4 = plt.plot(result['epoch'], result['valid_acc'], color='green', label='Validation accuracy')
plt.ylabel('Accuracy')

lns = ln1 + ln2 + ln3 + ln4
labs = [l.get_label() for l in lns]
plt.legend(lns, labs, bbox_to_anchor=(1, 0.1), loc='lower right')
if savefig:
    plt.savefig('train_validation_result.pdf', dpi=300)
plt.show()
