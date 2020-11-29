from bins.threadTrain import threaded_train as tt
import numpy as np

def order(data):
	val = []
	order = []
	for i in data:
		val.append(float(i[1]))
		order.append(int(i[0]))
	ret = []
	for i in range(len(order)):
		am = np.argmin(order)
		order.pop(am)
		ret.append(val.pop(am))
	return ret

def get_tree_data(dataset, ImportanceRank, train=False, pack_dict=None):
	if train:
		ret = tt(None, dataset, n_jobs=-1, pack_dict, rank=ImportanceRank)
		if ret == -1: 
			raise Exception('No rank, please insert rank and try again')
		ThreadAccuracy, ThreadPrecision, ThreadRecall, Threadf1 = ret
		with open('txt/Acc.txt', 'w') as acc:
			for element in ThreadAccuracy:
				write = '{},{}\n'.format(element[0], element[1])
				acc.write(write)
		with open('txt/Pre.txt', 'w') as Pre:
			for element in ThreadPrecision:
				write = '{},{}\n'.format(element[0], element[1])
				Pre.write(write)
		with open('txt/f1.txt', 'w') as f1:
			for element in Threadf1:
				write = '{},{}\n'.format(element[0], element[1])
				f1.write(write)
		with open('txt/Rec.txt', 'w') as Rec:
			for element in ThreadRecall:
				write = '{},{}\n'.format(element[0], element[1])
				Rec.write(write)
	else: 
		with open('txt/Acc.txt', 'r') as acc:
			f = acc.readlines()
			ThreadAccuracy = []
			for line in f:
				splitline = line.replace('\n', '').split(',')
				ThreadAccuracy.append(tuple(splitline))
		with open('txt/Pre.txt', 'r') as Pre:
			f = Pre.readlines()
			ThreadPrecision = []
			for line in f:
				splitline = line.replace('\n', '').split(',')
				ThreadPrecision.append(tuple(splitline))
		with open('txt/f1.txt', 'r') as f1:
			f = f1.readlines()
			Threadf1 = []
			for line in f:
				splitline = line.replace('\n', '').split(',')
				Threadf1.append(tuple(splitline))
		with open('txt/Rec.txt', 'r') as Rec:
			f = Rec.readlines()
			ThreadRecall = []
			for line in f:
				splitline = line.replace('\n', '').split(',')
				ThreadRecall.append(tuple(splitline))
	ThreadAccuracy = order(ThreadAccuracy)
	ThreadPrecision = order(ThreadPrecision)
	ThreadRecall = order(ThreadRecall)
	Threadf1 = order(Threadf1)
	return ThreadAccuracy, ThreadPrecision, ThreadRecall, Threadf1
