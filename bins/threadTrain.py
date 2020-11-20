from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from sklearn.model_selection import train_test_split
from imblearn.ensemble import BalancedRandomForestClassifier

import threading
import time
import os
import sys

Accuracy = []
F1 = []
Recall = []
Precision = []

train_size = 0
complete = 0

def update_status():
	global complete 
	complete += 1
	update_progress(complete/train_size)

def update_progress(progress):
    barLength = 40
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rComputing: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), int(progress*100), status)
    sys.stdout.write(text)
    sys.stdout.flush()

def single_thread_train(thread_name, model, thread_args, dataset):
	print("{} launched".format(thread_name))
	T = time.time()
	for value in thread_args[0]:
		model = BalancedRandomForestClassifier(
            			n_estimators=500,
            			max_features='auto',
            			max_depth=None,
            			n_jobs=-1,
            			class_weight=None,
            			criterion='entropy',
            			min_samples_split=2,
            			min_samples_leaf=1)
	
		elements = thread_args[1][:value]
		X_train, X_test, y_train, y_test = train_test_split(dataset[elements], dataset['classALeRCE'], test_size=0.33, random_state=42)
		model.fit(X_train, y_train)
		y_pred = model.predict(X_test)
		precision = precision_score(y_test, y_pred, average = 'weighted')
		recall = recall_score(y_test, y_pred, average = 'weighted')
		f1 = f1_score(y_test, y_pred, average = 'weighted')
		accuracy = accuracy_score(y_test, y_pred)
		Accuracy.append((value, accuracy))
		F1.append((value, f1))
		Recall.append((value, recall))
		Precision.append((value, precision))
		update_status()
	deltaT = time.time() - T
	print("Finished {} in {} seconds".format(thread_name, deltaT))

def threaded_train(model, dataset, n_jobs = -1, rank = None): 
	seconds = time.time()
	print('Started at:', time.ctime(seconds), '\n')
	Threads = []
	count = os.cpu_count()
	if n_jobs < 0 and n_jobs != -1:
		raise Exception("Invalid n_jobs, must be -1 for all or bigger than 0")
	if n_jobs == -1 or n_jobs >= count:
		n_jobs = count
	if rank == None:
		return -1
	block_size = len(rank)//n_jobs
	global train_size
	train_size = len(rank)
	thread_args = (range(2, block_size), rank)
	args = ('Thread {}'.format(1), model, thread_args, dataset)
	Threads.append(threading.Thread(target=single_thread_train, args=args))
	Threads[0].start()  
	for i in range(1, n_jobs - 1):
		thread_args = (range(i*block_size,(i+1)*block_size), rank)
		args = ('Thread {}'.format(i + 1), model, thread_args, dataset)
		Threads.append(threading.Thread(target=single_thread_train, args=args))
		Threads[i].start()   
	thread_args = (range((n_jobs - 1)*block_size, len(rank)), rank)
	args = ('Thread {}'.format(n_jobs), model, thread_args, dataset)
	Threads.append(threading.Thread(target=single_thread_train, args=args))
	Threads[-1].start()
	update_progress(0)
	for i in range(n_jobs):
		Threads[i].join()
	return Accuracy, Precision, Recall, F1
