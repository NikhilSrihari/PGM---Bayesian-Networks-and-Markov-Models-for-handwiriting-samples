import numpy as np
import pandas as pd
import math
import time
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling, GibbsSampling
from pgmpy.estimators.K2Score import K2Score
from pgmpy.estimators import BayesianEstimator, HillClimbSearch
from pgmpy.inference.ExactInference import BeliefPropagation

data = {}
andRawData = None
task1_independenceValues = None
task1_independence = []
numberOfFeatures = 6
indThresholdList = [0.11,0.12,0.13,0.14,0.15,0.16,0.17]
task2_bms = []
task2_best_bm = None
task4_bms = []
task4_best_bm = None
task2_best_mm = None
task4_best_mm = None

def readAndProcessCSVFile_type1(filename):
	global data
	df = pd.read_csv(filename)
	cols = list(df) 
	for i in cols: 
		if (i!="Values"):
			temp=[]
			for index, row in (df[i]).iteritems():
				if(isinstance(row, str)):
					temp.append(float(row[0:row.find('%')])/100.0)
			data[i[0:2]]=temp


def readAndProcessCSVFile_type2(filename):
	global data
	df = pd.read_csv(filename)
	cols = list(df) 
	conditionedOn = cols[0]
	temp1=[]
	prevRowHeader_xIndex = None 
	for index1, row in df.iterrows():
		if(index1!=0):
			temp2=[]
			for index2, cell in (row).iteritems():
				if(cell.find('x')!=0):
					try:
						temp2.append(float(cell[0:cell.find('%')])/100.0)
					except:
						temp2.append(float(cell[0:cell.find('(')])/100.0)
				else:
					if(prevRowHeader_xIndex==cell[2]):
						temp1.append(temp2)
						data['x'+str(cell[2])+'|'+str(conditionedOn)]=temp1
					else:
						temp1 = []
						temp1.append(temp2)
						prevRowHeader_xIndex=cell[2]
	

def readAndProcessCSVFile_type3(filename):
	global andRawData
	df = pd.read_csv(filename)
	andRawData = df[['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9']]


def fetchData():
	readAndProcessCSVFile_type1('Table2.csv')
	readAndProcessCSVFile_type2('Table3.csv')
	readAndProcessCSVFile_type2('Table4.csv')
	readAndProcessCSVFile_type2('Table5.csv')
	readAndProcessCSVFile_type2('Table6.csv')
	readAndProcessCSVFile_type2('Table7.csv')
	readAndProcessCSVFile_type2('Table8.csv')
	readAndProcessCSVFile_type3('AND-Features(1).csv')


def task1():
	global data, task1_independenceValues, task1_independence, numberOfFeatures, indThresholdList
	task1_independenceValues = np.empty((numberOfFeatures,numberOfFeatures))
	for i in range(0,numberOfFeatures):
		task1_independenceValues[i][i] = float('nan')
		task1_independenceValues[i][i] = float('nan')
	for i in range(0,numberOfFeatures):
		for j in range(0,numberOfFeatures):
			if ((math.isnan(task1_independenceValues[i][j]))==False):
				if ('x'+str(i+1)+'|x'+str(j+1)) in data:
					temp1 = np.array(data[('x'+str(i+1)+'|x'+str(j+1))])
					temp2 = np.array(data['x'+str(j+1)])[np.newaxis]
					temp3 = np.array(data['x'+str(i+1)])[np.newaxis]
					temp4 = np.multiply(temp1, temp2)
					temp5 = np.transpose(np.dot(np.transpose(temp2), temp3))
					task1_independenceValues[i][j] = np.sum(np.absolute(np.subtract(temp4, temp5)))
					task1_independenceValues[j][i] = task1_independenceValues[i][j]
				elif ('x'+str(j+1)+'|x'+str(i+1)) in data:
					temp1 = np.array(data[('x'+str(j+1)+'|x'+str(i+1))])
					temp2 = np.array(data['x'+str(i+1)])[np.newaxis]
					temp3 = np.array(data['x'+str(j+1)])[np.newaxis]
					temp4 = np.multiply(temp1, temp2)
					temp5 = np.transpose(np.dot(np.transpose(temp2), temp3))
					task1_independenceValues[i][j] = np.sum(np.absolute(np.subtract(temp4, temp5)))
					task1_independenceValues[j][i] = task1_independenceValues[i][j]
				else:
					task1_independenceValues[i][j] = 0
					task1_independenceValues[j][i] = 0
	for i in range(0,numberOfFeatures):
		task1_independenceValues[i][i] = 0
		task1_independenceValues[i][i] = 0
	print(task1_independenceValues)
	for m in indThresholdList:
		task1_independence_temp = np.ones((numberOfFeatures,numberOfFeatures))
		for i in range(0,numberOfFeatures):
			for j in range(0,numberOfFeatures): 
				if (task1_independenceValues[i][j]>=m):
					task1_independence_temp[i][j]=0
		task1_independence.append({'threshold':m,'independence':task1_independence_temp})
	print(task1_independence)


def task2():
	global data, task2_bms, task2_best_bm
	k2Scores = []
	# Defining individual CPDs.
	cpd_x1 = TabularCPD(variable='x1', variable_card=len(data['x1']), values=[data['x1']])
	cpd_x2 = TabularCPD(variable='x2', variable_card=len(data['x2']), values=[data['x2']])
	cpd_x3 = TabularCPD(variable='x3', variable_card=len(data['x3']), values=[data['x3']])
	cpd_x4 = TabularCPD(variable='x4', variable_card=len(data['x4']), values=[data['x4']])
	cpd_x5 = TabularCPD(variable='x5', variable_card=len(data['x5']), values=[data['x5']])
	cpd_x6 = TabularCPD(variable='x6', variable_card=len(data['x6']), values=[data['x6']])
	# Model 1
	edgelist_temp = [('x4','x1'), ('x1','x6'), ('x6','x2'), ('x2','x3'), ('x3','x5')]
	model_temp = BayesianModel(edgelist_temp)
	cpd__x1 = TabularCPD(variable='x1', variable_card=len(data['x1']), 
               values=data['x1|x4'], evidence=['x4'], evidence_card=[len(data['x4'])])
	cpd__x6 = TabularCPD(variable='x6', variable_card=len(data['x6']), 
               values=data['x6|x1'], evidence=['x1'], evidence_card=[len(data['x1'])])
	cpd__x2 = TabularCPD(variable='x2', variable_card=len(data['x2']), 
               values=data['x2|x6'], evidence=['x6'], evidence_card=[len(data['x6'])])
	cpd__x3 = TabularCPD(variable='x3', variable_card=len(data['x3']), 
               values=data['x3|x2'], evidence=['x2'], evidence_card=[len(data['x2'])])
	cpd__x5 = TabularCPD(variable='x5', variable_card=len(data['x5']), 
               values=data['x5|x3'], evidence=['x3'], evidence_card=[len(data['x3'])])
	# Associating the CPDs with the network
	model_temp.add_cpds(cpd_x1, cpd_x2, cpd_x3, cpd_x4, cpd_x5, cpd_x6, cpd__x1, cpd__x6, cpd__x2, cpd__x3, cpd__x5)
	# check_model checks for the network structure and CPDs and verifies that the CPDs are correctly 
	# defined and sum to 1.
	model_temp.check_model()
	task2_bms.append(model_temp)
	k2Score = K2Score((BayesianModelSampling(model_temp)).forward_sample(size=1000))
	k2Scores_temp = k2Score.score(model_temp)
	k2Scores.append(k2Scores_temp)
	print("	Bayesian Model 1 K2 Accuracy Score is "+str(k2Scores_temp))
	# Model 2
	edgelist_temp = [('x1','x4'), ('x4','x6'), ('x6','x2'), ('x2','x3'), ('x3','x5')]
	model_temp = BayesianModel(edgelist_temp)
	cpd__x4 = TabularCPD(variable='x4', variable_card=len(data['x4']), 
               values=data['x4|x1'], evidence=['x1'], evidence_card=[len(data['x1'])])
	cpd__x6 = TabularCPD(variable='x6', variable_card=len(data['x6']), 
               values=data['x6|x4'], evidence=['x4'], evidence_card=[len(data['x4'])])
	cpd__x2 = TabularCPD(variable='x2', variable_card=len(data['x2']), 
               values=data['x2|x6'], evidence=['x6'], evidence_card=[len(data['x6'])])
	cpd__x3 = TabularCPD(variable='x3', variable_card=len(data['x3']), 
               values=data['x3|x2'], evidence=['x2'], evidence_card=[len(data['x2'])])
	cpd__x5 = TabularCPD(variable='x5', variable_card=len(data['x5']), 
               values=data['x5|x3'], evidence=['x3'], evidence_card=[len(data['x3'])])
	# Associating the CPDs with the network
	model_temp.add_cpds(cpd_x1, cpd_x2, cpd_x3, cpd_x4, cpd_x5, cpd_x6, cpd__x4, cpd__x6, cpd__x2, cpd__x3, cpd__x5)
	# check_model checks for the network structure and CPDs and verifies that the CPDs are correctly 
	# defined and sum to 1.
	model_temp.check_model()
	task2_bms.append(model_temp)
	k2Score = K2Score((BayesianModelSampling(model_temp)).forward_sample(size=1000))
	k2Scores_temp = k2Score.score(model_temp)
	k2Scores.append(k2Scores_temp)
	print("	Bayesian Model 2 K2 Accuracy Score is "+str(k2Scores_temp))
	# Model 3
	edgelist_temp = [('x4','x1'), ('x1','x6'), ('x6','x3'), ('x3','x2'), ('x2','x5')]
	model_temp = BayesianModel(edgelist_temp)
	cpd__x1 = TabularCPD(variable='x1', variable_card=len(data['x1']), 
               values=data['x1|x4'], evidence=['x4'], evidence_card=[len(data['x4'])])
	cpd__x6 = TabularCPD(variable='x6', variable_card=len(data['x6']), 
               values=data['x6|x1'], evidence=['x1'], evidence_card=[len(data['x1'])])
	cpd__x3 = TabularCPD(variable='x3', variable_card=len(data['x3']), 
               values=data['x3|x6'], evidence=['x6'], evidence_card=[len(data['x6'])])
	cpd__x2 = TabularCPD(variable='x2', variable_card=len(data['x2']), 
               values=data['x2|x3'], evidence=['x3'], evidence_card=[len(data['x3'])])
	cpd__x5 = TabularCPD(variable='x5', variable_card=len(data['x5']), 
               values=data['x5|x2'], evidence=['x2'], evidence_card=[len(data['x2'])])
	# Associating the CPDs with the network
	model_temp.add_cpds(cpd_x1, cpd_x2, cpd_x3, cpd_x4, cpd_x5, cpd_x6, cpd__x1, cpd__x6, cpd__x3, cpd__x2, cpd__x5)
	# check_model checks for the network structure and CPDs and verifies that the CPDs are correctly 
	# defined and sum to 1.
	model_temp.check_model()
	task2_bms.append(model_temp)
	k2Score = K2Score((BayesianModelSampling(model_temp)).forward_sample(size=1000))
	k2Scores_temp = k2Score.score(model_temp)
	k2Scores.append(k2Scores_temp)
	print("	Bayesian Model 3 K2 Accuracy Score is "+str(k2Scores_temp))
	# Model 4
	data['x1|x2']= np.divide(np.multiply(np.transpose(data['x2|x1']), np.transpose(np.array(data['x1'])[np.newaxis])), (np.array(data['x2'])[np.newaxis]))
	edgelist_temp = [('x5','x3'), ('x3','x2'), ('x2','x1'), ('x1','x6'), ('x6','x4')]
	model_temp = BayesianModel(edgelist_temp)
	cpd__x3 = TabularCPD(variable='x3', variable_card=len(data['x3']), 
               values=data['x3|x5'], evidence=['x5'], evidence_card=[len(data['x5'])])
	cpd__x2 = TabularCPD(variable='x2', variable_card=len(data['x2']), 
               values=data['x2|x3'], evidence=['x3'], evidence_card=[len(data['x3'])])
	cpd__x1 = TabularCPD(variable='x1', variable_card=len(data['x1']), 
               values=data['x1|x2'], evidence=['x2'], evidence_card=[len(data['x2'])])
	cpd__x6 = TabularCPD(variable='x6', variable_card=len(data['x6']), 
               values=data['x6|x1'], evidence=['x1'], evidence_card=[len(data['x1'])])
	cpd__x4 = TabularCPD(variable='x4', variable_card=len(data['x4']), 
               values=data['x4|x6'], evidence=['x6'], evidence_card=[len(data['x6'])])
	# Associating the CPDs with the network
	model_temp.add_cpds(cpd_x1, cpd_x2, cpd_x3, cpd_x4, cpd_x5, cpd_x6, cpd__x3, cpd__x2, cpd__x1, cpd__x6, cpd__x4)
	# check_model checks for the network structure and CPDs and verifies that the CPDs are correctly 
	# defined and sum to 1.
	model_temp.check_model()
	task2_bms.append(model_temp)
	k2Score = K2Score((BayesianModelSampling(model_temp)).forward_sample(size=1000))
	k2Scores_temp = k2Score.score(model_temp)
	k2Scores.append(k2Scores_temp)
	print("	Bayesian Model 4 K2 Accuracy Score is "+str(k2Scores_temp))
	# Model 5
	edgelist_temp = [('x5','x3'), ('x3','x2'), ('x2','x1'), ('x1','x4'), ('x4','x6')]
	model_temp = BayesianModel(edgelist_temp)
	cpd__x3 = TabularCPD(variable='x3', variable_card=len(data['x3']), 
               values=data['x3|x5'], evidence=['x5'], evidence_card=[len(data['x5'])])
	cpd__x2 = TabularCPD(variable='x2', variable_card=len(data['x2']), 
               values=data['x2|x3'], evidence=['x3'], evidence_card=[len(data['x3'])])
	cpd__x1 = TabularCPD(variable='x1', variable_card=len(data['x1']), 
               values=data['x1|x2'], evidence=['x2'], evidence_card=[len(data['x2'])])
	cpd__x4 = TabularCPD(variable='x4', variable_card=len(data['x4']), 
               values=data['x4|x1'], evidence=['x1'], evidence_card=[len(data['x1'])])
	cpd__x6 = TabularCPD(variable='x6', variable_card=len(data['x6']), 
               values=data['x6|x4'], evidence=['x4'], evidence_card=[len(data['x4'])])
	# Associating the CPDs with the network
	model_temp.add_cpds(cpd_x1, cpd_x2, cpd_x3, cpd_x4, cpd_x5, cpd_x6, cpd__x3, cpd__x2, cpd__x1, cpd__x4, cpd__x6)
	# check_model checks for the network structure and CPDs and verifies that the CPDs are correctly 
	# defined and sum to 1.
	model_temp.check_model()
	task2_bms.append(model_temp)
	k2Score = K2Score((BayesianModelSampling(model_temp)).forward_sample(size=1000))
	k2Scores_temp = k2Score.score(model_temp)
	k2Scores.append(k2Scores_temp)
	print("	Bayesian Model 5 K2 Accuracy Score is "+str(k2Scores_temp))
	task2_best_bm = task2_bms[k2Scores.index(max(k2Scores))]
	print("	Best Bayesian Model with the highest accuracy score is thus Model "+str(1+k2Scores.index(max(k2Scores))))


def task4():
	global andRawData, task4_best_bm
	k2Scores = []
	andRawData_temp = pd.DataFrame(andRawData.values, columns=['f1','f2','f3','f4','f5','f6','f7','f8','f9'])
	#Model 1
	est = HillClimbSearch(andRawData_temp, scoring_method=K2Score(andRawData_temp))
	model_temp = est.estimate()
	estimator = BayesianEstimator(model_temp, andRawData_temp)
	for fx in ['f1','f2','f3','f4','f5','f6','f7','f8','f9']:
		cpd_fx = estimator.estimate_cpd(fx, prior_type="K2")
		model_temp.add_cpds(cpd_fx)
	task4_bms.append(model_temp)
	print("	Model 1: Model through HillClimbSearch is : "+str(model_temp.edges()))
	k2Score = K2Score((BayesianModelSampling(model_temp)).forward_sample(size=1000))
	k2Scores_temp = k2Score.score(model_temp)
	k2Scores.append(k2Scores_temp)
	print("	Model 1: K2 Accuracy Score is "+str(k2Scores_temp))
	#Model 2: Manual Model based on HillClimbSearch
	model_temp = BayesianModel([('f3', 'f4'), ('f4', 'f9'), ('f3', 'f8'), ('f1', 'f7'), ('f5', 'f3'), ('f9', 'f8'), ('f1', 'f6'), ('f9', 'f1'), ('f9', 'f6'), ('f9', 'f2')])
	estimator = BayesianEstimator(model_temp, andRawData_temp)
	for fx in ['f1','f2','f3','f4','f5','f6','f7','f8','f9']:
		cpd_fx = estimator.estimate_cpd(fx, prior_type="K2")
		model_temp.add_cpds(cpd_fx)
	task4_bms.append(model_temp)
	print("	Model 2: Manual Model based on HillClimbSearch is : "+str(model_temp.edges()))
	k2Score = K2Score((BayesianModelSampling(model_temp)).forward_sample(size=1000))
	k2Scores_temp = k2Score.score(model_temp)
	k2Scores.append(k2Scores_temp)
	print("	Model 2: K2 Accuracy Score is "+str(k2Scores_temp))
	#Model 3: Manual Model based on HillClimbSearch
	model_temp = BayesianModel([('f3', 'f4'), ('f4', 'f9'), ('f3', 'f8'), ('f5', 'f7'), ('f5', 'f3'), ('f9', 'f8'), ('f1', 'f2'), ('f9', 'f1'), ('f9', 'f6'), ('f9', 'f2')])
	estimator = BayesianEstimator(model_temp, andRawData_temp)
	for fx in ['f1','f2','f3','f4','f5','f6','f7','f8','f9']:
		cpd_fx = estimator.estimate_cpd(fx, prior_type="K2")
		model_temp.add_cpds(cpd_fx)
	task4_bms.append(model_temp)
	print("	Model 3: Manual Model based on HillClimbSearch is : "+str(model_temp.edges()))
	k2Score = K2Score((BayesianModelSampling(model_temp)).forward_sample(size=1000))
	k2Scores_temp = k2Score.score(model_temp)
	k2Scores.append(k2Scores_temp)
	print("	Model 3: K2 Accuracy Score is "+str(k2Scores_temp))
	#Model 4: Manual Model based on HillClimbSearch
	model_temp = BayesianModel([('f3', 'f4'), ('f4', 'f9'), ('f5', 'f7'), ('f5', 'f3'), ('f1', 'f2'), ('f9', 'f1'), ('f9', 'f6'), ('f9', 'f8'),])
	estimator = BayesianEstimator(model_temp, andRawData_temp)
	for fx in ['f1','f2','f3','f4','f5','f6','f7','f8','f9']:
		cpd_fx = estimator.estimate_cpd(fx, prior_type="K2")
		model_temp.add_cpds(cpd_fx)
	task4_bms.append(model_temp)
	print("	Model 4: Manual Model based on HillClimbSearch is : "+str(model_temp.edges()))
	k2Score = K2Score((BayesianModelSampling(model_temp)).forward_sample(size=1000))
	k2Scores_temp = k2Score.score(model_temp)
	k2Scores.append(k2Scores_temp)
	print("	Model 4: K2 Accuracy Score is "+str(k2Scores_temp))
	#Model 5: Manual Model based on Intuition
	model_temp = BayesianModel([('f3', 'f4'), ('f4', 'f9'), ('f4', 'f7'), ('f1', 'f2'), ('f8', 'f5'), ('f9', 'f6'), ('f9', 'f8')])
	estimator = BayesianEstimator(model_temp, andRawData_temp)
	for fx in ['f1','f2','f3','f4','f5','f6','f7','f8','f9']:
		cpd_fx = estimator.estimate_cpd(fx, prior_type="K2")
		model_temp.add_cpds(cpd_fx)
	task4_bms.append(model_temp)
	print("	Model 5: Manual Model based on HillClimbSearch is : "+str(model_temp.edges()))
	k2Score = K2Score((BayesianModelSampling(model_temp)).forward_sample(size=1000))
	k2Scores_temp = k2Score.score(model_temp)
	k2Scores.append(k2Scores_temp)
	print("	Model 5: K2 Accuracy Score is "+str(k2Scores_temp))
	task4_best_bm = task4_bms[k2Scores.index(max(k2Scores))]
	print("	Best Bayesian Model with the highest accuracy score is thus Model "+str(1+k2Scores.index(max(k2Scores))))


def task3():
	global task4_best_bm, task2_best_bm, task2_best_mm, task4_best_mm
	st1 = time.time()
	task2_best_bm_samples = (BayesianModelSampling(task2_best_bm)).forward_sample(size=1000)
	et1 = time.time()
	diff1 = et1 - st1
	
	task2_best_bm_samplesC = task2_best_bm_samples.copy()
	task2_best_bm_samplesC.drop('x1', axis=1, inplace=True)
	task2_bm_predicted = task2_best_bm.predict(task2_best_bm_samplesC)
	
	task2_best_mm = task2_best_bm.to_markov_model()
	st2 = time.time()
	task2_best_mm_samples = (GibbsSampling(task2_best_mm)).sample(size=1000)
	et2 = time.time()
	diff2 = et2 - st2
	task2_best_mm_samples_values = (task2_best_mm_samples.values)
	task2_mm_predicted=[]
	task2_mmprop = BeliefPropagation(task2_best_mm)
	for i in range(1000):
		nik_temp = np.array(task2_best_mm_samples_values[i,:])
		try:
			task2_mm_predicted.append((task2_mmprop.map_query(variables=['x1'],
														evidence={
														'x2':int(nik_temp[2]),
														'x3':int(nik_temp[1]),
														'x4':int(nik_temp[5]),
														'x5':int(nik_temp[0]),
														'x6':int(nik_temp[4])
														})))
		except:
			task2_mm_predicted.append({'x1':-1})
	cnt1=0
	cnt2=0
	data1 = task2_best_mm_samples[['x1']].as_matrix()
	data2 = task2_best_bm_samples[['x1']].as_matrix()
	for i in range(1000):
		if(task2_mm_predicted[i]['x1']==int(data1[i])):
			cnt1=cnt1+1
		#if(task2_bm_predicted[i]['x1']==int(data2[i])):
			#cnt2=cnt2+1
	task2_mm_acc = cnt1/10.0
	task2_bm_acc = cnt2/10.0
	print("	Bayesian Model for 'th' data : "+str(task2_best_bm.edges()))
	print("	Bayesian Model for 'th' data takes time : "+str(diff1))
	#print("	Bayesian Model for 'th' data has accuracy : "+str(task2_mm_acc))
	print("	Markov Model for 'th' data : "+str(task2_best_mm.edges()))
	print("	Markov Model for 'th' data takes time : "+str(diff2))
	print("	Markov Model for 'th' data has accuracy : "+str(task2_mm_acc))

	st3 = time.time()
	task4_best_bm_samples = (BayesianModelSampling(task4_best_bm)).forward_sample(size=1000)
	et3 = time.time()
	diff3 = et3 - st3
	

	'''
	task4_best_bm_samplesC = task4_best_bm_samples.copy()
	task4_best_bm_samplesC.drop('f1', axis=1, inplace=True)
	#print(task4_best_bm_samplesC)
	task4_bm_predicted = task4_best_bm.predict(task4_best_bm_samplesC)
	#print(task4_bm_predicted)
	'''
	task4_best_mm = task4_best_bm.to_markov_model()
	st4 = time.time()
	task4_best_mm_samples = (GibbsSampling(task4_best_mm)).sample(size=1000)
	et4 = time.time()
	diff4 = et4 - st4
	'''print(task4_best_mm_samples)
	task4_best_mm_samples_values = (task4_best_mm_samples.values)
	task4_mm_predicted=[]
	task4_mmprop = BeliefPropagation(task4_best_mm)
	for i in range(1000):
		nik_temp = np.array(task4_best_mm_samples_values[i,:])
		print((nik_temp))
		try:
			task4_mm_predicted.append((task4_mmprop.map_query(variables=['f1'],
														evidence={
														'f2':int(nik_temp[2]),
														'f3':int(nik_temp[1]),
														'f4':int(nik_temp[5]),
														'f5':int(nik_temp[0]),
														'f6':int(nik_temp[4]),
														'f7':int(nik_temp[2]),
														'f8':int(nik_temp[1]),
														'f9':int(nik_temp[5])
														})))
		except:
			task4_mm_predicted.append({'f1':-1})
	cnt1=0
	cnt2=0
	data1 = task4_best_mm_samples[['f1']].as_matrix()
	data2 = task4_best_bm_samples[['f1']].as_matrix()
	for i in range(1000):
		if(task2_mm_predicted[i]['x1']==int(data1[i])):
			cnt1=cnt1+1
		if(task2_bm_predicted[i]['x1']==int(data2[i])):
			cnt2=cnt2+1
	task2_mm_acc = cnt1/10.0
	task2_bm_acc = cnt2/10.0'''
	print("	Bayesian Model for 'and' data : "+str(task4_best_bm.edges()))
	print("	Bayesian Model for 'and' data takes time : "+str(diff3))
	#print("	Bayesian Model for 'th' data has accuracy : "+str(task2_mm_acc))
	print("	Markov Model for 'and' data : "+str(task4_best_mm.edges()))
	print("	Markov Model for 'and' data takes time : "+str(diff4))
	#print("	Markov Model for 'th' data has accuracy : "+str(task2_mm_acc))
	

def main():
	fetchData()
	print("All Data Fetched")
	print("Task 1 - For 'th' data:")
	task1()
	print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
	print()
	print("Task 2 - For 'th' data:")
	task2()
	print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
	print()
	print("Task 4 - For 'and' data:")
	task4()
	print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
	print()
	print("Task 3 - For 'th' and 'and' data:")
	task3()
	print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
	print()


main()