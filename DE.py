# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 22:54:05 2019

@author: Rishav
"""
import numpy as np
from sklearn.externals import joblib

Scaling_Factor=0.1
CR_Factor=0.3
Less_Than_list=[0.5,0.1,0.2,0.1,0.1]
Greater_Than_list=[0.4,0.09,0.09,0.09,0.09]
PopulationSize=100
Number_of_Variables=int(5)
Data=joblib.load("xdata.pkl")
Yval=joblib.load("ydata.pkl")
Data=np.log(Data)
Yval=np.log(Yval)
for i in range(5):
    max_Data=max(Data[i,:])
    min_Data=min(Data[i,:])
    Data[i,:]=(Data[i,:]-min_Data)/(max_Data-min_Data)
max_YVal=max(Yval)
min_YVal=min(Yval)
Yval=(Yval-min_YVal)/(max_YVal-min_YVal)
def CheckConstraints(Variables):
    for i in range(len(Variables)):
        if Variables[i]<Greater_Than_list[i] or Variables[i]>Less_Than_list[i]:
            return False
    return True
def CreateInitialPopulation(PopulationSize,Number_of_Variables,Less_than_constraints,Greater_than_constraints):
    InitialPop=[]
    for each_sol in range(PopulationSize):
        sol=[]
        for dim in range(Number_of_Variables):
            dim_sol=(Less_than_constraints[dim]-Greater_than_constraints[dim])*np.random.random_sample()+Greater_than_constraints[dim]
            sol.append(dim_sol)
        InitialPop.append(sol)
    return np.array(InitialPop)
def CreateNewPopulation(Initial_pop_of_Gen):
    New_Population=[]
    for each_sol in Initial_pop_of_Gen:
        while(True):
            index_of_a=0
            index_of_b=0
            index_of_a=np.random.randint(0,len(Initial_pop_of_Gen))
            index_of_b=np.random.randint(0,len(Initial_pop_of_Gen))
            a=Initial_pop_of_Gen[index_of_a]
            b=Initial_pop_of_Gen[index_of_b]
            each_sol_new_gen=each_sol+Scaling_Factor*(a-b)
            CR=np.random.rand(0,1)
            if(CheckConstraints(each_sol_new_gen)):
                if(CR>CR_Factor):
                    New_Population.append(each_sol_new_gen)
                    break
                else:
                    New_Population.append(each_sol)
                    break
        return np.array(New_Population)

def CheckFitness(Sol):
    #print(Sol.shape)
    Sol=Sol.reshape(1,5)
    Y_pred=Sol.dot(Data)
    Fitness=np.sum(Yval-Y_pred)/Data.shape[0]
    Fitness=Fitness**2
    return Fitness

def DoSearch(Number_of_Gen,error_percentage):
    Pop=CreateInitialPopulation(PopulationSize,Number_of_Variables,Less_Than_list,Greater_Than_list)
    itr=0
    best_Fitnessof_FirstGen=CheckFitness(Pop[0])
    while True:
        itr+=1
        #print(itr)
        best_sol=Pop[0]
        best_Fitness=CheckFitness(Pop[0])
        for each_sol in Pop:
            Fitness=CheckFitness(each_sol) #Lesser The Fitness Value Better The Solution (Minimization)
            if(Fitness<best_Fitness):
                best_sol=each_sol
                best_Fitness=Fitness
        if(abs(best_Fitnessof_FirstGen-best_Fitness)<=error_percentage or itr>Number_of_Gen):
            
            return best_sol
        else:
            best_Fitnessof_FirstGen=best_Fitness
            Pop=CreateNewPopulation(Pop)
        
Vals=DoSearch(100,0.0001) 
print(Vals)       