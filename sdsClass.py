'''

Authors: Samudrala, Plaza
Date: 02/2018
File name: sdsClass.py

This program uses the Stochastic Diffusion Search (sds) alg. .
The purpose of using sds is to solve the solution of imbalanced datasets.
It takes a majority class and undersamples it and takes an minority class
and oversamples it.
There are three phases in this alg.:
    -Initialisation --> 
    -Test
    -Diffusion

'''

############ IMPORTS ###############

import numpy as np
from sklearn.preprocessing import StandardScaler

###################################

#creating instance of imported StandardScaler
scaler = StandardScaler()  


class SDS:
    '''This class contains the SDS alg. 
        It takes in the following kwargs: 
            myArrayX, threshold, maxLenOfArrayX, numIterations, numAgents

        '''
    
    def __init__(self, myArrayX = 0, threshold = 0, maxLenOfArrayX = 0, numIterations = 30, numAgents = 50, **kwargs):        

        self.myArrayX = myArrayX
        self.threshold = threshold
        self.maxLenOfArrayX = maxLenOfArrayX
        self.numIterations = numIterations
        self.numAgents = numAgents


    def sdsStart(self):
        '''
            This function starts the following phases: 
                Initialisation, Test and Diffusion
            Returns: 
                myArrayX -- of type array
                model_ids -- of type array
            
        '''
        #transforming the data
        scaler.fit_transform(self.myArrayX) 
        
        #for storing IDs
        model_ids=[]
        
        while len(self.myArrayX) > self.maxLenOfArrayX:
            #################### INITIALISATION PHASE #######################

            #generating random id for picking a model
            id = np.random.randint(len(self.myArrayX), size = 1)
            
            model_ids.append(id[0])

            #taking a model from the search space
            model = self.myArrayX[id,:]
            
            #deleting the model from search space
            self.myArrayX = np.delete(self.myArrayX, id, axis = 0)
            
            #generatings ids for agents
            idx = np.random.randint(len(self.myArrayX), size = self.numAgents)
            
            #assigning agents from search space
            agents =  self.myArrayX[idx, :]
            
            #setting inital status for all agents as 'active' 
            agents_status = np.array(['active']).repeat(len(agents))
    
            ################## END OF INITIALISATION PHASE #########################
            
            for self.numIterations in range(0, self.numIterations):   
                ################### TEST PHASE ################################
                
                for i in range(len(agents)):
                    
                    #generating random index for comparing dimesions
                    j = np.random.randint(self.myArrayX.shape[1], size = 1)
                    
                    if agents[i][j-1] - model[0][j-1] > self.threshold: #checking if agent's jth dimension is within threshold of model's jth
                        
                        #resetting status as 'inactive' if it is greater than threshold
                        agents_status[i] = 'inactive' 
                
                ################# END OF TEST PHASE ##################################
                 
                
            ############### DIFFUSION PHASE ###################
                for i in range(len(agents_status)):
                    
                    if agents_status[i] == 'inactive':
                       
                        #generating a random number for inactive agent to pick another agent
                        random_pick = np.random.randint(len(agents), size = 1) 
                        
                        if agents_status[random_pick] == 'active':
                           
                            #if the picked agent is active, the inactive agent moves to active agent
                            agents[i] = agents[random_pick] 
                        
                        else:
                           
                            random_pick2=np.random.randint(len(myArrayX), size = 1)
                            
                            #if picked agent is inactive too, the inactive agent picks a random record from search space
                            agents[i] = myArrayX[random_pick2] 
            
            agent,counts = np.unique(agents, return_counts = True)
            
            for i in range(len(agent)):
               
                if counts[i] == max(counts):
                   
                    remove_id = np.where(np.equal(self.myArrayX, agents[i]).all(axis = 1) == True)[0][0]
                    
                    #deleting the record on where there are many agents from search space
                    self.myArrayX = np.delete(self.myArrayX, remove_id, axis = 0) 
        
        ################### END OF DIFFUSION PHASE ##################

        return self.myArrayX, model_ids #END OF sdsStart()
