from matplotlib import image
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.patches as patches
import scipy.stats as sp
import random

MAXIMUM_GENERATIONS = 500 
THRESHOLD = 1
NUMBERS = 1000 #Total Population generated initially

population=[]   # To store possible solutions of population in each generation.
fitness=[]      # To store Fitness of each generation with coordinates of possible solutions.
max_fitness=[]  # To store Maximum fitness value with cordinates of best solutions of each generation.            

avg_max_fitness=[] # Will be used to store Average fitness of each generation for ploting graph
gen_max_fitness=[] # Will be used to store max fitness of each generation for ploting graph
generations=[] #Will record the generation number in this list for ploting graph
genCounter = 0 # This will count generations.

class Template:
    def __init__(self):
        # READ IMAGES and store them in 2D arraylen(self.image1[0]
        # Image 1
        self.image1 = image.imread('group.jpg')
        print(self.image1 )
        print(self.image1.shape)
        print(f"rows :{len(self.image1)}")
        print(f" Columns : {len(self.image1[0])}")

        # Image 2
        self.image2 = image.imread('boothi.jpg')
        print(self.image2 )
        print(self.image2.shape)

        print("-----------------------------------------")
        
    # Initialize Population
    def InitializePopulation(self):
        for num in range(NUMBERS):
            x=random.randint(0,len(self.image1[0])-(len(self.image2[0]))) # 
            y=random.randint(0,len(self.image1)-(len(self.image2)))
            coordinates=[x,y]
            population.append(coordinates)
        #print(population)


    def Correlation(self,x,y):
        """
        We are using the built in function for correlation.
        It returns numbers between -1 to 1.  
        """
        arr=self.image1[y:y+len(self.image2),x:x+len(self.image2[0])] # slice array(Possible solution) from bigger image
        cor=sp.kendalltau(self.image2,arr).correlation
        return round(cor,2)


    def Fitness_saver(self): #Converter & Saver 
        """This funtion is extracting population [x,y] coordinates from 
        population list(population = [[x1, y1], [x2, y2], ...[xn, yn]]) and storing/saving 
        them along side fitness value in fitness list(fitness = [[[x1, y1], f1], [[x2, y2], f2],
        ..., [[xn, yn], fn]]]) 
        """
        for pop in population:
            x=pop[0]
            y=pop[1]
            fit=self.Correlation(x,y)
            indvidual=[pop,fit]
            fitness.append(indvidual)   
        
        global genCounter # This variable is storing the count of Generations
        generations.append(genCounter) #storing gen counter in array(generations) to plot graph later
        genCounter += 1
        #print(fitness)
    
    def Sort_Fitness(self):
        """
        The function is sorting possible solutions in the generated 
        population based on their fitness value.  
        """
        #print("\nSORTED\n")
        fitness.sort(key= lambda x: x[1], reverse=True)   # sort the array according to fitness in descending
        max_fitness.append(fitness[0])
        #print(max_fitness)
        gen_max_fitness.append(fitness[0][1])

        # calculating average fitness per Generation
        sum = 0
        for fit in fitness:
            sum = sum + fit[1]        
        Avg = round((sum / NUMBERS),4) 
        avg_max_fitness.append(Avg)
        print("\n\n<----------------->")
        print(f"Average Fitness = {Avg}")
    
    def Sort_MaxFitness(self):
        """
        This function is sorting best fits of each generations 
        stored in maxfitness on their fitness value.  
        """
        max_fitness.sort(key= lambda x:x[1], reverse=True)
        print("\n\nMaximum Match(Maximum Fitness) = {0} \n".format(max_fitness[0][1]) )
        print("Maximum Point = {0} \n\n".format(max_fitness[0][0]))
        print("Gen No = {0}".format(genCounter))

        #print("*********Max Fitness of All Generations*****")
        #print(max_fitness)
        #print("<-----------------\n\n")


    #----------------------------------------------------------------------
    #----------------------------------------------------------------------
    #----------------------------------------------------------------------
    
    def converter_for_CrossOver(self): #Converter
        """This funtion is performing opposite operation of our function named Fitness_saver
        because it is extracting population [x,y] coordinates from
        fitness list(fitness = [[[x1, y1], f1], [[x2, y2], f2], ..., [[xn, yn], fn]]])
        and stores it in list named Array(Array = [[x1, y1], [x2, y2], ...[xn, yn]])
        and then return's that Array which in turn will be used by our crossover function.  
        """

        Array = []
        #print(fitness)
        #print("<===================================\n\n\n")
        for element in fitness:
            Array.append(element[0])
        #print(Array)
        #print("<===================================\n\n\n")
        fitness.clear()
        return Array

    def crossover(self,array):
        population.clear() 
        for row in range(0,NUMBERS,2):
            x=array[row][0]
            y=array[row][1]
            x1=np.binary_repr(x,width=11)  # 11 digit binary
            y1=np.binary_repr(y,width=11)
            # converting the string to list for performing the crossover 
            x1 = list(x1) 
            y1 = list(y1)
            specie_1=x1+y1       #concatinate x1 and y1

            x=array[row+1][0]
            y=array[row+1][1]
            # conversion into binary
            x2=np.binary_repr(x,width=11)
            y2=np.binary_repr(y,width=11)
            # converting the string to list for performing the crossover 
            x2 = list(x2) 
            y2 = list(y2)
            specie_2=x2+y2     #concatinate x2 and y2

            # generating the random point to perform crossover 
            k = random.randint(1,21) 
            
            
            # interchanging the values randomly
            for i in range(k, len(specie_1)): 
                specie_1[i], specie_2[i] = specie_2[i], specie_1[i] #swaping 
            specie_1 = ''.join(specie_1) 
            specie_2 = ''.join(specie_2)
            
            """
            # interchanging values from index 7 to 15 index--- total size will be 11+11 = 22 
            
            for i in range(7, 15): 
                specie_1[i], specie_2[i] = specie_2[i], specie_1[i] #swaping 
            specie_1 = ''.join(specie_1) 
            specie_2 = ''.join(specie_2)
            """
            
            """
            # interchanging values from index 6 to 21 index--- total size will be 11+11 = 22 
            for i in range(4, 21): 
                specie_1[i], specie_2[i] = specie_2[i], specie_1[i] #swaping 
            specie_1 = ''.join(specie_1) 
            specie_2 = ''.join(specie_2)
            """

            x1=specie_1[0:12]
            y1=specie_1[12:23] 
            x2=specie_2[0:12]
            y2=specie_2[12:23]
            
            #conversion back to decimal
            int_x1=int(x1,2)
            int_y1=int(y1,2)
            int_x2=int(x2,2)
            int_y2=int(y2,2)
            
            ##################################################
            
            #Mutation

            #The following bounds will make sure that our new generated
            # children remain within the boundries of bigger image            
            height_upperBound = len(self.image1)-len(self.image2)      
            width_upperBound = len(self.image1[0])-len(self.image2[0])

            # if values remain same after crossover mutate them

            if int_y1==int_y2 and int_y1<(height_upperBound):
                int_y1+=1
            if int_x1==int_x2 and int_x1<(width_upperBound):
                int_x1+=1
            # Checking Corner cases
            if int_x1>(width_upperBound):
                int_x1=random.randint(1,width_upperBound)
                #int_x1=len(self.image1[0])//2
            if int_x2>(width_upperBound):
                int_x2=random.randint(1,width_upperBound)
            if int_y1>(height_upperBound):
                int_y1=random.randint(1,height_upperBound)
            if int_y2>(height_upperBound):
                int_y2=random.randint(1,height_upperBound)
            
            population.append([int_x1,int_y1])
            population.append([int_x2,int_y2])
    
    # Functioin to iterate Generations
    def StoppCritaria(self):
        self.Fitness_saver()
        self.Sort_Fitness()
        array=self.converter_for_CrossOver()
        self.crossover(array)

        for i in range(MAXIMUM_GENERATIONS):
            self.Sort_MaxFitness()
            if max_fitness[0][1]<THRESHOLD:   #Stop when reached the required threshold
                self.Fitness_saver()
                self.Sort_Fitness()
                array=self.converter_for_CrossOver()
                self.crossover(array)
            else:
                # If fitness meet with threshold
                print(f"Generation No:{i}")
                break

    # Plot Rectangle on which Fitness is Highest
    def Plot_Rectangle(self):
        max_match=max_fitness[0][0]
        #max_match=fitness[0][0]
        x=max_match[0]
        y=max_match[1]
        axiss=plt.gca()
        rect=patches.Rectangle((x,y),
        len(self.image2[0]),
        len(self.image2),
        linewidth=1,
        edgecolor='cyan',
        fill=False)
        axiss.add_patch(rect)
        plt.imshow(self.image1,cmap='gray')
        plt.show()
        #plt.imshow(self.image2,cmap='gray')
        #plt.show()

        """
        #This for loop can be used to draw 
        # multiple triangles above a certian threshold
        for fitness in max_fitness:  
            if fitness[1] >= THRESHOLD - 0.2:
                x = fitness[0][0]
                y = fitness[0][1]
                axiss=plt.gca()
                rect=patches.Rectangle((x,y),
                len(self.image2[0]),
                len(self.image2),
                linewidth=1,
                edgecolor='cyan',
                fill=False)
                axiss.add_patch(rect)

        plt.imshow(self.image1,cmap='gray')
        plt.show()
        """
    
    def Plot_fitness(self):
        #This function Plotting Gitness vs Generations Graph
        plt.figure()
        plot1, = plt.plot(generations, gen_max_fitness)
        plot2, = plt.plot(generations, avg_max_fitness)
        plt.title("Graph")
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.legend([plot1,plot2],["Max Fitess per Generation", "Avg Fitness per Generation"])
        plt.show()

def main():
    t1=Template()
    t1.InitializePopulation()
    t1.StoppCritaria()
    t1.Sort_MaxFitness() # Sorting max fitnesses of each generation
    t1.Plot_Rectangle() # Drawing Rectangle
    t1.Plot_fitness() #Ploting Fitness vs Generations Graph

if __name__=="__main__":
    main()