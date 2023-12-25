#import what you have to
import pandas as pd 
import numpy as np 

#read the file
shootings = pd.read_csv('shootings.csv')

#defining functions we will be using in order to create a datacube

def get_mental_illness():
    #used for finding the percentage of police shootings where the suspect has shown signs of mental illness
    #copy the file into the one which we will be using for procedures
    shootings2 = shootings.copy()

    #find the percentage of shot suspects that have shown signs of mental illnes 
    shootings2 = shootings2[shootings2['signs_of_mental_illness'] == True]

    #create a new dataframw which we will be using for our datacube
    percentage = pd.DataFrame(columns = ['shown_signs_of_mental_illness'])
    percentage = (len(shootings2) / len(shootings)) * 100
    print('percentage of mentally ill', percentage)
    return percentage

def get_armed_mental_illness():
    #here we will be getting the percentage of shot suspects have have shown signs of mental illness whilst being armed
    #cope the original dataframe in order to do procedures
    shootings2 = shootings.copy()

    armed_mental = shootings2[(shootings2['signs_of_mental_illness'] == True) & (shootings2['arms_category'] == 'Guns')]
    armed_mental = len(armed_mental) / len(shootings2) * 100
    print('mentally ill and with real guns', armed_mental)
    return armed_mental




def get_mental_toygun():
    #here we will find the number of shot suspects that were mentally ill and had only a toygun

    #copy the original dataframe
    shootings2 = shootings.copy()

    #find the mentally ill with toyguns
    toy_weapon = shootings2[(shootings2['signs_of_mental_illness'] == True) & (shootings2['armed'] == 'toy weapon')]
    print('mentally ill with toyguns',len(toy_weapon))

    return len(toy_weapon)



#START OF ETL
#print(shootings.head(5))




#START OF AGGREGATION
mental_percentage = get_mental_illness()
mental_armed_percentage = get_armed_mental_illness()
mental_toygun = get_mental_toygun()