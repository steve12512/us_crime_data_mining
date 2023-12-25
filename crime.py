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
    var = shootings2[shootings2['signs_of_mental_illness'] == True]

    percentage = (len(shootings2) / len(shootings)) * 100
    print('percentage of mentally ill', percentage)
    return var

def get_armed_mental_illness():
    #here we will be getting the percentage of shot suspects have have shown signs of mental illness whilst being armed
    #cope the original dataframe in order to do procedures
    shootings2 = shootings.copy()

    armed_mental = shootings2[(shootings2['signs_of_mental_illness'] == True) & (shootings2['arms_category'] == 'Guns')]
    armed_mental_percentage = len(armed_mental) / len(shootings2) * 100
    print('mentally ill and with real guns', armed_mental_percentage)
    return armed_mental


def get_mental_toygun():    
    #here we will find the number of shot suspects that were mentally ill and had only a toygun

    #copy the original dataframe
    shootings2 = shootings.copy()

    #find the mentally ill with toyguns
    toy_weapon = shootings2[(shootings2['signs_of_mental_illness'] == True) & (shootings2['armed'] == 'toy weapon')]
    print('mentally ill with toyguns',len(toy_weapon))

    return toy_weapon


def get_peace_sane_unarmed():
    #here we will explore how many suspects were shot even though they were unarmed and showed no sign of mental illness
    
    #cope the original dataframe
    shootings2 = shootings.copy()

    #find the number of shot people that were unarmed, unthreatening and not mentally unstable
    var = shootings2[ (shootings2['armed'] == 'unarmed') & (shootings2['threat_level'] == 'other') & (shootings2['signs_of_mental_illness'] == False)]
    
    #find the percentage of those, that were black or hispanic
    color = len (var[ (var['race']  == 'black')  | (var['race'] == 'Hispanic') ])  / len(var) *100

    print('length of unarmed,unthreatening,not mentally unstable is; ', len(var))
    print('of which, the ', color, ' were black and hispanic')
    print('we have not reached any conclusion that would indicate a racial bias based on the numbers.')

    return var

def get_color():
    #here we wil find how many people of color are part of the dataframe

    #first lets copy the original dataframe to operate upon
    shootings2 = shootings.copy()

    #only keep black and hispanic people
    var = shootings2[(shootings2['race'] == 'Black') | (shootings2['race'] == 'Hispanic')]
    percentage = len(var) / len(shootings2) * 100

    print('Black and Hispanic people make up ', percentage, ' of the dataset')
    print('Again we cannot reach any conclusion that indicates a racial motivation to lie within the general tendecy.')
    return var

def groupby_state_year():
    #in this function we will be using the groupby method on the state and year attributes. first obj is unordered,2nd is ordered by state and 3rd is ordered by date
    #group by state and year to see where most incidents occured

    state_year = df.groupby(['state', 'year']).size().reset_index(name='incidents_per_state_per_year')
    print(state_year)

    #take the above groupby and order by count descending, to see which state remains the most violent one
    stateord_year = state_year.sort_values(by = 'incidents_per_state_per_year', ascending = False)
    print(stateord_year.head(20))

    #this time, order by year
    state_yearord = state_year.sort_values(by = 'year', ascending= False)
    print(state_yearord)

    return state_year, stateord_year, state_yearord


#START OF PROGRAM


#START OF ETL
#print(shootings.head(5))

#convert date into a datetime object
shootings['date'] = pd.to_datetime(shootings['date'])
print(shootings.dtypes)

#keep a copy of the dataframe to operate upob
df = shootings.copy()

#add a year column
df['year'] = pd.to_datetime(df['date']).dt.year


#START OF AGGREGATION
mental = get_mental_illness()
mental_armed = get_armed_mental_illness()
mental_toygun = get_mental_toygun()
peace_sane_unarmed = get_peace_sane_unarmed()
color = get_color()


#START OF CREATING THE DATACUBE

#state and year groupbys
state_year, stateORD_year, state_yearORD = groupby_state_year()


