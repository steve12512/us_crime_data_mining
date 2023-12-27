#import what you have to
import pandas as pd 
import numpy as np 

#display max columns
pd.set_option('display.max_columns', None)  # Set to None to display all columns


#read the files
shootings = pd.read_csv('shootings.csv')
population = pd.read_csv('us_pop_by_state.csv')


#defining functions we will be using in order to create a datacube
#the functions below return dataframe objects

def get_mental_illness():
    #used for getting a dataframe of the police shootings where the suspect has shown signs of mental illness

    #copy the file into the one which we will be using for procedures
    shootings2 = shootings.copy()

    #find the percentage of shot suspects that have shown signs of mental illnes 
    var = shootings2[shootings2['signs_of_mental_illness'] == True]

    percentage = (len(var) / len(shootings)) * 100
    print('percentage of mentally ill', percentage)
    return var

def get_armed_mental_illness():
    #here we will be getting a dataframe of shot suspects have have shown signs of mental illness whilst being armed

    #copy the original dataframe in order to do procedures
    shootings2 = shootings.copy()

    armed_mental = shootings2[(shootings2['signs_of_mental_illness'] == True) & (shootings2['arms_category'] == 'Guns')]
    armed_mental_percentage = len(armed_mental) / len(shootings2) * 100
    print('mentally ill and with real guns', armed_mental_percentage)
    return armed_mental


def get_mental_toygun():    
    #here we will get a dataframe consisting of shot suspects that were mentally ill and had only a toygun

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

#the functions below return dataframes that will be used for creating the datacube

def create_dictionaries():
    #this function creates a dictionary to map state code to state population and another to map state code to state name
    a = dict(zip(population['state_code'], population['2020_census']))
    b = dict(zip(population['state_code'], population['state']))

    return a, b



def groupby_state_year():
    #in this function we will be using the groupby method on the state and year attributes. first obj is unordered,2nd is ordered by state and 3rd is ordered by date
    #group by state and year to see where most incidents occured
    #BEFORE doing the above, we want to map each state to its population and then its indicdents per 100k of its population
    #firstly we have to do the mapping

    shootings2 = df.copy()
    shootings2 = pd.merge(shootings2, population, left_on = 'state', right_on = 'state_code', how= 'left')
    state_year = shootings2.groupby(['state_x', 'year']).size().reset_index(name='incidents_per_state_per_year')

    #get a dictionary to map state codes to state populations, and one to map state_codes to state names
    population_dictionary, state_dictionary = create_dictionaries()

    #then, use the population dict and append the values to a new column
    state_year['state_pop'] = state_year['state_x'].map(population_dictionary)

    #now we want to take the above groupby and calculate the incidents per 100k of its population
    state_year['per 100k of its population'] = state_year['incidents_per_state_per_year'] / state_year['state_pop'] * 100000

    #take the above groupby and order by count descending, to see which state remains the most violent one
    stateord_year = state_year.sort_values(by = 'incidents_per_state_per_year', ascending = False)

    #this time, order by year
    state_yearord =  state_year.sort_values(by =['year','per 100k of its population'], ascending= False)

    #sort values based on incidents per 100k inhabitants
    state_year = state_year.sort_values(by = 'per 100k of its population', ascending = False)



    #map state codes to state names so we have more beuatiful data
    map_state_codes(state_year,stateord_year, state_yearord, state_dictionary)
    modify_dataframes(state_year, stateord_year, state_yearord)
   
    print('\n police shootings grouped by; state, year, sorted per 100k of its population\n', state_year.head(30))
    print('\npolice shootings grouped by;state, year, sorted by number of state incidents \n', stateord_year.head(30))
    print('\npolice shootings grouped by; state, year,sorted by year and then sorted by incidents per 100k of each state\'s population  \n', state_yearord.head(30))

    return state_year, stateord_year, state_yearord


def map_state_codes(a,b,c, f):
    #this function maps state codes to state years for the 3 different groupbies dataframes
    a['state_x'] = a['state_x'].map(f)
    b['state_x'] = b['state_x'].map(f)
    c['state_x'] = c['state_x'].map(f)
    return None

def get_female():
    #this function returns a dataframe where the victims were women
    var = shootings[shootings['gender'] == 'F']
    print('\n percentage of female shootings is; ', len(var) / len(shootings) * 100, '%')
    print('\n of which ', len(var[var['armed'] == 'gun']) / len(var) * 100, ' were armed with guns \n ')
    print('whereas men, whose shootings make up 95,46%\ of the dataset, had an armed with guns percentage of; ', len(shootings[(shootings['gender'] == 'M') & (shootings['armed'] == 'gun')]) / len(shootings[(shootings['gender'] == 'M')]) * 100, '%\n')
    return var


def modify_dataframes(a,b,c):
    #drops unnecessary columns for each dataframe
    a.drop(['incidents_per_state_per_year', 'state_pop'], axis = 1, inplace = True)
    b.drop('per 100k of its population', axis = 1, inplace = True)
    c.drop(['state_pop','incidents_per_state_per_year'], axis =1, inplace = True)
    
    return None

def groupby_race_state_flee():
    #this functions does a group by race, state and whether or not the suspect was fleeing
    race_state_flee = df.groupby(['race','state','flee'])
    print('\ngrouped by race, state, and whether or not the suspect attempted to flee \n ', race_state_flee.head(20))

    return race_state_flee

def groupby_mental_threat_arms():
    #this function returns a groupby whether or not the suspect had shown signs of mentall illness, was a threat and whether or not he/she was armed
    
    mental_threat_arms = df.groupby(['signs_of_mental_illness', 'threat_level', 'arms_category'])
    print('\n grouped by whether or not the suspcet had shown signs of mental illness, his threat level and his arms category', mental_threat_arms.head(30))

    return mental_threat_arms




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

#mental == df where people showed signs of mental illness
#armed == df where people were armed
#color == df where people are black or hispanic


#START OF AGGREGATION
mental = get_mental_illness()
mental_armed = get_armed_mental_illness()
mental_toygun = get_mental_toygun()
peace_sane_unarmed = get_peace_sane_unarmed()
color = get_color()
female = get_female()

#START OF CREATING THE DATACUBE

#state and year groupbys
state_year, stateORD_year, state_yearORD = groupby_state_year()
race_state_flee = groupby_race_state_flee()
mental_threat_arms = groupby_mental_threat_arms()


