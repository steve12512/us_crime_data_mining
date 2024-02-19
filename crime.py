#import what you have to
import pandas as pd 
import numpy as np 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from kmodes.kprototypes import KPrototypes
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score



#display max columns
pd.set_option('display.max_columns', None)  # Set to None to display all columns


#read the files
shootings = pd.read_csv('shootings.csv')
population = pd.read_csv('us_pop_by_state.csv')


#defining functions we will be using in order to create a datacube
#the functions below return dataframe objects

def calculate_severity(row):
    #in this function we will be calculating the severity of each listing, based on how badly we perceive of the suspect's behavior. then we will append it to the original dataframe
    armed_values = {
    'air conditioner': 10,
    'air pistol': 20,
    'ax': 50,
    'barstool': 15,
    'baseball bat': 30,
    'baseball bat and bottle': 35,
    'baseball bat and fireplace poker': 40,
    'baseball bat and knife': 50,
    'baton': 25,
    'bayonet': 55,
    'BB gun': 20,
    'BB gun and vehicle': 30,
    'bean-bag gun': 30,
    'beer bottle': 20,
    'blunt object': 25,
    'bow and arrow': 40,
    'box cutter': 30,
    'brick': 40,
    'car, knife, and mace': 60,
    'carjack': 40,
    'chain': 35,
    'chainsaw': 60,
    'chair': 20,
    "contractor's level": 25,
    'cordless drill': 30,
    'crossbow': 50,
    'crowbar': 40,
    'fireworks': 30,
    'flagpole': 25,
    'flashlight': 15,
    'garden tool': 25,
    'glass shard': 20,
    'grenade': 80,
    'gun': 70,
    'gun and car': 75,
    'gun and knife': 75,
    'gun and sword': 80,
    'gun and vehicle': 80,
    'guns and explosives': 90,
    'hammer': 30,
    'hand torch': 25,
    'hatchet': 40,
    'hatchet and gun': 50,
    'ice pick': 35,
    'incendiary device': 80,
    'knife': 50,
    'lawn mower blade': 45,
    'machete': 55,
    'machete and gun': 65,
    'meat cleaver': 45,
    'metal hand tool': 30,
    'metal object': 25,
    'metal pipe': 35,
    'metal pole': 40,
    'metal rake': 30,
    'metal stick': 25,
    'motorcycle': 50,
    'nail gun': 45,
    'oar': 25,
    'pellet gun': 20,
    'pen': 10,
    'pepper spray': 15,
    'pick-axe': 50,
    'piece of wood': 25,
    'pipe': 35,
    'pitchfork': 40,
    'pole': 30,
    'pole and knife': 40,
    'rock': 30,
    'samurai sword': 60,
    'scissors': 20,
    'screwdriver': 25,
    'sharp object': 30,
    'shovel': 35,
    'spear': 40,
    'stapler': 20,
    'straight edge razor': 30,
    'sword': 60,
    'Taser': 40,
    'toy weapon': 10,
    'unarmed': 5,
    'unknown': 5,
    'vehicle': 40,
    'vehicle and gun': 70,
    'vehicle and machete': 60,
    'walking stick': 15,
    'wasp spray': 10,
    'wrench': 30
}
    mental_illness_value = 40 if row['signs_of_mental_illness'] else 0

    threat_level_values = {'attack': 70, 'undetermined': 20, 'other': 10}

    armed_value = armed_values.get(row['armed'], 0)
    threat_level_value = threat_level_values.get(row['threat_level'], 0)

    # Calculate the severity based on the weighted sum of values
    severity = (0.45 * armed_value + 0.2 * mental_illness_value + 0.35 * threat_level_value)
    return severity



def add_raceCount(shootings):
    #here we will be adding a new column to our dataframe, that counts the incidents per race

    #copy the shootings dataframe
    df = shootings.copy()

    #group by race and count its incidents
    df = df.groupby('race').size().reset_index(name = 'race_incidents')

    #append our new column to the shootings dataframe
    shootings = pd.merge(shootings, df, left_on='race', right_on='race', how='left')

    return shootings

    



def get_mental_illness():
    #used for getting a dataframe of the police shootings where the suspect has shown signs of mental illness

    #copy the file into the one which we will be using for procedures
    shootings2 = shootings.copy()

    #find the percentage of shot suspects that have shown signs of mental illnes 
    var = shootings2[shootings2['signs_of_mental_illness'] == True]

    percentage = (len(var) / len(shootings)) * 100
    #print('percentage of mentally ill', percentage)
    return var

def get_armed_mental_illness():
    #here we will be getting a dataframe of shot suspects have have shown signs of mental illness whilst being armed

    #copy the original dataframe in order to do procedures
    shootings2 = shootings.copy()

    armed_mental = shootings2[(shootings2['signs_of_mental_illness'] == True) & (shootings2['arms_category'] == 'Guns')]
    armed_mental_percentage = len(armed_mental) / len(shootings2) * 100
    #print('mentally ill and with real guns', armed_mental_percentage)
    return armed_mental


def get_mental_toygun():    
    #here we will get a dataframe consisting of shot suspects that were mentally ill and had only a toygun

    #copy the original dataframe
    shootings2 = shootings.copy()

    #find the mentally ill with toyguns
    toy_weapon = shootings2[(shootings2['signs_of_mental_illness'] == True) & (shootings2['armed'] == 'toy weapon')]
    #print('mentally ill with toyguns',len(toy_weapon))

    return toy_weapon


def get_peace_sane_unarmed():
    #here we will explore how many suspects were shot even though they were unarmed and showed no sign of mental illness
    
    #cope the original dataframe
    shootings2 = shootings.copy()

    #find the number of shot people that were unarmed, unthreatening and not mentally unstable
    var = shootings2[ (shootings2['armed'] == 'unarmed') & (shootings2['threat_level'] == 'other') & (shootings2['signs_of_mental_illness'] == False)]
    
    #find the percentage of those, that were black or hispanic
    color = len (var[ (var['race']  == 'black')  | (var['race'] == 'Hispanic') ])  / len(var) *100

    #print('length of unarmed,unthreatening,not mentally unstable is; ', len(var))
    #print('of which, the ', color, ' were black and hispanic')
    #print('we have not reached any conclusion that would indicate a racial bias based on the numbers.')

    return var

def get_color():
    #here we wil find how many people of color are part of the dataframe

    #first lets copy the original dataframe to operate upon
    shootings2 = shootings.copy()

    #only keep black and hispanic people
    var = shootings2[(shootings2['race'] == 'Black') | (shootings2['race'] == 'Hispanic')]
    percentage = len(var) / len(shootings2) * 100

    #print('Black and Hispanic people make up ', percentage, ' of the dataset')
    #print('Again we cannot reach any conclusion that indicates a racial motivation to lie within the general tendecy.')
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
   
    #print('\n police shootings grouped by; state, year, sorted per 100k of its population\n', state_year.head(30))
    #print('\npolice shootings grouped by;state, year, sorted by number of state incidents \n', stateord_year.head(30))
    #print('\npolice shootings grouped by; state, year,sorted by year and then sorted by incidents per 100k of each state\'s population  \n', state_yearord.head(30))
    state_yearord.to_csv('state_per_100k.csv')

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
    #print('\n percentage of female shootings is; ', len(var) / len(shootings) * 100, '%')
    #print('\n of which ', len(var[var['armed'] == 'gun']) / len(var) * 100, ' were armed with guns \n ')
    #print('whereas men, whose shootings make up 95,46%\ of the dataset, had an armed with guns percentage of; ', len(shootings[(shootings['gender'] == 'M') & (shootings['armed'] == 'gun')]) / len(shootings[(shootings['gender'] == 'M')]) * 100, '%\n')
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
    #print('\ngrouped by race, state, and whether or not the suspect attempted to flee \n ', race_state_flee.head(20))

    return race_state_flee

def groupby_mental_threat_arms():
    #this function returns a groupby whether or not the suspect had shown signs of mentall illness, was a threat and whether or not he/she was armed
    
    mental_threat_arms = df.groupby(['signs_of_mental_illness', 'threat_level', 'arms_category'])
    #print('\n grouped by whether or not the suspcet had shown signs of mental illness, his threat level and his arms category', mental_threat_arms.head(30))

    return mental_threat_arms


def clusters():
    #here we will proceed with our creation of clusters

    #first make a copy of our dataframe
    df = shootings.copy()
    df = df.drop(columns=['id','name'],errors='ignore')

    #encode string variables
    df = pd.get_dummies(df, columns=['race', 'state', 'signs_of_mental_illness', 'threat_level', 'flee', 'body_camera', 'arms_category'])

    #standardize our data
    scaler = StandardScaler()
    df['severity_scaled'] = scaler.fit_transform(df[['severity']])

    #extract some date specific attributes
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek

    #select the data upon which we will be running our algorithm
    #of course, we have to reduce a lot of that data

    selected_features = df[['severity_scaled', 'race_Asian', 'race_Black', 'race_Hispanic', 'race_Native',
       'race_Other', 'race_White' ]]

    kproto = KPrototypes(n_clusters=3, init='Cao', verbose=2)
    df['cluster'] = kproto.fit_predict(selected_features.values, categorical=list(range(4, len(selected_features.columns))))


    # Assuming df_clustered is your DataFrame with 'cluster' column from KPrototypes
    # Exclude 'date' and 'cluster' columns from PCA
    pca_columns = [col for col in selected_features.columns if col not in ['date', 'cluster']]
    pca = PCA(n_components=7)
    reduced_features = pca.fit_transform(selected_features[pca_columns])

    # Scatter plot with legend
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=df['cluster'], cmap='viridis')
    plt.title('KPrototypes Clusters (2D)')  
    plt.xlabel('standardized_severity')
    plt.ylabel('race')


    # Add legend
    legend_labels = [f'Cluster {i}' for i in range(max(df['cluster']) + 1)]
    plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels, title='Clusters', loc='lower right')

    plt.show()


def regression_df(shootings):
    #here we will be modifying the dataframe we need for our regression model
     #here we are going to run a regression model for our code

    #first let us make a copy of our dataframe, to operate upon
    reg_df = shootings.copy()

    #group by race, year and count

    reg_df = df.groupby(['race', 'year']).agg({'id' : 'count', 'severity' : 'mean'}).reset_index()
    reg_df = reg_df.rename(columns={'id': 'incidents_per_race_per_year'})

    return reg_df










def regression(a):
   #here we will be running our regression model
    
   # Extracting features (X) and target variable (y)
    X = a[['year', 'severity']]
    y = a['incidents_per_race_per_year']

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Creating and fitting the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Making predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluating the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')

    ## Visualizing the regression line along with the scatter plot
    plt.scatter(X_test['year'], y_test, color='black', label='Actual data')
    plt.scatter(X_test['year'], y_pred, color='blue', linewidth=3, label='Predicted data')
    plt.plot(X_test['year'], y_pred, color='red', linewidth=2, label='Regression line')
    plt.xlabel('Year')
    plt.ylabel('Incidents')
    plt.title('Linear Regression')
    plt.legend()
    plt.show()

#START OF PROGRAM

#convert date into a datetime object
shootings['date'] = pd.to_datetime(shootings['date'])

#keep a copy of the dataframe to operate upob
df = shootings.copy()

#add a year column
df['year'] = pd.to_datetime(df['date']).dt.year

#mental == df where people showed signs of mental illness
#armed == df where people were armed
#color == df where people are black or hispanic

#add counts per race





#START OF AGGREGATION
mental = get_mental_illness()
mental_armed = get_armed_mental_illness()
mental_toygun = get_mental_toygun()
peace_sane_unarmed = get_peace_sane_unarmed()
color = get_color()
female = get_female()


#state and year groupbys
state_year, stateORD_year, state_yearORD = groupby_state_year()
race_state_flee = groupby_race_state_flee()
mental_threat_arms = groupby_mental_threat_arms()


#edit date
shootings['day_of_month'] = shootings['date'].dt.day
shootings['day_of_week_name'] = shootings['date'].dt.day_name()
shootings['day_of_week_number'] = shootings['date'].dt.dayofweek
shootings['month_number'] = shootings['date'].dt.month
shootings['month_name'] = shootings['date'].dt.month_name()
shootings['year'] = shootings['date'].dt.year



shootings['severity'] = shootings.apply(calculate_severity, axis=1)
shootings.to_excel('severity.xlsx', index =False)
shootings.to_csv('shootings2.csv', index = False)


shootings = add_raceCount(shootings)


reg_df = regression_df(shootings)
reg_df = reg_df[reg_df['year'] != 2020]





black = reg_df[reg_df['race'] == 'Black']
print(black.head(10))
print(black.dtypes)

regression(black)